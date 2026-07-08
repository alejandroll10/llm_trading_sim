"""
Thread-safe, per-run accumulator for realized LLM token usage and API cost (#104).

Design mirrors LoggingService: a process-global class with class-level state that
is reset() at the start of every run. Both the trading-agent call path
(agents/LLMs/services/llm_services.py) and the news generator
(services/news_service.py) write directly into it, so a single tracker sees every
API call in a run even though each LLMAgent owns its own LLMService instance and
decisions are collected on a ThreadPoolExecutor. All mutation goes through a lock
because agent decisions run in parallel.

At the end of a run BaseSimulation writes data/llm_usage.csv (per-call rows) and
stashes summary() into the run's metadata.json; run_sweep / aggregate_sweep roll
those per-cell summaries up to sweep totals.
"""

import csv
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.model_pricing import active_backend, compute_cost, is_known_model

# Per-call CSV columns, in order.
CSV_FIELDS = [
    "round", "agent_id", "model", "call_type",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "cost_usd", "latency_s", "attempts", "success",
]

# Scalar fields summed when rolling per-run summaries up to a sweep total.
_SUMMABLE = [
    "calls", "successful_calls", "failed_calls", "attempts",
    "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd",
]


def _empty_totals() -> Dict[str, Any]:
    return {k: 0 for k in _SUMMABLE}


def _add_summable(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k in _SUMMABLE:
        dst[k] = dst.get(k, 0) + (src.get(k, 0) or 0)


def _with_averages(b: Dict[str, Any]) -> Dict[str, Any]:
    calls = b.get("calls", 0)
    b["cost_usd"] = round(b.get("cost_usd", 0.0), 6)
    b["avg_tokens_per_call"] = round(b["total_tokens"] / calls, 1) if calls else 0.0
    b["avg_prompt_tokens_per_call"] = round(b["prompt_tokens"] / calls, 1) if calls else 0.0
    b["avg_completion_tokens_per_call"] = round(b["completion_tokens"] / calls, 1) if calls else 0.0
    return b


def aggregate_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll a list of per-run summary() dicts up into one combined summary.

    Shared by run_sweep (in-process, from simulation objects) and aggregate_sweep
    (from manifest cell records), so sweep totals are computed one way. Ignores
    None/empty entries (runs with no LLM calls).
    """
    totals = _empty_totals()
    by_model: Dict[str, Dict[str, Any]] = {}
    by_call_type: Dict[str, Dict[str, Any]] = {}
    unpriced: set = set()
    backends: set = set()
    runs_with_usage = 0

    for s in summaries:
        if not s or not s.get("calls"):
            continue
        runs_with_usage += 1
        _add_summable(totals, s)
        for m, mb in (s.get("by_model") or {}).items():
            _add_summable(by_model.setdefault(m, _empty_totals()), mb)
        for t, tb in (s.get("by_call_type") or {}).items():
            _add_summable(by_call_type.setdefault(t, _empty_totals()), tb)
        unpriced.update(s.get("unpriced_models") or [])
        if s.get("pricing_backend"):
            backends.add(s["pricing_backend"])

    _with_averages(totals)
    for b in by_model.values():
        _with_averages(b)
    for b in by_call_type.values():
        _with_averages(b)

    totals["by_model"] = by_model
    totals["by_call_type"] = by_call_type
    totals["unpriced_models"] = sorted(unpriced)
    # Usually one backend across a sweep; join if a sweep somehow mixed backends.
    totals["pricing_backend"] = "+".join(sorted(backends)) if backends else active_backend()
    totals["runs_with_usage"] = runs_with_usage
    return totals


class UsageTracker:
    """Process-global, thread-safe LLM usage accumulator (reset per run)."""

    _lock = threading.Lock()
    _records: List[Dict[str, Any]] = []

    @classmethod
    def reset(cls) -> None:
        """Clear all records. Call once at the start of each simulation run."""
        with cls._lock:
            cls._records = []

    @classmethod
    def record(
        cls,
        *,
        round_number: int,
        agent_id: str,
        model: str,
        call_type: str = "decision",
        prompt_tokens: Optional[int] = 0,
        completion_tokens: Optional[int] = 0,
        total_tokens: Optional[int] = None,
        latency_s: float = 0.0,
        attempts: int = 1,
        success: bool = True,
    ) -> None:
        """Record a single LLM API call (one row of llm_usage.csv).

        `attempts` counts every try including retries, not just the successful
        one. Token counts may be 0 when the endpoint omits usage or the call
        failed before returning any (a failed call is still recorded, with
        success=False, so the realized-vs-estimated call count reconciles).
        """
        pt = int(prompt_tokens or 0)
        ct = int(completion_tokens or 0)
        tt = int(total_tokens) if total_tokens is not None else pt + ct
        rec = {
            "round": round_number,
            "agent_id": agent_id,
            "model": model,
            "call_type": call_type,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "cost_usd": round(compute_cost(model, pt, ct), 6),
            "latency_s": round(float(latency_s), 3),
            "attempts": int(attempts),
            "success": bool(success),
        }
        with cls._lock:
            cls._records.append(rec)

    @classmethod
    def records(cls) -> List[Dict[str, Any]]:
        """Snapshot copy of all recorded calls."""
        with cls._lock:
            return list(cls._records)

    @classmethod
    def summary(cls) -> Dict[str, Any]:
        """Aggregate all recorded calls into run totals + per-model breakdown.

        Returned dict is JSON-serializable and safe to drop into metadata.json.
        """
        recs = cls.records()

        totals = _empty_totals()
        by_model: Dict[str, Dict[str, Any]] = {}
        by_call_type: Dict[str, Dict[str, Any]] = {}
        unpriced: set = set()

        for r in recs:
            row = {
                "calls": 1,
                "successful_calls": 1 if r["success"] else 0,
                "failed_calls": 0 if r["success"] else 1,
                "attempts": r["attempts"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["total_tokens"],
                "cost_usd": r["cost_usd"],
            }
            _add_summable(totals, row)
            _add_summable(by_model.setdefault(r["model"], _empty_totals()), row)
            _add_summable(by_call_type.setdefault(r["call_type"], _empty_totals()), row)
            if r["total_tokens"] and not is_known_model(r["model"]):
                unpriced.add(r["model"])

        _with_averages(totals)
        for b in by_model.values():
            _with_averages(b)
        for b in by_call_type.values():
            _with_averages(b)

        totals["by_model"] = by_model
        totals["by_call_type"] = by_call_type
        # Which backend's price table produced cost_usd (deepinfra vs self_hosted),
        # so the dollar figure is auditable -- the same model is free self-hosted
        # but billed on DeepInfra.
        totals["pricing_backend"] = active_backend()
        # Models that had real token usage but no price-table entry: their dollar
        # figure is understated (counted as 0). Surface them so it is not silent.
        totals["unpriced_models"] = sorted(unpriced)
        return totals

    @classmethod
    def save_csv(cls, path) -> Optional[Path]:
        """Write per-call rows to `path`. Returns the path, or None if no calls.

        Dependency-free (stdlib csv). No file is written when a run made no LLM
        calls (e.g. an all-deterministic scenario), to avoid empty artifacts.
        """
        recs = cls.records()
        if not recs:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in recs:
                writer.writerow({k: r[k] for k in CSV_FIELDS})
        return path
