"""
Prompt sanity check for experiment A1 (issue #103).

Renders the actual trading prompt the `default` persona receives under
INFINITE_ROUNDS for each FUNDAMENTAL_INFO_MODE, WITHOUT hitting the LLM API.

It monkeypatches LLMService.get_decision so every agent "holds" (empty order
list) and, as a side effect, records the fully rendered system + user prompt for
each agent/round. The simulation machinery (market state, dividend history,
signal extraction, formatting) runs for real, so the captured text is exactly
what the model would see. We run a few rounds so REALIZATIONS_ONLY / AVERAGE have
non-empty dividend history to render (the round-1 edge case is exercised too).

Usage:
    python scripts/a1_prompt_sanity_check.py            # print round-1 and last-round
                                                        # prompts for the first default agent
    python scripts/a1_prompt_sanity_check.py --rounds 4 --out /tmp/a1_prompts
"""

import argparse
import sys
from pathlib import Path

# Make `import ...` resolve against src/ just like run_base_sim does.
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from agents.LLMs.services.llm_services import LLMService, LLMResponse  # noqa: E402

MODES = ["full", "process_only", "realizations_only", "average", "none"]

# Records populated by the patched get_decision: one per (mode, round, agent).
CAPTURED = []


def install_prompt_capture(current_mode_holder):
    """Patch LLMService.get_decision to record prompts and return a hold decision."""
    def patched(self, request):
        CAPTURED.append({
            "mode": current_mode_holder[0],
            "round": request.round_number,
            "agent_id": request.agent_id,
            "system_prompt": request.system_prompt,
            "user_prompt": request.user_prompt,
        })
        # Build a valid "hold" decision so the round completes without an API call.
        decision = self.get_fallback_decision(request.agent_id, request.enabled_features)
        decision["reasoning"] = "prompt sanity check: hold"
        return LLMResponse(decision=decision, raw_response="{}")
    LLMService.get_decision = patched


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3,
                    help="Rounds to render (>=2 gives REALIZATIONS/AVERAGE some history).")
    ap.add_argument("--out", type=str, default=None,
                    help="Directory to dump every rendered prompt (default: scratch under logs/).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else (SRC.parent / "logs" / "a1_prompt_sanity")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imported lazily so the sys.path shim above is already in place.
    from run_base_sim import run_scenario

    mode_holder = [None]
    install_prompt_capture(mode_holder)

    for mode in MODES:
        mode_holder[0] = mode
        print(f"\n=== rendering mode={mode} ===")
        try:
            run_scenario(
                "a1_information_ladder",
                param_overrides={
                    "FUNDAMENTAL_INFO_MODE": mode,
                    "NUM_ROUNDS": args.rounds,
                    # 1x FV anchor so the price line is unambiguous while reading.
                    "INITIAL_PRICE": 28.0,
                },
                sim_type_override=f"a1_prompt_sanity/{mode}",
            )
        except Exception as e:  # noqa: BLE001 -- surface any template/render failure loudly
            print(f"  [FAIL] mode={mode} raised {type(e).__name__}: {e}")
            raise

    # Dump every captured prompt to disk, and print round-1 + last-round for the
    # first default agent per mode.
    print(f"\nCaptured {len(CAPTURED)} prompt renders. Writing to {out_dir} ...")
    last_round = args.rounds
    for mode in MODES:
        recs = [r for r in CAPTURED if r["mode"] == mode]
        if not recs:
            print(f"[warn] no prompts captured for mode={mode}")
            continue
        agent0 = sorted({r["agent_id"] for r in recs})[0]
        for r in recs:
            fpath = out_dir / f"{mode}__{r['agent_id']}__round{r['round']}.txt"
            fpath.write_text(
                f"MODE: {mode}\nAGENT: {r['agent_id']}\nROUND: {r['round']}\n"
                f"\n----- SYSTEM PROMPT -----\n{r['system_prompt']}\n"
                f"\n----- USER PROMPT -----\n{r['user_prompt']}\n"
            )
        for rnd in sorted({1, last_round}):
            match = [r for r in recs if r["agent_id"] == agent0 and r["round"] == rnd]
            if not match:
                continue
            rec = match[0]
            print("\n" + "=" * 78)
            print(f"MODE={mode}  AGENT={agent0}  ROUND={rnd}")
            print("=" * 78)
            print("----- SYSTEM PROMPT -----")
            print(rec["system_prompt"])
            print("----- USER PROMPT -----")
            print(rec["user_prompt"])

    print(f"\nAll rendered prompts saved under {out_dir}")


if __name__ == "__main__":
    main()
