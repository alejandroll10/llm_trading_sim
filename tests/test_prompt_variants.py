"""
Tests for the prompt-family variant packs and SYSTEM_PROMPT_OVERRIDES plumbing
(issue #102).

Covers:
    - Every JSON pack under sweeps/variants/ parses and is structurally valid
      (known agent types, non-empty prompts, paraphrases actually differ from
      the originals, one unmodified control per pack, valid info modes).
    - The packs on disk match the generator script (no hand-edit drift).
    - LLMAgent honors system_prompt_override, including with SELF_MODIFY.
    - BaseSimulation-level validation logic rejects bad override targets.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _logging_stub
_logging_stub.install()

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANTS_DIR = REPO_ROOT / "sweeps" / "variants"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import generate_prompt_variant_packs as gen

from agents.agent_types import AGENT_TYPES
from agents.deterministic.deterministic_registry import DETERMINISTIC_AGENTS
from scenarios.base import FundamentalInfoMode

# Top-level scenario params a variant pack is allowed to override.
ALLOWED_OVERRIDE_KEYS = {"SYSTEM_PROMPT_OVERRIDES", "FUNDAMENTAL_INFO_MODE", "PROMPT_FAMILY"}

PACK_PATHS = sorted(VARIANTS_DIR.glob("*.json"))


def load_pack(path):
    with open(path) as f:
        return json.load(f)


def variant_items(pack):
    return {k: v for k, v in pack.items() if not k.startswith("_")}


class TestPackStructure:
    def test_packs_exist(self):
        assert len(PACK_PATHS) >= 9, f"Expected >= 9 packs in {VARIANTS_DIR}, found {len(PACK_PATHS)}"

    @pytest.mark.parametrize("path", PACK_PATHS, ids=lambda p: p.stem)
    def test_pack_is_valid(self, path):
        pack = load_pack(path)

        meta = pack.get("_meta")
        assert isinstance(meta, dict) and meta.get("version"), f"{path.name}: missing versioned _meta"

        variants = variant_items(pack)
        assert variants, f"{path.name}: no variants"

        controls = [name for name, ov in variants.items() if ov == {}]
        assert len(controls) == 1, f"{path.name}: expected exactly one unmodified control variant"

        for name, overrides in variants.items():
            assert isinstance(overrides, dict), f"{path.name}:{name}: overrides must be an object"
            unknown = set(overrides) - ALLOWED_OVERRIDE_KEYS
            assert not unknown, f"{path.name}:{name}: unexpected override keys {unknown}"

            if "FUNDAMENTAL_INFO_MODE" in overrides:
                FundamentalInfoMode(overrides["FUNDAMENTAL_INFO_MODE"])  # raises if invalid

            for agent_type, prompt in overrides.get("SYSTEM_PROMPT_OVERRIDES", {}).items():
                assert agent_type in AGENT_TYPES, f"{path.name}:{name}: unknown agent type '{agent_type}'"
                assert agent_type not in DETERMINISTIC_AGENTS, \
                    f"{path.name}:{name}: '{agent_type}' is deterministic (no prompt)"
                assert isinstance(prompt, str) and prompt.strip(), \
                    f"{path.name}:{name}: empty prompt for '{agent_type}'"
                assert prompt.strip() != AGENT_TYPES[agent_type].system_prompt.strip(), \
                    f"{path.name}:{name}: paraphrase for '{agent_type}' is identical to the original"

    def test_paraphrases_are_mutually_distinct(self):
        """Within each persona, the 4 paraphrases + original are all different."""
        for persona, texts in gen.PARAPHRASES.items():
            all_texts = [AGENT_TYPES[persona].system_prompt.strip()] + [t.strip() for t in texts]
            assert len(set(all_texts)) == len(all_texts), f"duplicate paraphrase for '{persona}'"

    def test_packs_match_generator(self):
        """Checked-in JSON must be regenerable from the script (no hand edits)."""
        for stem, pack in gen.build_packs().items():
            path = VARIANTS_DIR / f"{stem}.json"
            assert path.exists(), f"missing generated pack {path.name}"
            assert path.read_text() == gen.render(pack), \
                f"{path.name} drifted from generator; rerun scripts/generate_prompt_variant_packs.py"

    def test_family_pack_covers_all_personas(self):
        pack = load_pack(VARIANTS_DIR / "persona_families.json")
        for name, overrides in variant_items(pack).items():
            if overrides == {}:
                continue
            assert set(overrides["SYSTEM_PROMPT_OVERRIDES"]) == set(gen.PERSONAS), \
                f"persona_families:{name}: must override every workhorse persona"


class TestRunSweepIntegration:
    def test_variants_file_meta_key_skipped(self):
        """run_sweep must not treat _meta as a variant."""
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from run_sweep import build_cells

        pack = load_pack(PACK_PATHS[0])
        variants = {k: v for k, v in pack.items() if not k.startswith("_")}
        cells = build_cells([42], [0.0], ["m"], variants)
        assert len(cells) == len(variants)
        assert all("_meta" not in c["cell_id"] for c in cells)

    def test_aggregate_prompt_family_fallbacks(self):
        from aggregate_sweep import cell_prompt_family
        assert cell_prompt_family({"prompt_family": "value_p1"}) == "value_p1"
        assert cell_prompt_family(
            {"param_overrides": {"PROMPT_FAMILY": "fam"}}) == "fam"
        assert cell_prompt_family({"variant": "v"}) == "v"  # pre-#102 manifest
        assert cell_prompt_family({}) == "baseline"


class DummyLLMService:
    pass


@pytest.fixture
def llm_agent_factory(monkeypatch):
    """Build LLMAgents without touching the OpenAI client."""
    import agents.LLMs.llm_agent as llm_agent_module
    monkeypatch.setattr(llm_agent_module, "LLMService", DummyLLMService)
    from agents.LLMs.services.schema_features import Feature

    def make(agent_type="value", override=None, features=frozenset()):
        return llm_agent_module.LLMAgent(
            agent_id=0,
            agent_type=agent_type,
            enabled_features=set(features),
            system_prompt_override=override,
            initial_cash=1000.0,
            initial_shares=10,
            initial_price=28.0,
        ), Feature
    return make


class TestLLMAgentOverride:
    def test_default_prompt_without_override(self, llm_agent_factory):
        agent, _ = llm_agent_factory("value", override=None)
        assert agent.get_current_system_prompt() == AGENT_TYPES["value"].system_prompt

    def test_override_replaces_prompt(self, llm_agent_factory):
        agent, _ = llm_agent_factory("value", override="You pick cheap assets and sell dear ones.")
        assert agent.get_current_system_prompt() == "You pick cheap assets and sell dear ones."

    def test_override_seeds_self_modify_history(self, llm_agent_factory):
        from agents.LLMs.services.schema_features import Feature
        agent, _ = llm_agent_factory("value", override="Custom base.",
                                     features={Feature.SELF_MODIFY})
        assert agent.current_system_prompt == "Custom base."
        assert agent.prompt_history == [(0, "Custom base.")]
        assert agent.get_current_system_prompt() == "Custom base."

    @pytest.mark.parametrize("path", PACK_PATHS, ids=lambda p: p.stem)
    def test_every_pack_prompt_loads_into_agent(self, llm_agent_factory, path):
        """Every override in every pack constructs an agent whose prompt is the override."""
        for name, overrides in variant_items(load_pack(path)).items():
            for agent_type, prompt in overrides.get("SYSTEM_PROMPT_OVERRIDES", {}).items():
                agent, _ = llm_agent_factory(agent_type, override=prompt)
                assert agent.get_current_system_prompt() == prompt


class TestOverrideValidation:
    """The validator BaseSimulation.__init__ runs on SYSTEM_PROMPT_OVERRIDES."""

    def test_rejects_unknown_type(self):
        from base_sim import validate_system_prompt_overrides
        with pytest.raises(ValueError, match="unknown agent type"):
            validate_system_prompt_overrides({"not_a_type": "prompt"})

    def test_rejects_deterministic_type(self):
        from base_sim import validate_system_prompt_overrides
        with pytest.raises(ValueError, match="deterministic"):
            validate_system_prompt_overrides({"gap_trader": "prompt"})

    def test_rejects_empty_prompt(self):
        from base_sim import validate_system_prompt_overrides
        with pytest.raises(ValueError, match="non-empty string"):
            validate_system_prompt_overrides({"value": "   "})

    def test_accepts_every_shipped_pack(self):
        from base_sim import validate_system_prompt_overrides
        for path in PACK_PATHS:
            for name, overrides in variant_items(load_pack(path)).items():
                validate_system_prompt_overrides(
                    overrides.get("SYSTEM_PROMPT_OVERRIDES", {}))

    def test_none_is_accepted(self):
        from base_sim import validate_system_prompt_overrides
        validate_system_prompt_overrides(None)
