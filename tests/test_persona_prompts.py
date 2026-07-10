"""Regression tests for the persona prompt extraction (issue #107).

LLM persona prompts moved from inline strings in agent_types.py to
src/agents/prompts/<type_id>.md. Prompt text drives paper results, so every
persona's system prompt is pinned here by sha256, computed from the original
inline strings BEFORE the extraction. If a hash mismatch is intentional
(you deliberately edited a prompt), update the hash here in the same commit.
"""
import hashlib
from pathlib import Path

import pytest

from agents.agent_types import (
    AGENT_TYPES,
    PROMPTS_DIR,
    _DETERMINISTIC_PLACEHOLDER_NAMES,
    _parse_persona_file,
)
from agents.LLMs.llm_prompt_templates import STANDARD_USER_TEMPLATE

# sha256 of each persona's system_prompt as it existed inline in
# agent_types.py at commit 044c087 (pre-extraction baseline).
EXPECTED_PERSONAS = {
    "value": ("Value Investor", "29cc967fc28f499e496711ac5bffaac5aa49a5e84a416475d08e954611aa59b1"),
    "momentum": ("Momentum Trader", "2fe5fe4e2297fb17f590c952445dcc3a83433315ddd3d81aa990e3d92438d9c9"),
    "market_maker": ("Market Maker", "c1e27e511753da031e5e4b7bd617941dbdf24945519f1c5e859637273f25e473"),
    "contrarian": ("Contrarian Trader", "fef6d9aff227aa53b1764a9f25078cf55ed16ab9b4b779a44b5657aeba0c86cb"),
    "news": ("News Trader", "f88eaa490ca998988c1d0e5e676df04ecc0c41421fe64fdf25fabaa50848fd5a"),
    "default": ("Default Trader", "406331932a80269de7c1ca5e8b43f73bf6a622a54d8347902910364bb331fb4c"),
    "default_llama": ("Default Trader", "406331932a80269de7c1ca5e8b43f73bf6a622a54d8347902910364bb331fb4c"),
    "default_gpt": ("Default Trader", "406331932a80269de7c1ca5e8b43f73bf6a622a54d8347902910364bb331fb4c"),
    # sha256 of the empty string: 'minimal' intentionally has no system prompt.
    "minimal": ("Minimal Trader", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "speculator": ("Speculator", "b93fd3895d0aeadcb58421b1f4a3814939255c9e62a2ee9f540c2e17fba61a53"),
    "college_student": ("College Student", "e7a49259fc9f1e3a1018e715409f1204f859121bc5ef3520a1834918bef2c255"),
    "retail": ("Retail Trader", "29af48d4dd1ca908a8cd932eb667b0b5b4c62ac52b6826a11e21596aee5409f9"),
    "optimistic": ("Optimistic", "ff9476cbf38b8e3de2816fad7547a02394f0c2033791c904c5ac29c278e2cb0b"),
    "pessimistic": ("Pessimistic", "5ba038bcbf2524f0d69fb3ae605b0eb37209eb7660508d3d05a9155f7f311fa8"),
    "short_seller": ("Short Seller", "384d71393be5134712a7cbafa9e1f27d65d1fa18c851454dad97eb00c333fca2"),
    "leverage_trader": ("Leverage Trader", "50dc29468fbfd19d2d3867843a690a7decceebaaf56415eaebea9d092fe7a7c4"),
    "long_short": ("Long-Short Trader", "b4a1a2713df665b501d8cc6e28566b94eb84fd724c29f0e7c1179b8c7633b5bd"),
    "influencer": ("Market Influencer", "752b6d0f22c33f49a53cf20b6155bd11e76af697260f590a0a989b32643a0d0f"),
    "herd_follower": ("Herd Follower", "c873edd21f15e35ca4aac8bda74b5f5894e16674ed40e56c3b74b3e34c871b54"),
    "hold_llm": ("LLM Hold Trader", "3f6eb759045a142662deed5625cf509d2296bd120e45d84369a2818d65025303"),
    "adaptive_learner": ("Adaptive Learner", "5351dee13484e1a22263e247fa458e6a79ad3e18ccd25585239002121273c857"),
    "profit_maximizer": ("Profit Maximizer", "c40630d03e2ce6e3d1a3a4587f0c04216cf7b3f393d6f39fece5d43a50ce8d51"),
    "strategy_experimenter": ("Strategy Experimenter", "d3f99f45beba2cdfc235632024d01f8b5375ab986ce1ff815eb80c679a7945ad"),
}


class TestExtractionByteIdentity:
    @pytest.mark.parametrize("type_id", sorted(EXPECTED_PERSONAS))
    def test_prompt_hash_matches_pre_extraction_inline_string(self, type_id):
        expected_name, expected_sha = EXPECTED_PERSONAS[type_id]
        agent_type = AGENT_TYPES[type_id]
        actual_sha = hashlib.sha256(agent_type.system_prompt.encode("utf-8")).hexdigest()
        assert actual_sha == expected_sha, (
            f"Persona '{type_id}' prompt no longer matches the pre-extraction "
            f"baseline. If the edit to prompts/{type_id}.md was intentional, "
            f"update EXPECTED_PERSONAS with sha256 {actual_sha}.")
        assert agent_type.name == expected_name
        assert agent_type.type_id == type_id
        assert agent_type.user_prompt_template == STANDARD_USER_TEMPLATE

    def test_prompt_files_match_expected_persona_set(self):
        on_disk = {p.stem for p in PROMPTS_DIR.glob("*.md")}
        expected = set(EXPECTED_PERSONAS)
        assert on_disk == expected, (
            f"prompts/ drifted from the pinned persona set: "
            f"missing={sorted(expected - on_disk)}, "
            f"unpinned={sorted(on_disk - expected)}. New personas must be "
            f"added to EXPECTED_PERSONAS with their hash.")

    def test_deterministic_placeholders_unchanged(self):
        for type_id, name in _DETERMINISTIC_PLACEHOLDER_NAMES.items():
            agent_type = AGENT_TYPES[type_id]
            assert agent_type.name == name
            assert agent_type.system_prompt == "Deterministic agent - no prompt needed"
            assert agent_type.user_prompt_template == ""
            assert agent_type.type_id == type_id


class TestPersonaFileParser:
    def _write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "example.md"
        path.write_text(text, encoding="utf-8")
        return path

    def test_body_is_byte_exact_minus_one_trailing_newline(self, tmp_path):
        prompt = "Line one.\n        Indented, trailing spaces:  \nLast line."
        path = self._write(tmp_path, f"---\nname: Example\n---\n{prompt}\n")
        name, body = _parse_persona_file(path)
        assert name == "Example"
        assert body == prompt

    def test_only_one_trailing_newline_is_stripped(self, tmp_path):
        path = self._write(tmp_path, "---\nname: Example\n---\nprompt ends blank\n\n")
        _, body = _parse_persona_file(path)
        assert body == "prompt ends blank\n"

    def test_no_trailing_newline_is_fine(self, tmp_path):
        path = self._write(tmp_path, "---\nname: Example\n---\nno newline at EOF")
        _, body = _parse_persona_file(path)
        assert body == "no newline at EOF"

    def test_hr_line_in_body_is_not_mistaken_for_front_matter_fence(self, tmp_path):
        # The parser takes the FIRST '\n---\n' after the opening fence as the
        # closing fence, so a markdown horizontal rule inside the prompt body
        # must survive intact.
        prompt = "Section one.\n---\nSection two after a horizontal rule."
        path = self._write(tmp_path, f"---\nname: Example\n---\n{prompt}\n")
        name, body = _parse_persona_file(path)
        assert name == "Example"
        assert body == prompt

    def test_empty_body_allowed(self, tmp_path):
        path = self._write(tmp_path, "---\nname: Example\n---\n")
        _, body = _parse_persona_file(path)
        assert body == ""

    def test_missing_front_matter_rejected(self, tmp_path):
        path = self._write(tmp_path, "just a prompt with no header\n")
        with pytest.raises(ValueError, match="front-matter"):
            _parse_persona_file(path)

    def test_unterminated_front_matter_rejected(self, tmp_path):
        path = self._write(tmp_path, "---\nname: Example\nprompt text\n")
        with pytest.raises(ValueError, match="unterminated"):
            _parse_persona_file(path)

    def test_missing_name_rejected(self, tmp_path):
        path = self._write(tmp_path, "---\n---\nprompt\n")
        with pytest.raises(ValueError, match="name"):
            _parse_persona_file(path)

    def test_unknown_front_matter_key_rejected(self, tmp_path):
        path = self._write(tmp_path, "---\nname: Example\nmodel: gpt-4\n---\nprompt\n")
        with pytest.raises(ValueError, match="unknown front-matter key"):
            _parse_persona_file(path)
