# Memory Prompt Testing Results

## Summary

We tested memory prompt improvements to reduce redundancy in agents' `notes_to_self` field. The key finding: **prompt quality matters less than market activity** for generating valuable notes.

## Prompts Tested

### Option 2 (BEST - Currently Implemented)
```
MEMORY SYSTEM:
Use 'notes_to_self' to record what you LEARNED this round (what worked/failed).
Don't repeat trade history or prices - you'll see those next round.
```

### Option C (Previously Tested)
```
MEMORY SYSTEM:
Use 'notes_to_self' to record your strategic thinking and learnings.
What are you trying? What did you learn from previous rounds?
Don't repeat trades or prices.
```

## Results Comparison

| Metric | Option 2<br/>(Fair Price,<br/>No Trades) | Option C<br/>(Fair Price,<br/>No Trades) | Option 2<br/>(Mispriced,<br/>30 Trades) |
|--------|------------|------------|----------------|
| **Scenario** | gptoss_social_with_memory | gptoss_social_with_memory | gptoss_social_mispriced |
| **Initial Price** | $28 (= fundamental) | $28 (= fundamental) | $35 (25% above) |
| **Trades Executed** | 0 | 0 | 30 |
| **Avg Note Length** | 70 chars | 67 chars | **140 chars** ✅ |
| **Unique Notes** | 60% | 65% | **91%** ✅ |
| **Generic Phrases** | **47%** ✅ | 69% | 86% |
| **Strategic Indicators** | 19% | 17% | **47%** ✅ |
| **Max Repetition** | 11x | 8x | **3x** ✅ |

## Key Findings

### 1. Market Activity Drives Note Quality
With actual trading (mispriced scenario):
- **91% unique notes** vs 60-65% in fair equilibrium
- **140 char average** vs 67-70 (agents have more to say)
- **47% contain strategic indicators** (executed, cancelled, bought, sold, profit)
- **Only 3x max repetition** vs 8-11x

### 2. Generic Phrases Are Often Strategic
High generic phrase count (86%) in mispriced scenario is actually **good** because agents are writing conditional plans:
- "If price drops below $28, consider covering early"
- "Monitor execution; if price falls below $27.5, consider buying"
- "If market reacts quickly, consider adding more short orders"

These are valuable strategic notes, not redundant restating.

### 3. Examples of High-Quality Notes (Option 2 + Trading)

**Round 2, Agent 2:**
> "Cancelled high limit sells; placed market sell for 6k shares (1k owned + 5k short) to align with bearish crowd."

**Round 4, Agent 4:**
> "Executed market sell of all owned shares to capture cash before anticipated price decline."

**Round 7, Agent 7:**
> "Executed full sell of holdings to capture overvaluation premium. Cash now available for higher‑return opportunities."

### 4. Fair Equilibrium Problem
When price = fundamental value:
- No arbitrage opportunities
- All agents calculate same fair value
- No trades occur
- Notes become generic ("monitor price", "watch for changes")
- High redundancy because everyone sees identical market

## Recommendation

✅ **Keep Option 2 prompt** (currently implemented in `src/agents/LLMs/services/prompt_builder.py:26-29`)

✅ **Test memory features on scenarios with mispricing** to see actual strategic learning

⚠️ **Avoid testing on fair equilibrium scenarios** - they don't generate meaningful memory usage

## Files Modified

1. `src/agents/LLMs/services/prompt_builder.py` - Reverted to Option 2 prompt
2. `src/scenarios/feature_ab_tests.py` - Added `gptoss_social_mispriced` scenario

## Test Logs

- Fair equilibrium tests: `logs/gptoss_social_with_memory/20251120_235558/` (Option 2), `logs/gptoss_social_with_memory/20251121_001728/` (Option C)
- Mispriced test: `logs/gptoss_social_mispriced/20251121_015036/` (Option 2)
