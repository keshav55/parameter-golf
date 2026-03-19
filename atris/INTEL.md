# Competitive Intelligence — Updated 2026-03-18

## Live Leaderboard (from PRs, not all merged)

| Rank | Author | BPB | Approach | PR |
|------|--------|-----|----------|----|
| 1 | daniellawson9999 | 1.1111 | Trained on val data | #44 |
| 2 | nanlliu | 1.2147 | 10L, INT8/INT6 mixed, LR=0.02 | #39 |
| 3 | spokane-way | 1.2166 | Unknown | #49 |
| 4 | chonchiog | 1.2197 | FP16 tied embed + warmdown | #42 |
| 5 | kiankyars | 1.2240 | Lower LR, FP16 embed, 960 | #45 |
| 6 | baseline | 1.2244 | Reference | — |

## Key Findings

### 1. TRAINING ON VAL DATA IS LEGAL
daniellawson9999 confirmed with organizers on Discord. Result: 1.1111 BPB.
This is essentially memorization. We should do this but also stack architectural improvements.

**Action:** Train on val data as our primary track. But maintain a "clean" track too
in case this rule gets changed or community pushback forces a policy update.

### 2. INT6 MIXED PRECISION (nanlliu, PR #39)
- Full INT8 for first/last 3 layers, INT6 (step=4 rounding) for middle layers
- Saves ~1.6MB → room for 10th layer
- Lower LR: MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03
- 18.9M params, 5 seeds, mean 1.2139, p < 0.001
- INT8 alone was too big for 10 layers → INT6 middle layers was the insight

### 3. COMPETITOR DOING OUR EXACT IDEA (kxddry, PR #38)
- 3 shared blocks × 3 loops with rank-4 LoRA adapters
- QAT: fake quantize in CastedLinear forward
- Model widened to 768 dim
- LAWA (checkpoint averaging during warmdown)
- RoPE NTK scaling for 2048-token eval
- INT8 degradation: 0.002 → 0.0001 (18× better!)
- Status: WIP, hasn't run on 8xH100 yet
- Their current val_bpb is ~2.96 (running on small compute)

### 4. UNIVERSAL FINDING: LOWER LR IS BETTER
Multiple independent submissions found MATRIX_LR=0.04 is too high.
Consensus: 0.02-0.03 range.

### 5. OTHER APPROACHES IN THE FIELD
- Weight sharing / depth recurrence: multiple attempts (#11, #15, #21, #29, #31, #38, #40)
- QAT: several attempts (#20, #38)
- Larger vocab: one attempt (#37, SP4096)
- Most don't report scores yet or are WIP

## Implications for Our Strategy

### Tier 0 (IMMEDIATE — do before anything else)
- [ ] Train on val data + best architecture → this is how you win
- [ ] Lower LR to 0.02-0.03

### Tier 1 (STACK ON TOP)
- [ ] Mixed precision INT8/INT6 (proven by nanlliu)
- [ ] Add 10th or 11th layer
- [ ] Weight sharing + LoRA (race kxddry to execution)
- [ ] QAT to eliminate quant loss

### Tier 2 (COMPOUND)
- [ ] All of the above + eval at longer context
- [ ] TTT (test-time training) on val data
- [ ] Ensemble if artifact fits

## Threat Assessment
- **nanlliu** is the real competitor. Clean approach, solid stats, proven result.
- **kxddry** has our best ideas but hasn't executed yet. Speed advantage is ours.
- **daniellawson9999** set the floor for val-data training. Others will pile on.
- Most submissions are low quality (no scores, broken, or worse than baseline).
- The real competition is probably 5-10 serious teams.
