# Competitive Intelligence — Updated 2026-03-19 (Cycle 2)

## Live Leaderboard (from PRs, sorted by BPB)

| Rank | Author | BPB | Approach | PR |
|------|--------|-----|----------|----|
| 1 | yesbhautik | **1.0149** | Val-train + sliding window + 10L INT6 + Muon tuning | #64 |
| 2 | daniellawson9999 | 1.1111 | Val-train only | #44 |
| 3 | jfprincz | **1.1659** | MLP 3x + INT6 + sliding window (stride=256) + zstd | #70 |
| 4 | saml212 | 1.1793 | Long-context sliding window | #61 |
| 5 | aquariouseworkman | 1.1808 | Seq4096 + sliding window eval | #65 |
| 6 | arjun-krishna1 | 1.1833 | Longer training context | #66 |
| 7 | mattqlf | **1.1925** | Sliding window eval (stride=64), zero training changes | #50 |
| 8 | spokane-way | 1.2014 | Unknown | #52 |
| 9 | yahya010 | 1.2067 | Seq2048 + FP16 tied embedding | #63 |
| 10 | nanlliu | 1.2147 | 10L mixed precision, lower LR | #39 |
| 11 | notapplica | 1.2160 | NTK eval + overtone init | #60 |
| 12 | chonchiog | 1.2197 | FP16 tied embed + warmdown | #42 |
| 13 | baseline | 1.2244 | Reference | — |

## KEY DISCOVERY: Sliding Window Eval is the #1 Technique

**Sliding window eval alone gives ~0.03 BPB for FREE** (no training changes, no artifact cost).

### How it works:
- Standard eval: each token gets 0 to seq_len-1 tokens of context (average ~512)
- Sliding window: evaluate with stride << seq_len, so every token gets ~960+ context tokens
- stride=64: score 1024-token windows, but advance only 64 tokens between windows
- Only count the last `stride` tokens' losses per window (they all have near-full context)
- Result: every token is scored with maximum context

### Impact by submission:
- mattqlf (stride=64): 1.2244 → 1.1925 = **-0.032 BPB** (eval only, zero training changes!)
- jfprincz (stride=256): contributed ~0.033 BPB of their total improvement
- yesbhautik (stride=64): contributed to reaching 1.0149

### Our action:
IMPLEMENT SLIDING WINDOW EVAL IMMEDIATELY. This is free BPB.

## Key Findings (Updated)

### 1. SLIDING WINDOW EVAL (NEW, HIGHEST PRIORITY)
- ~0.03 BPB free improvement
- stride=64 is optimal (more context per token)
- Eval time: ~70s on 8xH100 (well within 10-min eval budget)
- Combined with seq4096: even better

### 2. MLP 3x WIDER (NEW)
- jfprincz: MLP_MULT=3 gives ~0.019 BPB improvement
- Makes hidden dim 1536 instead of 1024
- Needs INT6 quantization to fit in 16MB

### 3. ZSTD COMPRESSION (NEW)
- jfprincz uses zstd level 22 instead of zlib
- Better compression ratio → more room for parameters

### 4. VAL-DATA TRAINING (confirmed)
- Still allowed per organizers
- yesbhautik combined it with everything else for 1.0149

### 5. MUON TUNING (NEW)
- yesbhautik: momentum=0.99 (was 0.95), warmup_start=0.92 (was 0.85)
- seq_len=4096 for training
- These are significant departures from baseline

## Revised Priority Stack

### Tier 0: IMPLEMENT NOW (free BPB)
1. Sliding window eval (stride=64) → ~0.03 BPB free
2. Longer eval context (EVAL_SEQ_LEN=4096) → compounds with sliding window

### Tier 1: PROVEN WINS
3. MLP_MULT=3 → ~0.02 BPB
4. INT6 middle layers (already in our v1)
5. 10 layers (already in our v1)
6. Lower LR 0.02 (already in our v1)
7. Muon momentum=0.99

### Tier 2: COMPOUND
8. Train on val data + all above → target: < 1.05 BPB
9. zstd-22 instead of zlib
10. Train at seq_len=4096

### Tier 3: ARCHITECTURE
11. Weight sharing + wider model
12. QAT
13. BitNet/ternary
