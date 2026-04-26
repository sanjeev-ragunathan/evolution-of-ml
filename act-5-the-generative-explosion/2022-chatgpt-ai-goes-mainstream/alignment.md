# Alignment

> How AI responses are made to align with human preferences.

---

## The Story

**2020 — GPT-3 problem:** powerful but unpredictable. Trained to predict the next token, not to be helpful, harmless, or honest. Confidently wrong, sometimes harmful, often unhelpful.

**Why SFT alone isn't enough:** writing perfect example responses for every possible prompt is impossible. And for any prompt there are many valid answers — SFT can't teach which one humans actually prefer.

**The fix — use RL with human feedback:**
1. Generate multiple responses
2. Have humans rank them
3. Train a model to predict the ranking (reward model)
4. Use RL to make the LLM generate higher-ranking responses

This is RLHF. Same model, new training, aligned behavior.

---

## RL Algorithms vs Pipelines

These are different things and people conflate them.

| | What it is | Examples |
|---|---|---|
| **Pipeline** | The full recipe for alignment | RLHF, Constitutional AI, DPO |
| **Algorithm** | The optimizer used inside the RL step | PPO, GRPO |

A pipeline can use any RL algorithm. PPO and GRPO are interchangeable engines.

---

## Pipelines

### RLHF (2022) — OpenAI

```
1. SFT on ideal conversations
2. Train reward model from human rankings
3. RL using reward model + KL penalty
```

Uses humans to label preferences. Quality is high but expensive and slow.

### Constitutional AI (2022) — Anthropic

```
1. SFT with self-critique against a written constitution
2. Train reward model from AI-generated rankings
3. RL using reward model + KL penalty
```

Same shape as RLHF but AI replaces humans for feedback. Scalable, transparent (the constitution is human-readable). Also called RLAIF.

### DPO (2023) — Stanford

```
1. SFT
2. Train directly on preference pairs (no reward model, no RL)
```

Skips the reward model and the RL step entirely. Treats alignment as binary classification — increase probability of preferred responses, decrease probability of rejected ones, anchored to SFT.

Simpler, cheaper, often comparable. Loses some ceiling at frontier scale because it can't generate new responses for evaluation.

---

## Algorithms

### PPO (2017) — OpenAI

The original RL algorithm used in RLHF. Uses a learned value network as a baseline for advantage estimation. Stable but memory-heavy — value network roughly doubles training cost.

### GRPO (2024) — DeepSeek

Replaces the value network with **group statistics**. For each prompt, sample N responses, use the group's mean reward as the baseline. No value network needed. More efficient for LLM training, especially for sparse rewards (e.g., reasoning tasks).

---

## Reward Hacking — The Constant Threat

The model finds ways to score high on the reward model without actually being helpful. RLHF and CAI both use a **KL divergence penalty** against the SFT model — improvements must be small, no drifting too far. Think of it as a leash.

---

## Who Uses What

| Company | Model | Pipeline | Algorithm |
|---------|-------|----------|-----------|
| OpenAI | ChatGPT, GPT-4 | RLHF | PPO |
| Anthropic | Claude | Constitutional AI | PPO |
| DeepSeek | DeepSeek-R1 | RL with verifiable rewards | GRPO |
| Meta | Llama 3 (some variants) | DPO | — |
| Mistral | Mixtral instruct | DPO | — |

---

## TL;DR

- **Why alignment exists:** raw LLMs are token predictors, not assistants.
- **RLHF (2022)** introduced the 3-step recipe with humans as labelers.
- **Constitutional AI (2022)** swaps humans for AI feedback at scale.
- **DPO (2023)** drops the reward model and RL entirely — just supervised learning on preferences.
- **PPO (2017)** is the classic RL algorithm; **GRPO (2024)** is the modern LLM-specific replacement.
- The choice is always a tradeoff between alignment quality, cost, and engineering simplicity.
