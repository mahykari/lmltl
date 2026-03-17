# Can Language Models Follow Temporal Specs?
## Research Game Plan

**Goal.** Determine whether RL fine-tuning with automaton-derived rewards can teach a language model to follow temporal specifications — and find the expressiveness boundary where this stops working.

**Core claim.** External enforcement (the Rothkopf wrapper) is fragile because the constraint is outside the model's learned behavior. Training against a reward machine internalizes the constraint. We conjecture that star-free (LTL-definable) properties are learnable this way, but counting properties (non-star-free) are not.

---

## What Rothkopf Actually Did (So We Can Compare)

From the paper (arXiv:2402.16905v2):

- **Model:** GPT-4 via API, JavaScript interface.
- **Domain:** Choose-your-own-adventure games. Each game = 20 turns of story generation.
- **Evaluation:** 75 games per task. A game is "violating" if *any* turn breaks the spec. Adherence = % of non-violating games.
- **Architecture:** Evaluator LLM (binary 0/1 predicate queries on previous output) → synthesized Mealy machine → prompt modifier → generator LLM.
- **Specs tested (4 tasks, increasing complexity):**

| Task | Spec | Automaton states | TSL adherence | LLM-only adherence |
|------|------|:---:|:---:|:---:|
| 1 | Visit cave only after town + market | 8 | 96.00% | 86.60% |
| 2 | Task 1 + initial setting = forest | 9 | 96.00% | 89.33% |
| 3 | Task 1 + count 3 safe choices → visit town | 16 | 98.67% | 14.67% |
| 4 | Tasks 1 + 2 + 3 combined | 17 | 97.33% | 16.00% |

- **Key finding from Table 2:** Of the LLM-only violations, 14.33% were hallucinations (wrong location) and 33.67% were *arithmetic errors* (miscounting safe choices). The counting task is where LLMs collapse.
- **The 4% failure of TSL:** All TSL failures were point-in-time (the LLM didn't follow the prompt modifier), never procedural (the automaton is correct by construction).

**What this tells us for our design:** Their Task 3 (counting) is exactly the kind of property we predict is hard to internalize. Their Tasks 1–2 (sequencing/ordering) are star-free and should be learnable. We can design our spec categories to mirror theirs.

**Important caveat:** The comparison with Rothkopf is qualitative, not quantitative. Our setup differs in model size (~1000x), model family, training method, domain, and predicate evaluation. We do not claim our numbers are directly comparable to theirs. The value is that they observed the same counting barrier in an independent setup, which corroborates our findings.

---

## Phase 0 — Infrastructure (Week 1)

**Goal:** Get a model loading and fine-tuning on your Slurm cluster. No research content yet.

**Steps:**

1. **Pick a model:** Qwen 2.5-1.5B (Apache 2.0 license, strong for its size, well-supported in HuggingFace). Small enough to iterate fast on a single GPU.
2. **Set up environment:** Create a conda environment on your cluster with PyTorch, `transformers`, `trl`, `peft` (for LoRA), `accelerate`.
3. **Hello-world SFT job:** Fine-tune the model for 100 steps on a trivial dataset (e.g., a few hundred instruction-response pairs). The point is to verify that: the model loads, LoRA works, the GPU has enough VRAM, the Slurm job completes.
4. **Deliverable:** A working Slurm batch script and training script. You submit it, it runs, it produces a checkpoint.

**What I write:** The conda setup script, the training script, the Slurm batch file.
**What you do:** Tell me your GPU type and VRAM, your Slurm partition name, and any module system commands needed. Then submit the job and report back.

**Before you do anything:** Find out:
- What GPUs are on the cluster? (A100? V100? H100? How much VRAM?)
- Is there a shared conda/module system, or do you install your own?
- Maximum wall time for a single job? (24h? 48h? Unlimited?)
- Do you have internet access from compute nodes (to download model weights), or do you need to stage files?

---

## Phase 1 — Build the Monitors (Week 2)

**Goal:** Implement two categories of temporal specs as Python monitor automata, and test whether the formal boundary between them is also a learnability boundary.

A *monitor* is a tiny Python class. It has:
- A set of states (just integers).
- A current state (starts at 0).
- A `step(observation)` method that reads the LLM's output, evaluates predicates (using string matching or a classifier), transitions to the next state, and returns a reward.

No LLM-as-judge. We use simple, deterministic predicate evaluation — keyword/phrase matching, or a small classifier. This eliminates the noisy-sensor problem from the Rothkopf architecture entirely. The predicates are things like "does this response contain the sentence ŝ?" or "does this response mention location X?" — checkable without an LLM.

**Predicate robustness:** Raw keyword matching is vulnerable to reward hacking (the model inserts keywords without meaningful constraint satisfaction). To mitigate this, predicates should check that the keyword appears in a coherent sentence — using template-based checks or a lightweight NLI classifier. A sample of "satisfying" traces must be manually inspected for degeneracy.

### Spec Categories

The specs are organized into two **categories** based on their position in the formal language hierarchy, not by difficulty. The ordering within each category is not meaningful — some "lower" specs may be harder to learn than "higher" ones.

#### Category A — Star-free (LTL-definable)

| Spec | Property (English) | Formal | Monitor states |
|------|-------------------|--------|:---:|
| A1 | Every response contains keyword *k* | **G**(*p*) | 1 |
| A2 | After user asks about A, next response mentions B | **G**(*a* → **X***b*) | 2 |
| A3 | After user says X, respond with A; after user says Y, respond with B; these must strictly alternate | (user-driven alternation) | 2 |
| A4 | Mention A before B, B before C (Rothkopf-style sequencing) | ¬*c* **W** (*a* ∧ *b*) | ~4–8 |
| A5 | Mention A in the first response, never mention A again, mention B in the last response | (positional, no counting) | 3 |

#### Category B — Non-star-free (ω-regular, not LTL-definable)

| Spec | Property (English) | Formal | Monitor states |
|------|-------------------|--------|:---:|
| B1 | Every 3rd response contains sentence ŝ | *s* at positions *i* ≡ 0 (mod 3) | 3 |
| B2 | An even number of responses mention keyword *k* | parity of *k*-occurrences | 2 |

Specs A1–A5 are LTL-definable. Specs B1–B2 are ω-regular but not LTL. The Schützenberger-McNaughton-Papert theorem says this is the formal boundary. Our experiment tests whether it's also the *learnability* boundary.

**Note on A3 vs. the original alternation spec:** The original Tier 3 (strict alternation of own output) conflated temporal reasoning with self-output tracking. A3 isolates the temporal constraint by making alternation depend on *user* inputs, which the model can see in context.

**Note on A5:** This spec serves as a difficulty-matched control for B1. Both require precise positional tracking and have similar monitor complexity (3 states), but A5 is star-free. If A5 succeeds and B1 fails after training, the counting explanation is strengthened.

**Note on finite vs. infinite traces:** The non-star-free character of B1 and B2 holds for infinite words. On fixed-length conversations, these reduce to finite conjunctions. To preserve the counting difficulty, conversation lengths must be variable (sampled uniformly from, say, 10–30 turns) so the model cannot memorize a fixed pattern. This is discussed further in Phase 2.

**What I write:** A `Monitor` base class and concrete subclasses for each spec. Unit tests. Optionally, I use Spot (the automata library) to compile LTL formulas and cross-check the hand-coded monitors.
**What you review:** Are the specs correct? Do the monitors faithfully implement them? This is pure formal methods — your home turf.

---

## Phase 2 — Baseline Evaluation (Week 3)

**Goal:** Measure how well the *base model* (no training) follows each spec using only prompting. This replicates the Rothkopf "LLM alone" condition on our spec categories.

**Setup:**

1. For each spec, write a system prompt explaining the constraint in plain English (mirroring what Rothkopf did for their NL prompts).
2. Simulate 200 multi-turn conversations per spec. Conversation length is sampled uniformly from 10–30 turns (not fixed — see Phase 1 note on finite vs. infinite traces).
3. The monitor scores each conversation: did it satisfy φ? At which turn did it first violate?
4. Compute adherence = % of conversations with zero violations. Report 95% confidence intervals. Use Fisher's exact test for pairwise comparisons between specs (especially A5 vs. B1 — the critical matched-difficulty pair).

**Simulated user design:** The "fake user" sends templated inputs, but these must be carefully designed per spec:
- For sequencing specs (A4): include adversarial orderings (ask about C before A).
- For alternation specs (A3): include long stretches without relevant triggers.
- For positional specs (A5, B1): vary conversation length so the model cannot rely on fixed position.
- Document all templates as part of the experimental methodology.

**Train/test split:** User prompt templates are split into separate sets for training (Phase 3) and evaluation. Evaluation uses different topics, phrasings, and conversation starters than training. Report both in-distribution and out-of-distribution adherence.

**Deliverable:** Table 1 of the paper. "Baseline adherence of Qwen 2.5-1.5B to temporal specs, prompt-only, by category." The key comparison is Category A vs. Category B, not a monotonic degradation across tiers.

**What I write:** The evaluation harness (conversation simulator + monitor scoring), the system prompts, the Slurm job to run inference.
**What you do:** Review the prompts (are they fair? are they equivalent to what a reasonable developer would write?), submit the job, look at the results.

---

## Phase 3 — RL Training (Weeks 4–5)

**Goal:** Train the model to follow each spec using GRPO with monitor-derived rewards.

### What GRPO Does (Plain Language)

GRPO = Group Relative Policy Optimization. Here's the loop:

1. **Sample:** The model generates *G* candidate multi-turn conversations for a given spec (e.g., G = 8).
2. **Score:** The monitor automaton scores each conversation. Reward = how many turns the spec was satisfied, normalized. A conversation that satisfies φ for all 20 turns gets reward 1.0; one that violates at turn 5 gets reward 5/20 = 0.25.
3. **Rank:** The G conversations are ranked by reward. The model is updated to make high-reward conversations more likely and low-reward ones less likely. No critic network, no learned reward model — just relative ranking within the group.
4. **Repeat** for *K* training steps.

That's it. The automaton provides the reward signal for free. HuggingFace TRL has a `GRPOTrainer` class that handles steps 3–4. I write the code that connects the monitor (step 2) to the trainer (step 3).

### Training Details

- **One LoRA adapter per spec.** LoRA adds a small number of trainable parameters on top of the frozen base model. Cheap to train, cheap to store, clean comparison.
- **Conversation length:** Variable, sampled uniformly from 10–30 turns per rollout (matching Phase 2).
- **Group size:** G = 4 or 8 (we tune this).
- **Training steps:** Start with K = 500, check learning curves, extend if needed up to K = 5000.
- **Checkpointing:** Save every 50 steps so we can plot adherence vs. training step.
- **Evaluation at each checkpoint:** Run the Phase 2 evaluation harness (200 conversations, monitor-scored, using held-out templates) to get an adherence number.

### Ablations

- **LoRA rank sweep (critical comparison only):** For the A5 vs. B1 comparison, sweep LoRA rank ∈ {8, 16, 32, 64} and training steps ∈ {500, 1000, 2000, 5000}. If A5 converges across ranks while B1 does not, the argument that counting is fundamentally hard (not just capacity-limited) is much stronger.
- **Reward scheme:** Test both partial-credit reward (turns_satisfied / total_turns) and binary reward (1.0 if the entire conversation satisfies φ, 0.0 otherwise). Partial credit creates a smoother learning signal for safety properties than for counting — if the gap persists under binary reward, the finding is cleaner.
- **Model scale:** Run the critical comparison (A5 vs. B1) on Qwen 2.5-7B as well as 1.5B. A single model size cannot distinguish "counting is fundamentally unlearnable" from "1.5B parameters is insufficient for counting." If 7B also fails at B1, the argument is much stronger. If 7B succeeds, the finding is equally interesting (the boundary shifts with scale).

### What Could Go Wrong

- **Reward hacking:** The model learns to output the literal keyword/phrase in every response regardless of context. Mitigation: predicates check for coherent usage (see Phase 1), and we manually inspect a sample of satisfying traces for degeneracy.
- **Training instability:** GRPO on small models can be finicky. If GRPO fails, we report the failure itself as a finding and diagnose the cause. We do **not** silently fall back to DPO, because DPO is offline contrastive learning, not online RL — it cannot support the paper's claim about RL internalization. If we run DPO, it is presented as a separate experiment with different claims.
- **Multi-turn rollout cost:** Generating 8 × 30-turn conversations per training step is expensive. Mitigation: start with shorter rollouts; extend once we know it works.

**What I write:** The GRPO training script, the reward function wrapper, the evaluation-at-checkpoint script, the Slurm job.
**What you do:** Submit training jobs, monitor loss curves and adherence curves, flag anything weird.

---

## Phase 4 — Analysis & Paper (Weeks 6–7)

### The Key Figure

A plot with 7 groups (one per spec) and 2 bars each: baseline (prompt-only) adherence vs. post-GRPO adherence. The prediction:

```
Category A (star-free):
  A1 (always k):         ████████████ 95%+  (baseline already high)
  A2 (a→Xb):             █████████░░░ ~80%  →  ████████████ 95%+
  A3 (alternation):      ██████░░░░░░ ~50%  →  ███████████░ ~90%
  A4 (sequencing):       ████░░░░░░░░ ~30%  →  █████████░░░ ~80%
  A5 (positional):       ████░░░░░░░░ ~30%  →  █████████░░░ ~80%

Category B (non-star-free):
  B1 (every 3rd):        ██░░░░░░░░░░ ~15%  →  ████░░░░░░░░ ~35%  ← the boundary
  B2 (parity):           ██░░░░░░░░░░ ~15%  →  ████░░░░░░░░ ~35%  ← the boundary
```

The critical comparison is A5 vs. B1: both require positional tracking with 3-state monitors, but A5 is star-free and B1 is not. If A5 reaches high adherence and B1 does not, the counting explanation is specifically supported (not just "harder specs are harder").

If Category A specs reach high adherence and Category B specs do not, the star-free / non-star-free boundary is the learnability boundary. That's the paper's main result.

### Secondary Analyses

- **Learning curves:** Adherence vs. training step, per spec. Is there a qualitative difference in convergence behavior between categories?
- **LoRA rank sensitivity:** Does increasing adapter capacity close the gap for Category B? If not, the limitation is fundamental, not capacity-driven.
- **Model scale:** Does the boundary hold at 7B? Shift? Disappear?
- **Quality check:** Perplexity or coherence scores before/after training. Does GRPO degrade response quality? (If not, that's a strong advantage over the wrapper approach.)
- **Reward scheme comparison:** Do partial-credit and binary reward produce the same category-level gap?
- **Comparison with Rothkopf:** Qualitative only. Our A4 mirrors their Task 1. We note that they observed the same counting barrier independently, corroborating our finding. No direct numerical comparison (different model, domain, and evaluation).

### Paper Structure

1. Introduction: alignment is a temporal property; current approaches are external enforcement.
2. Background: LTL, ω-regular languages, star-free hierarchy, reward machines.
3. The problem: can RLHF internalize temporal specs?
4. Method: monitor compilation + GRPO training.
5. Experiments: the spec categories, baseline, training, results.
6. Analysis: the expressiveness boundary.
7. Connection to dpm / private RV (future work).
8. Conclusion.

**Target venue:** AAAI, ICLR, or NeurIPS (depending on whether the contribution reads more as AI safety, formal methods + ML, or RL).

---

## Tools & Dependencies

| Component | Tool | Notes |
|-----------|------|-------|
| Base model | Qwen 2.5-1.5B | Apache 2.0, HuggingFace |
| Training framework | TRL (GRPOTrainer) | HuggingFace, pip install |
| Parameter-efficient tuning | PEFT (LoRA) | pip install |
| Monitor compilation | Spot (optional) | For cross-checking hand-coded monitors |
| Experiment tracking | Weights & Biases (free tier) | Or just CSV logs |
| Plotting | matplotlib | |
| Cluster | Slurm | Your IST cluster |

---

## Claude Code Skills

The project is built around Claude Code with specialist skills — reusable agents that handle specific components. You supervise at the high level; the skills handle the boilerplate. Each skill lives in the project repo and can be invoked by name.

### `monitor`

**Purpose:** Takes a temporal specification and produces a Python monitor class.

**Input:** An LTL formula string (e.g., `G(a -> X b)`) or a plain-English description of the property.

**Output:** A Python file containing a `Monitor` subclass with:
- An `__init__` method defining states and initial state.
- A `step(observation: dict[str, bool]) -> float` method that reads predicate truth values from the LLM's output, transitions the automaton, and returns a reward (1.0 if the spec is satisfied at this step, 0.0 if violated, with optional shaping based on progress through accepting states).
- A `reset()` method.
- A `done() -> bool` property indicating if the monitor has reached a permanent verdict.
- Unit tests exercising accept/reject traces.

**When to invoke:** Phase 1 (building the spec categories), and whenever we add or modify a spec.

**Dependencies:** None for hand-coded monitors. Spot (`pip install spot`) for automated LTL-to-automaton compilation and cross-checking.

### `evaluation`

**Purpose:** Runs multi-turn rollouts of a model against a monitor and collects adherence statistics.

**Input:** A model path (base or fine-tuned checkpoint), a monitor class, a simulated-user configuration (number of conversations, turns per conversation, user prompt templates), and a system prompt describing the constraint in natural language.

**Output:**
- A CSV log: one row per conversation, columns for conversation ID, turn of first violation (or "none"), total reward, raw transcript.
- Summary statistics: adherence rate (% of conversations with zero violations), mean turns before first violation, reward distribution.
- A JSON file with all transcripts for manual inspection.

**When to invoke:** Phase 2 (baseline evaluation), Phase 3 (evaluation at each training checkpoint), Phase 4 (final comparison).

**Dependencies:** `transformers`, `torch`, the `monitor` skill's output.

### `training`

**Purpose:** Produces a complete GRPO (or DPO fallback) training script wired to a monitor's reward signal.

**Input:** A model name (e.g., `Qwen/Qwen2.5-1.5B-Instruct`), a monitor class, and a hyperparameter config (LoRA rank, learning rate, group size G, number of training steps K, rollout length, checkpoint interval).

**Output:**
- A Python training script using TRL's `GRPOTrainer` with the monitor as the reward function.
- A LoRA configuration via PEFT.
- Checkpoint saving at specified intervals.
- Integration with the `evaluation` skill: runs the evaluation harness at each checkpoint and logs adherence to a CSV.

**Fallback:** If GRPO proves unstable, the skill can generate a DPO script instead. DPO takes pairs of traces (one satisfying φ, one violating), generated by the model and labeled by the monitor, and fine-tunes with a contrastive loss. Simpler but less expressive.

**When to invoke:** Phase 3.

**Dependencies:** `trl`, `peft`, `transformers`, `accelerate`, `torch`, the `monitor` skill's output.

### `slurm`

**Purpose:** Generates correct Slurm batch scripts for any job on your cluster.

**Input:** A Python script to run, resource requirements (GPUs, memory, wall time), and the cluster configuration (partition name, module loads, conda environment path, internet access from compute nodes).

**Output:** A `.sbatch` file with correct headers, environment setup, and the `srun` command. Handles: module loading, conda activation, GPU allocation, output/error log paths, job naming.

**Configuration:** This skill needs your cluster details before it can produce anything. Once configured, it's a one-liner to generate a batch script for any phase of the project.

**When to invoke:** Every phase — any time we need to run something on the cluster.

**Dependencies:** Your answers to the cluster questions below.

### `plotting`

**Purpose:** Takes evaluation CSVs and produces publication-quality figures.

**Input:** One or more CSV files from the `evaluation` skill, plus a figure type (bar chart, learning curve, comparison table).

**Output:** PDF/PNG figures and the matplotlib script that generated them. Specific figures planned:
- **Figure 1 (the key result):** Grouped bar chart — baseline vs. post-GRPO adherence per spec, grouped by category (A vs. B). Highlight the A5 vs. B1 matched-difficulty comparison.
- **Figure 2 (learning curves):** Adherence vs. training step, one line per spec, colored by category.
- **Figure 3 (LoRA rank ablation):** Adherence vs. LoRA rank for A5 and B1 — does capacity close the gap?
- **Figure 4 (model scale):** A5 vs. B1 at 1.5B and 7B — does the boundary hold?
- **Table 1:** Full numerical results with 95% confidence intervals.

**When to invoke:** Phase 4, but also useful during Phase 3 to eyeball learning curves as training progresses.

**Dependencies:** `matplotlib`, `pandas`.

### Skill Dependency Graph

```
Phase 0:  slurm (configure) ──────────────────────────────────┐
Phase 1:  monitor (build spec categories)                      │
Phase 2:  evaluation (baseline) ← monitor + slurm             │
Phase 3:  training (GRPO) ← monitor + evaluation + slurm      │
Phase 4:  plotting (figures) ← evaluation outputs              │
          All phases use slurm ←───────────────────────────────┘
```

### What I Write vs. What You Review

| Skill | I write | You review |
|-------|---------|------------|
| `monitor` | Python classes, unit tests | Correctness of the automata against the formal specs |
| `evaluation` | Rollout harness, scoring, statistics | Fairness of prompts, sanity of results |
| `training` | GRPO/DPO scripts, reward wiring | Loss curves, adherence curves, anything weird |
| `slurm` | Batch scripts | That they actually run on your cluster |
| `plotting` | Figure scripts | That the figures tell the right story |

---

## Open Questions (For You)

1. **Cluster details:** GPU type, VRAM, wall-time limit, module system?
2. **Do you want to target a specific venue?** This affects framing and page limits.
3. **The dpm connection:** Do you want this in the paper (as a section or future work), or as a separate follow-up?
4. **Collaboration:** Is this solo, or do you want to involve Maria Christakis or anyone else?
5. **Timeline:** Is the 7-week plan realistic given your other commitments (the PLDI rebuttal, the IST transition)?
