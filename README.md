# Quantum Circuit Optimization — Two Approaches

This repo holds two independent, self-contained pipelines for optimizing raw
MNISQ OpenQASM circuits, developed in sequence as the project's approach
evolved:

| | `qco_pipeline/` | `qcs_pipeline/` |
|---|---|---|
| Approach | Neural (GNN, two-stage decoupled model) | Rule-based ("smell intelligence" — mined rewrite rules + deterministic pattern matcher) |
| Learns optimization via | Supervised training on Qiskit-transpiler-labeled examples | Mining exact before/after diffs from Qiskit transpiler runs, then symbolic pattern matching — no training/gradient descent at all |
| Correctness guarantee | Statistical (checksum-gated, fallback on failure) | Exact — every rule application either reproduces fidelity 1.0 or is quarantined |
| Status | Working scaffold (see its own section below) | Working scaffold — **the current, actively developed approach** |

Both directories are fully independent (no shared imports except `qcs_pipeline`
reusing `qco_pipeline`'s Checksum 1 fidelity module — see below). Read on for
`qcs_pipeline/` (the smell-intelligence pipeline) first; the original neural
pipeline is documented further down.

---

# `qcs_pipeline/` — Quantum Circuit Smell Intelligence

Optimization here is not learned statistically — it's **mined** as exact,
provable gate-rewrite rules from Qiskit transpiler before/after diffs, then
applied via deterministic pattern matching (a "smell detector," directly
analogous to a software code-smell linter like SonarQube/ESLint, but matching
gate sequences instead of AST patterns). Because gate cancellation/merging is
fundamentally a symbolic-algebra fact (`H·H=I`, `RZ(a)·RZ(b)=RZ(a+b)`), not a
statistical pattern, this sidesteps the whole "train a model to approximate
an exact identity" problem the neural pipeline runs into.

## Pipeline

```
raw QASM (MNISQ)
      │
      ▼
Step 1 — mining (mining/pair_diff.py)
      │  Qiskit-transpile each raw circuit; diff (raw, transpiled) gate
      │  sequences (difflib/LCS alignment) to extract removed-gate windows
      │  with local context.
      ▼
Step 2 — canonicalization (rules/canonicalize.py, rules/rule_database.py)
      │  Abstract away literal qubit indices (-> relative roles) and literal
      │  angle values (-> symbolic relations: copy/sum/difference/negate).
      │  Dedup + frequency-rank into a reusable rule database (JSON).
      ▼
Smell detector (detector/wire_matcher.py, detector/smell_detector.py)
      │  Match rules against a (possibly unseen) circuit using TRUE WIRE
      │  ADJACENCY — not raw instruction-list position — so a match is only
      │  accepted where it's actually a valid local rewrite.
      ▼
Bisection repair (verification/bisect_repair.py)
      │  Exact state-fidelity check (reuses qco_pipeline's Checksum 1).
      │  On failure, recursively bisect the applied-rule set to isolate and
      │  quarantine the specific unsound rule — never "calibrate" the output
      │  numerically; a true identity must reproduce fidelity 1.0 exactly.
      ▼
optimized QASM (verified — same qubit count, exact fidelity)
```

## Layout

```
qcs_pipeline/
├── mining/
│   ├── gate_sequence.py       # QASM <-> GateOp list (shared representation)
│   ├── pair_diff.py           # Step 1: difflib-based raw/transpiled diff -> mined patterns
│   └── build_mined_dataset.py # CLI: MNISQ dir -> mined_pairs.jsonl
├── rules/
│   ├── canonicalize.py        # Step 2: relative-qubit + symbolic-param canonicalization
│   └── rule_database.py       # dedup, frequency ranking, JSON persistence
├── detector/
│   ├── wire_matcher.py        # wire-adjacency-aware pattern matching (correctness-critical)
│   └── smell_detector.py      # detect() for review-only, apply_rules() to rewrite
├── verification/
│   └── bisect_repair.py       # exact fidelity gate + bisection-based rule quarantine
├── pipeline.py                 # QuantumCircuitSmellOptimizer — end-to-end inference
└── cli.py                      # `mine` / `build-rules` / `optimize` subcommands
```

## Usage

```bash
pip install -r requirements.txt

# Step 1 — mine raw/transpiled diff patterns from a directory of .qasm files
python -m qcs_pipeline.cli mine --raw-dir data/mnisq_raw --out data/mined_pairs.jsonl

# Step 2 — canonicalize into a deduped, frequency-ranked rule database
python -m qcs_pipeline.cli build-rules --mined data/mined_pairs.jsonl --out data/rules.json --min-frequency 2

# Apply the smell detector + verified repair to a circuit
python -m qcs_pipeline.cli optimize --rules data/rules.json --circuit path/to/circuit.qasm --out optimized.qasm
```

```python
from pathlib import Path
from qcs_pipeline.pipeline import QuantumCircuitSmellOptimizer

optimizer = QuantumCircuitSmellOptimizer(rule_db_path=Path("data/rules.json"))

# Review only — flag smells without touching the circuit
smells = optimizer.detect_smells(open("circuit.qasm").read())

# Optimize with exact-fidelity verification + automatic bad-rule quarantine
result = optimizer.optimize(open("circuit.qasm").read())
print(result.n_gates_before, "->", result.n_gates_after, "gates")
print("fidelity:", result.fidelity)
print("quarantined this run:", result.quarantined_rule_count)
```

### Running on Kaggle

`notebooks/` has three separate notebooks, one per stage, chained through
Kaggle's own "attach another notebook's saved output as an Input" mechanism
rather than one combined file — each is small enough to read and iterate on
independently, and a Stage 3 bug fix doesn't require replaying Stage 1/2.

1. **`01_mining.ipynb`** — mines `mined_pairs.jsonl` from the
   `veerukhannan/mnisq-optbench-pairs` dataset's pre-transpiled pairs (~10 min
   on Kaggle's 4-core CPU notebooks). Save a version when done.
2. **`02_rule_building.ipynb`** — attach `01_mining`'s saved output as an
   Input; canonicalizes into `rules.json` (~40 min — the expensive stage).
   Save a version when done.
3. **`03_optimize.ipynb`** — attach both the original dataset (for sample
   circuits) and `02_rule_building`'s saved output; runs the verified
   optimizer and reports results (seconds — the stage worth iterating on).

Each notebook's setup cell states exactly which Input(s) it needs and fails
fast with a clear message if one isn't attached.

## Known limitations / things to validate before production use

- **Step 1's diff is approximate.** `mining/pair_diff.py` diffs the raw
  circuit's instruction-LIST order, not true wire adjacency — documented
  in-file. This only affects which CANDIDATE patterns get mined; it does not
  affect correctness of what gets applied, because the detector re-verifies
  true wire adjacency independently (`detector/wire_matcher.py`) before ever
  matching a rule against a new circuit.
- **Unresolved param relations are structural-only.** If `canonicalize.py`
  can't express a replacement gate's angle as a simple copy/sum/difference of
  the removed gates' angles, the rule is kept (its gate-removal structure is
  still useful) but marked `UNRESOLVED`; `smell_detector.py` raises rather
  than guessing a value if such a rule is ever matched with a nonempty
  rewrite. In practice this should only matter for merge-type rules, never
  pure-cancellation ones (which have no rewrite params at all).
- **Conflicting rules are never auto-applied.** If the same canonical pattern
  mapped to more than one distinct rewrite during mining, both are kept in
  the database (for `detect()`/audit visibility) but `apply_rules()` skips
  them — see `rule_database.py`'s docstring for why a conflict usually means
  "needs more context to resolve," not "pick the majority vote."
- **Bisection assumes rule independence** (documented in
  `bisect_repair.py`): true identities applied to disjoint node sets compose
  safely. The final assembled circuit is still re-verified defensively after
  bisection completes, and falls back to the unmodified raw circuit (logged
  as an error) in the unlikely case that assumption is violated.

---

# `qco_pipeline/` — Two-Stage Decoupled Neural Model

Trains and runs a two-stage neural pipeline that optimizes raw MNISQ OpenQASM
circuits: **Stage 1 (Horizontal)** cancels/merges gates without changing qubit
count, **Stage 2 (Vertical)** prunes idle wires and re-indexes the survivors
to a contiguous register. Every model output is cross-checked against the
raw circuit's quantum state before being trusted, with an automatic classical
fallback on any verification failure.

## Why "balanced"

Two distinct senses of balance are handled explicitly, not left implicit:

1. **Class-imbalance balance** (`training/losses.py`) — inverse-frequency
   class weighting for Stage 1's 3-way action label (KEEP dominates
   CANCEL/MERGE in real circuits) and `pos_weight` for Stage 2's prune/keep
   binary label (prune events are the rare class). Without this the model
   collapses to "always predict the majority class" while still showing a
   deceptively low loss.
2. **Compression-vs-fidelity balance** — a model can trivially "win" by
   deleting everything (max compression, zero fidelity) or deleting nothing
   (max fidelity, zero compression). Supervision targets are fidelity-exact
   by construction (Phase 0 labels come from a deterministic classical
   compiler / explicit rewrite rules that never alter the circuit's unitary),
   and checkpoint selection during training uses **Checksum pass rate on a
   held-out split**, not just loss — this is what actually keeps training
   balanced between over- and under-aggressive rewriting in practice.

## Pipeline

```
raw QASM (MNISQ)
      |
      v
Phase 0  ──  deterministic classical compiler (Qiskit transpile, PyZX)
      |      produces (X_raw -> Y_horiz) and (Y_horiz -> Y_balanced) pairs
      v
Stage 1 (GNN, node classification)  ──  Checksum 1 (state fidelity, same n)
      |                                        |
      | pass                                fail -> classical fallback
      v
Stage 2 (GNN, per-wire pruning)     ──  Checksum 2 (partial trace + Uhlmann fidelity, n -> k)
      |                                        |
      | pass                                fail -> classical fallback
      v
balanced QASM (k <= n qubits, semantically verified)
```

## Layout

```
qco_pipeline/
├── phase0/               # Dataset 1 & 2 generation + Stage-1 node labels
│   ├── horizontal_pairs.py
│   ├── horizontal_labels.py
│   ├── vertical_pairs.py
│   └── build_dataset.py  # CLI: raw QASM dir -> phase0/*.jsonl
├── graph/
│   └── circuit_graph.py  # QASM -> PyTorch Geometric graph (dynamic size, fixed weights)
├── models/
│   ├── stage1_horizontal.py
│   ├── stage2_vertical.py
│   ├── apply_stage1.py   # model output -> QASM
│   └── apply_stage2.py   # model output -> QASM
├── verification/
│   ├── checksum1.py      # same-dimension state fidelity
│   └── checksum2.py      # partial trace + permutation + Uhlmann fidelity
├── training/
│   ├── losses.py          # balanced (class-weighted) loss functions
│   ├── train_stage1.py
│   └── train_stage2.py
├── fallback/
│   └── classical_fallback.py  # deterministic fallback + JSONL diagnostics
└── pipeline.py            # end-to-end inference orchestration
```

## Usage

```bash
pip install -r requirements.txt

# 1. Phase 0 — build training pairs from a directory of raw MNISQ .qasm files
python -m qco_pipeline.phase0.build_dataset --raw-dir data/mnisq_raw --out-dir data/phase0

# 2. Train Stage 1
python -m qco_pipeline.training.train_stage1 \
    --dataset data/phase0/horizontal_pairs.jsonl --out checkpoints/stage1.pt

# 3. Train Stage 2
python -m qco_pipeline.training.train_stage2 \
    --dataset data/phase0/vertical_pairs.jsonl --out checkpoints/stage2.pt
```

```python
# 4. Run inference
from pathlib import Path
from qco_pipeline.pipeline import QuantumCircuitOptimizer

optimizer = QuantumCircuitOptimizer(
    stage1_ckpt=Path("checkpoints/stage1.pt"),
    stage2_ckpt=Path("checkpoints/stage2.pt"),
    diagnostics_path=Path("logs/diagnostics.jsonl"),
)
result = optimizer.optimize(raw_qasm=open("circuit.qasm").read(), source_name="circuit.qasm")
print(result.n_qubits_in, "->", result.n_qubits_out, "qubits")
print("Stage 1 fallback used:", result.stage1_used_fallback)
print("Stage 2 fallback used:", result.stage2_used_fallback)
print(result.balanced_qasm)
```

## Known limitations / things to validate before production use

- `verification/checksum2._permute_density_matrix`'s manual `np.transpose`
  fallback path is unexercised by the current pipeline (kept_qubits is always
  already in ascending order by construction) and has **not** been numerically
  verified against Qiskit's little-endian qubit convention — only exercise it
  if you build a Stage 2 variant that reorders wires.
- Fidelity verification (`checksum1.py`, `checksum2.py`) uses full statevector
  / density-matrix simulation, which is exponential in qubit count — capped
  at `MAX_QUBITS_FOR_STATEVECTOR = 24` by default. For MNISQ-scale benchmarks
  this is normally fine; for larger circuits you'll need a sampling-based
  fidelity estimator instead.
- `phase0/vertical_pairs.prune_idle_wires` only detects *untouched* wires
  (a sound but conservative subset of "zero-contribution" wires). It does not
  attempt to detect entangled-but-separable wires — Checksum 2 is what
  protects you if a future, more aggressive pruning heuristic over-claims.
