"""
Checksum 2 — Stage 2 gate: cross-dimension verification via partial trace.

Stage 2 changes qubit count (k < n), so a direct state-vector inner product is
undefined (mismatched Hilbert space dimensions: 2^n vs 2^k). Per the spec:

  1. Partial Trace : rho_reduced = Tr_B(|psi_raw><psi_raw|)   — trace out the
     pruned ("B") registers from the raw circuit's density matrix.
  2. Permutation    : align rho_reduced's wire order with the balanced
     circuit's wire order via permutation matrix P.
  3. Fidelity       : Tr( sqrt( sqrt(rho_stage2) . rho_reduced . sqrt(rho_stage2) ) )^2
                       >= 1 - 1e-6   (Uhlmann fidelity between mixed states)

qiskit.quantum_info.state_fidelity implements exactly this Uhlmann formula for
DensityMatrix inputs (see `_manual_uhlmann_fidelity` below for a literal,
spec-matching re-derivation used only to cross-check the library call in
tests — don't use it in the hot path, it's O(k^3) slower via scipy.sqrtm).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
from scipy.linalg import sqrtm

logger = logging.getLogger(__name__)

FIDELITY_THRESHOLD = 1 - 1e-6
MAX_QUBITS_FOR_STATEVECTOR = 24


@dataclass
class Checksum2Result:
    passed: bool
    fidelity: float
    n_qubits_raw: int
    n_qubits_reduced: int
    reason: str = ""


def _strip_measurements(circ: QuantumCircuit) -> QuantumCircuit:
    stripped = circ.copy()
    stripped.remove_final_measurements(inplace=True)
    return stripped


def _manual_uhlmann_fidelity(rho: DensityMatrix, sigma: DensityMatrix) -> float:
    """Literal implementation of the spec's formula — reference/test only."""
    rho_mat = rho.data
    sigma_mat = sigma.data
    sqrt_rho = sqrtm(rho_mat)
    inner = sqrt_rho @ sigma_mat @ sqrt_rho
    sqrt_inner = sqrtm(inner)
    fidelity = np.real(np.trace(sqrt_inner)) ** 2
    return float(np.clip(fidelity, 0.0, 1.0))


def run_checksum2(raw_qasm: str, balanced_qasm: str, kept_qubits: list[int]) -> Checksum2Result:
    """
    kept_qubits: original raw-circuit qubit indices that survive pruning,
    listed in the order they appear in `balanced_qasm` (index i of this list
    is qubit i of the balanced circuit). This is exactly what
    models/apply_stage2.apply_stage2_mask / phase0/vertical_pairs.prune_idle_wires
    both return alongside the QASM string.
    """
    raw_circ = _strip_measurements(QuantumCircuit.from_qasm_str(raw_qasm))
    balanced_circ = _strip_measurements(QuantumCircuit.from_qasm_str(balanced_qasm))

    n = raw_circ.num_qubits
    k = balanced_circ.num_qubits

    if len(kept_qubits) != k:
        return Checksum2Result(
            passed=False, fidelity=0.0, n_qubits_raw=n, n_qubits_reduced=k,
            reason=f"kept_qubits length ({len(kept_qubits)}) != balanced circuit width ({k})",
        )
    if k >= n:
        return Checksum2Result(
            passed=False, fidelity=0.0, n_qubits_raw=n, n_qubits_reduced=k,
            reason="Stage 2 did not actually reduce qubit count — nothing to verify here; route to Checksum 1 instead",
        )
    if n > MAX_QUBITS_FOR_STATEVECTOR:
        return Checksum2Result(
            passed=False, fidelity=0.0, n_qubits_raw=n, n_qubits_reduced=k,
            reason=f"{n} qubits exceeds statevector verification budget ({MAX_QUBITS_FOR_STATEVECTOR})",
        )

    # 1. Partial trace: qiskit's `qargs` for partial_trace specifies which
    #    subsystems to KEEP is the inverse convention — partial_trace(state, qargs)
    #    traces OUT `qargs`. So we pass the *pruned* (non-kept) indices.
    pruned_qubits = [q for q in range(n) if q not in kept_qubits]
    try:
        psi_raw = Statevector.from_instruction(raw_circ)
    except Exception as exc:  # noqa: BLE001
        return Checksum2Result(passed=False, fidelity=0.0, n_qubits_raw=n, n_qubits_reduced=k, reason=f"Raw simulation error: {exc}")

    rho_reduced_native_order = partial_trace(psi_raw, pruned_qubits)
    # qiskit's partial_trace preserves the remaining subsystems in their
    # original ascending index order. `kept_qubits` is produced by our own
    # pipeline (vertical_pairs.py / apply_stage2.py) using the same ascending
    # filter, so no further reorder is needed *by construction of this
    # pipeline* — but we still build and apply the permutation explicitly so
    # this check remains correct if a future Stage 2 variant reorders wires
    # (e.g. picks the highest-fidelity contiguous relabeling instead of the
    # order-preserving one).
    native_order = sorted(kept_qubits)
    if native_order != kept_qubits:
        perm = [native_order.index(q) for q in kept_qubits]  # permutation matrix P, applied by relabeling
        rho_reduced = _permute_density_matrix(rho_reduced_native_order, perm)
    else:
        rho_reduced = rho_reduced_native_order

    # 2. Density matrix of the model's actual (dimension-reduced) output.
    try:
        rho_balanced = DensityMatrix.from_instruction(balanced_circ)
    except Exception as exc:  # noqa: BLE001
        return Checksum2Result(passed=False, fidelity=0.0, n_qubits_raw=n, n_qubits_reduced=k, reason=f"Balanced-circuit simulation error: {exc}")

    # 3. Uhlmann fidelity (qiskit's state_fidelity implements the sqrt-sqrt-sqrt
    #    trace formula from the spec directly for DensityMatrix inputs).
    fidelity = state_fidelity(rho_reduced, rho_balanced)
    passed = fidelity >= FIDELITY_THRESHOLD

    if not passed:
        logger.warning(
            "Checksum 2 FAILED: fidelity=%.10f < threshold=%.10f (n=%d -> k=%d, kept=%s)",
            fidelity, FIDELITY_THRESHOLD, n, k, kept_qubits,
        )

    return Checksum2Result(
        passed=passed,
        fidelity=fidelity,
        n_qubits_raw=n,
        n_qubits_reduced=k,
        reason="" if passed else f"fidelity {fidelity:.10f} below threshold {FIDELITY_THRESHOLD:.10f}",
    )


def _permute_density_matrix(rho: DensityMatrix, perm: list[int]) -> DensityMatrix:
    """
    Relabel subsystem order i -> perm[i].

    NOTE: this path is not exercised by the current pipeline (vertical_pairs.py
    and apply_stage2.py both always emit kept_qubits already in ascending
    order, so `native_order == kept_qubits` above and this function is never
    called in practice). It's kept for correctness if a future Stage 2 variant
    reorders wires instead of just filtering them. Before relying on it,
    validate against Qiskit's little-endian qubit/tensor-axis convention with
    a small hand-checked unit test — the manual np.transpose fallback below
    has NOT been numerically verified.
    """
    return rho.reshuffle(perm) if hasattr(rho, "reshuffle") else DensityMatrix(
        np.transpose(
            rho.data.reshape([2] * (2 * len(perm))),
            axes=[perm[i] for i in range(len(perm))] + [len(perm) + perm[i] for i in range(len(perm))],
        ).reshape(2 ** len(perm), 2 ** len(perm))
    )
