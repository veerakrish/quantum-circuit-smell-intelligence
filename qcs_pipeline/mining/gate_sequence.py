"""
Shared gate-sequence representation used by mining, canonicalization, and the
detector.

A GateOp is the atomic unit both the miner (diffing raw vs. transpiled) and
the detector (matching rules against a new circuit) operate on. Multi-qubit
gates are represented as ONE GateOp spanning multiple qubits — this keeps a
2-qubit gate atomic through diffing/matching rather than accidentally
splitting it across two independent per-wire streams.
"""
from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit


@dataclass(frozen=True)
class GateOp:
    name: str
    qubits: tuple[int, ...]     # absolute qubit indices in the source circuit
    params: tuple[float, ...]   # rounded to PARAM_PRECISION for stable comparison/hashing

    def diff_key(self) -> tuple:
        """Key used for difflib alignment — includes concrete qubits/params, so
        an actual identical gate instance compares equal, not just same type."""
        return (self.name, self.qubits, self.params)


PARAM_PRECISION = 9  # decimal places; matches float round-trip safety margin


def circuit_to_gate_ops(circ: QuantumCircuit) -> list[GateOp]:
    ops: list[GateOp] = []
    for instruction in circ.data:
        op = instruction.operation
        if op.name in ("barrier", "measure"):
            continue  # not part of the unitary — irrelevant to gate-cancellation mining
        qubits = tuple(circ.find_bit(q).index for q in instruction.qubits)
        params = tuple(round(float(p), PARAM_PRECISION) for p in op.params)
        ops.append(GateOp(name=op.name, qubits=qubits, params=params))
    return ops


def gate_ops_to_circuit(ops: list[GateOp], n_qubits: int, n_clbits: int = 0) -> QuantumCircuit:
    """Rebuild an executable circuit from a GateOp sequence (used after the
    detector rewrites a matched window)."""
    circ = QuantumCircuit(n_qubits, n_clbits)
    for op in ops:
        circ.append(_instruction_for(op.name, op.params), op.qubits)
    return circ


def _instruction_for(name: str, params: tuple[float, ...]):
    from qiskit.circuit.library import standard_gates  # local import: avoid import cost when unused

    gate_cls_map = {
        "id": standard_gates.IGate, "x": standard_gates.XGate, "y": standard_gates.YGate,
        "z": standard_gates.ZGate, "h": standard_gates.HGate, "s": standard_gates.SGate,
        "sdg": standard_gates.SdgGate, "t": standard_gates.TGate, "tdg": standard_gates.TdgGate,
        "rx": standard_gates.RXGate, "ry": standard_gates.RYGate, "rz": standard_gates.RZGate,
        "cx": standard_gates.CXGate, "cy": standard_gates.CYGate, "cz": standard_gates.CZGate,
        "ch": standard_gates.CHGate, "swap": standard_gates.SwapGate, "crz": standard_gates.CRZGate,
        "ccx": standard_gates.CCXGate,
    }
    cls = gate_cls_map.get(name)
    if cls is None:
        raise ValueError(f"Unsupported gate '{name}' in rewrite — extend gate_cls_map in gate_sequence.py")
    return cls(*params) if params else cls()
