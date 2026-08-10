"""
Qiskit QASM2 compatibility shim.

qiskit.qasm2.loads() (the replacement for the removed QuantumCircuit.from_qasm_str())
runs a strict parser that requires every gate to be formally defined in scope.
The standard `qelib1.inc` header these QASM strings reference never actually
defined `sx`/`sxdg` — Qiskit's old, now-removed .from_qasm_str() tolerated
them anyway via a hardcoded legacy gate list. Circuits from datasets exported
with that old exporter fail strict parsing on every such gate ("'sx' is not
defined in this scope") unless we opt back into that same tolerance.

Route all QASM2 load/dump calls through this module instead of calling
qiskit.qasm2 directly, so there's exactly one place to fix if another such
parser gap turns up. (Mirrors qcs_pipeline/qasm_compat.py — duplicated
rather than shared, since the two packages are otherwise fully independent.)
"""
from __future__ import annotations

from qiskit import qasm2
from qiskit.circuit import QuantumCircuit


def loads(qasm: str) -> QuantumCircuit:
    return qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def dumps(circ: QuantumCircuit) -> str:
    return qasm2.dumps(circ)
