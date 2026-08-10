"""
QASM <-> graph conversion shared by Stage 1 and Stage 2.

Representation (this is the "dynamic structure" piece — the graph's node/edge
count is exactly the circuit's gate/dependency count, so the same GNN weights
apply unchanged to circuits of any qubit count or depth; nothing is padded to
a fixed size):

  Node   = one gate instance: (gate_type_one_hot, params, qubit_indices, wire_position)
  Edge   = "happens-before-on-the-same-wire" — connects gate i to the next gate
           that touches any qubit gate i also touches (the standard circuit DAG).
           Two-qubit gates therefore create edges that cross wires, which is
           exactly the information the GNN needs to reason about commutation.

Qubit index is passed both as a node feature (so the model can condition on
"which wire is this on") and is recoverable per-node for Stage 2's per-wire
pooling.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit import Instruction
from torch_geometric.data import Data

# Fixed, small, closed vocabulary of gate types the model can appear in MNISQ-
# derived circuits. Unknown gates fall back to an "OTHER" slot rather than
# crashing — extend this list to cover your basis gate set.
GATE_VOCAB = [
    "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "u1", "u2", "u3",
    "cx", "cy", "cz", "ch", "swap", "crz", "cu1", "cu3", "ccx",
    "measure", "barrier",
    "OTHER",
]
GATE_INDEX = {name: i for i, name in enumerate(GATE_VOCAB)}
NUM_GATE_TYPES = len(GATE_VOCAB)
MAX_GATE_PARAMS = 3  # covers u3(theta, phi, lambda); zero-padded otherwise


@dataclass
class CircuitGraph:
    data: Data              # PyG graph: x=node features, edge_index=DAG edges
    n_qubits: int
    node_qubits: list[list[int]]   # per node: which qubit indices it touches
    node_gate_names: list[str]     # per node: original gate name (for reconstruction)


def _node_features(op: Instruction, qubits: list[int], n_qubits: int) -> torch.Tensor:
    gate_idx = GATE_INDEX.get(op.name, GATE_INDEX["OTHER"])
    one_hot = torch.zeros(NUM_GATE_TYPES)
    one_hot[gate_idx] = 1.0

    params = list(op.params) + [0.0] * MAX_GATE_PARAMS
    params = torch.tensor([float(p) for p in params[:MAX_GATE_PARAMS]])

    # Normalized qubit-position features: which wires this node touches,
    # expressed as a fraction of n_qubits so the feature scale is
    # architecture-independent regardless of circuit width.
    qubit_feat = torch.zeros(2)  # [min_qubit_frac, max_qubit_frac]
    if qubits:
        qubit_feat[0] = min(qubits) / max(n_qubits - 1, 1)
        qubit_feat[1] = max(qubits) / max(n_qubits - 1, 1)

    is_two_qubit = torch.tensor([1.0 if len(qubits) >= 2 else 0.0])

    return torch.cat([one_hot, params, qubit_feat, is_two_qubit])


NODE_FEATURE_DIM = NUM_GATE_TYPES + MAX_GATE_PARAMS + 2 + 1


def qasm_to_graph(qasm: str) -> CircuitGraph:
    circ = qasm2.loads(qasm)
    n_qubits = circ.num_qubits

    node_feats: list[torch.Tensor] = []
    node_qubits: list[list[int]] = []
    node_gate_names: list[str] = []
    src_edges: list[int] = []
    dst_edges: list[int] = []

    last_node_on_wire: dict[int, int] = {}  # qubit index -> most recent node id

    for node_id, instruction in enumerate(circ.data):
        op = instruction.operation
        qubits = [circ.find_bit(q).index for q in instruction.qubits]

        node_feats.append(_node_features(op, qubits, n_qubits))
        node_qubits.append(qubits)
        node_gate_names.append(op.name)

        for q in qubits:
            if q in last_node_on_wire:
                src_edges.append(last_node_on_wire[q])
                dst_edges.append(node_id)
            last_node_on_wire[q] = node_id

    x = torch.stack(node_feats) if node_feats else torch.zeros((0, NODE_FEATURE_DIM))
    edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long) if src_edges else torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, num_nodes=len(node_feats))

    return CircuitGraph(
        data=data,
        n_qubits=n_qubits,
        node_qubits=node_qubits,
        node_gate_names=node_gate_names,
    )
