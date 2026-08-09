"""Quantum Circuit Smell Intelligence — mined rewrite-rule pattern detector.

Where qco_pipeline/ (the earlier work) learns optimization statistically via
neural nets, this package mines EXACT, provable rewrite rules from Qiskit
transpiler before/after diffs, then applies them via deterministic pattern
matching (a "smell detector," analogous to a code-smell linter) with an
exact-fidelity verification + bisection-repair loop instead of a trained
model's approximate confidence.
"""
