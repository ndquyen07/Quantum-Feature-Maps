"""Shot-based quantum simulation primitives (no density matrices or statevectors)."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)
from typing import List, Optional


def create_noise_model(
    depol_1q: float = 0.0,
    depol_2q: float = 0.0,
    t1: Optional[float] = None,
    t2: Optional[float] = None,
    gate_time_1q: float = 50e-9,
    gate_time_2q: float = 300e-9,
    readout_error: float = 0.0,
) -> Optional[NoiseModel]:
    """
    Build a Qiskit Aer NoiseModel.

    """
    has_noise = (
        depol_1q > 0 or depol_2q > 0
        or (t1 is not None and t1 > 0)
        or readout_error > 0
    )
    if not has_noise:
        return None

    noise_model = NoiseModel()

    # Single-qubit gate targets in your circuits: ry, rz
    single_qubit_gates = ["ry", "rz", "u", "u1", "u2", "u3"]
    two_qubit_gates    = ["cx", "cz"]

    #  depolarizing errors
    if depol_1q > 0:
        err_1q = depolarizing_error(depol_1q, 1)
        noise_model.add_all_qubit_quantum_error(err_1q, single_qubit_gates)

    if depol_2q > 0:
        err_2q = depolarizing_error(depol_2q, 2)
        noise_model.add_all_qubit_quantum_error(err_2q, two_qubit_gates)

    #  thermal relaxation errors 
    if t1 is not None and t1 > 0:
        t2_val = t2 if (t2 is not None and t2 > 0) else t1  # default T2 = T1

        err_therm_1q = thermal_relaxation_error(t1, t2_val, gate_time_1q)
        # 2-qubit thermal error = tensor product of two independent 1-qubit errors
        err_therm_2q = thermal_relaxation_error(t1, t2_val, gate_time_2q).expand(
            thermal_relaxation_error(t1, t2_val, gate_time_2q)
        )

        noise_model.add_all_qubit_quantum_error(err_therm_1q, single_qubit_gates)
        noise_model.add_all_qubit_quantum_error(err_therm_2q, two_qubit_gates)

    #  readout errors
    if readout_error > 0:
        p = readout_error
        ro_err = ReadoutError([[1 - p, p], [p, 1 - p]])
        noise_model.add_all_qubit_readout_error(ro_err)

    return noise_model


def create_sampler(shots: int = 1024, noise_model: Optional[NoiseModel] = None):
    """
    Create an AerSampler, optionally with a noise model.

    """
    if noise_model is not None:
        return AerSampler(
            backend_options={"noise_model": noise_model},
            run_options={"shots": shots},
        )
    return AerSampler(run_options={"shots": shots})


def bind_data_circuit(circuit_with_theta: QuantumCircuit, data_params: List, sample: np.ndarray):
    return circuit_with_theta.assign_parameters(dict(zip(data_params, sample)))


def batch_measure_probabilities(
    circuit_with_theta: QuantumCircuit,
    data_params: List,
    samples: np.ndarray,
    sampler,
) -> List[dict]:
    if len(samples) == 0:
        return []

    # Build the measurement circuit once (theta bound, data params free), then bind
    # every sample as a batch of parameter values so Aer transpiles a single circuit
    # instead of copying + measuring a fresh circuit per sample.
    template = circuit_with_theta.copy()
    template.measure_all()

    pos = {p: k for k, p in enumerate(data_params)}
    order = [pos[p] for p in template.parameters]
    parameter_values = [[sample[idx] for idx in order] for sample in samples]

    quasi_dists = sampler.run(
        [template] * len(samples), parameter_values=parameter_values
    ).result().quasi_dists
    return list(quasi_dists)


def fidelity_to_basis_state(quasi_dist: dict, basis_label: str) -> float:
    return float(quasi_dist.get(int(basis_label, 2), 0.0))


def _build_inversion_test_circuit(
    psi_circ: QuantumCircuit,
    phi_circ: QuantumCircuit,
    num_qubits: int,
) -> QuantumCircuit:
    """
    Build inversion-test circuit: |0> -> U_psi -> U_phi^dag -> measure.
    P(|0...0>) = |<psi|phi>|^2.
    """
    qc = QuantumCircuit(num_qubits)
    qc.compose(psi_circ, inplace=True)
    qc.compose(phi_circ.inverse(), inplace=True)
    qc.measure_all()
    return qc


def inversion_test_overlap_samples(
    circuit_with_theta: QuantumCircuit,
    data_params: List,
    sample1: np.ndarray,
    sample2: np.ndarray,
    sampler,
) -> float:
    """Estimate |<psi(x1)|psi(x2)>|^2 via inversion test."""
    num_qubits = len(data_params)
    psi_circ = bind_data_circuit(circuit_with_theta, data_params, sample1)
    phi_circ = bind_data_circuit(circuit_with_theta, data_params, sample2)
    qc = _build_inversion_test_circuit(psi_circ, phi_circ, num_qubits)

    p0 = sampler.run(qc).result().quasi_dists[0].get(0, 0.0)
    return float(np.clip(p0, 0.0, 1.0))


def _build_parametrized_inversion_template(
    circuit_with_theta: QuantumCircuit,
    data_params: List,
    num_qubits: int,
):
    """Parametrized inversion-test template U(x_a) . U(x_b)^dag with measurement.

    theta is already bound in circuit_with_theta. The first sample's data params
    are the original data_params (x_a); the second sample's are a fresh vector
    (x_b). Built once per batch so Aer transpiles a single circuit and binds all
    pairs as parameter values, instead of composing/inverting a circuit per pair.
    """
    x_b = list(ParameterVector("x_inv", num_qubits))
    phi_circ = circuit_with_theta.assign_parameters(
        {data_params[k]: x_b[k] for k in range(num_qubits)}
    )
    template = QuantumCircuit(num_qubits)
    template.compose(circuit_with_theta, inplace=True)   # U(x_a), theta bound
    template.compose(phi_circ.inverse(), inplace=True)   # U(x_b)^dag
    template.measure_all()
    return template, x_b


def batch_inversion_test_overlaps(
    circuit_with_theta: QuantumCircuit,
    data_params: List,
    pairs: List[tuple],
    sampler,
) -> List[float]:
    if not pairs:
        return []

    num_qubits = len(data_params)
    template, x_b = _build_parametrized_inversion_template(
        circuit_with_theta, data_params, num_qubits
    )

    # Order each pair's [x_a, x_b] values to match template.parameters, then bind
    # the whole batch at once (single transpile inside Aer).
    xa_pos = {p: k for k, p in enumerate(data_params)}
    xb_pos = {p: k for k, p in enumerate(x_b)}
    spec = [
        (0, xa_pos[p]) if p in xa_pos else (1, xb_pos[p])
        for p in template.parameters
    ]
    parameter_values = [
        [(s1 if sel == 0 else s2)[idx] for sel, idx in spec]
        for s1, s2 in pairs
    ]

    quasi_dists = sampler.run(
        [template] * len(pairs), parameter_values=parameter_values
    ).result().quasi_dists
    return [float(np.clip(qd.get(0, 0.0), 0.0, 1.0)) for qd in quasi_dists]


def ensemble_overlap_samples(
    samples1: np.ndarray,
    samples2: np.ndarray,
    circuit_with_theta: QuantumCircuit,
    data_params: List,
    sampler,
) -> float:
    if len(samples1) == 0 or len(samples2) == 0:
        return 0.0
    pairs = [(s1, s2) for s1 in samples1 for s2 in samples2]
    overlaps = batch_inversion_test_overlaps(circuit_with_theta, data_params, pairs, sampler)
    return float(np.mean(overlaps))


def ensemble_trace_distance_samples(
    samples1: np.ndarray,
    samples2: np.ndarray,
    circuit_with_theta: QuantumCircuit,
    data_params: List,
    sampler,
) -> float:
    if len(samples1) == 0 or len(samples2) == 0:
        return 0.0
    pairs = [(s1, s2) for s1 in samples1 for s2 in samples2]
    overlaps = batch_inversion_test_overlaps(circuit_with_theta, data_params, pairs, sampler)
    distances = [float(np.sqrt(max(0.0, 1.0 - ov))) for ov in overlaps]
    return float(np.mean(distances))


def compute_kernel_matrix_shots(
    X1: np.ndarray,
    X2: np.ndarray,
    theta_params: np.ndarray,
    circuit: QuantumCircuit,
    sampler,
) -> np.ndarray:
    n_features = X1.shape[1]
    data_params = list(circuit.parameters)[:n_features]
    theta_params_list = list(circuit.parameters)[n_features:]
    theta_dict = {p: float(v) for p, v in zip(theta_params_list, theta_params)}
    circuit_with_theta = circuit.assign_parameters(theta_dict)

    is_symmetric = X1 is X2 or np.array_equal(X1, X2)
    pairs = []
    index_map = []

    for i in range(X1.shape[0]):
        j_range = range(i, X2.shape[0]) if is_symmetric else range(X2.shape[0])
        for j in j_range:
            pairs.append((X1[i], X2[j]))
            index_map.append((i, j))

    overlaps = batch_inversion_test_overlaps(circuit_with_theta, data_params, pairs, sampler)

    kernel = np.zeros((X1.shape[0], X2.shape[0]))
    for (i, j), overlap in zip(index_map, overlaps):
        kernel[i, j] = overlap
        if is_symmetric and i != j:
            kernel[j, i] = overlap

    if is_symmetric:
        kernel = (kernel + kernel.T) / 2.0
        np.fill_diagonal(kernel, 1.0)

    return kernel
