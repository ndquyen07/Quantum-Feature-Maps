import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

from src.shot_simulation import (
    create_sampler,
    create_noise_model,
    batch_measure_probabilities,
    fidelity_to_basis_state,
    bind_data_circuit,
    ensemble_overlap_samples,
    ensemble_trace_distance_samples,
    compute_kernel_matrix_shots,
)

FIDELITY_CLIP_MIN = 1e-7
FIDELITY_CLIP_MAX = 1.0


class LossStrategy(ABC):
    @staticmethod
    def prepare_targets(labels: np.ndarray, class_patterns: Optional[Dict],
                        num_qubits: int) -> Optional[Dict]:
        return None

    @staticmethod
    @abstractmethod
    def compute(class_samples: List[np.ndarray], num_classes: int, **kwargs) -> float:
        raise NotImplementedError


class HilbertSchmidtLoss(LossStrategy):
    @staticmethod
    def compute(class_samples: List[np.ndarray], num_classes: int, **kwargs) -> float:
        from src.shot_simulation import batch_inversion_test_overlaps

        sampler = kwargs['sampler']
        circuit_with_theta = kwargs['circuit_with_theta']
        data_params = kwargs['data_params']
        num_pairs = num_classes * (num_classes - 1) // 2
        if num_pairs == 0:
            return 0.0

        # Build all pairs for all (o_ii, o_jj, o_ij) blocks in one sampler.run() call.
        all_pairs: List[tuple] = []
        block_sizes: List[tuple] = []  # (n_ii, n_jj, n_ij) per class pair

        for i in range(num_classes - 1):
            for j in range(i + 1, num_classes):
                si, sj = class_samples[i], class_samples[j]
                ii_pairs = [(a, b) for a in si for b in si]
                jj_pairs = [(a, b) for a in sj for b in sj]
                ij_pairs = [(a, b) for a in si for b in sj]
                all_pairs.extend(ii_pairs)
                all_pairs.extend(jj_pairs)
                all_pairs.extend(ij_pairs)
                block_sizes.append((len(ii_pairs), len(jj_pairs), len(ij_pairs)))

        # Single batched execution across all class pairs
        all_overlaps = batch_inversion_test_overlaps(
            circuit_with_theta, data_params, all_pairs, sampler
        )

        # Unpack results
        loss = 0.0
        cursor = 0
        for (n_ii, n_jj, n_ij) in block_sizes:
            o_ii = float(np.mean(all_overlaps[cursor: cursor + n_ii]))
            cursor += n_ii
            o_jj = float(np.mean(all_overlaps[cursor: cursor + n_jj]))
            cursor += n_jj
            o_ij = float(np.mean(all_overlaps[cursor: cursor + n_ij]))
            cursor += n_ij
            loss += max(0.0, o_ii + o_jj - 2 * o_ij)

        return 1 - 0.5 * loss / num_pairs


class InnerLoss(LossStrategy):
    @staticmethod
    def prepare_targets(labels: np.ndarray, class_patterns: Optional[Dict],
                        num_qubits: int) -> Dict:
        if class_patterns is None:
            raise ValueError("class_patterns must be set for inner_loss")
        return {label: class_patterns[label] for label in labels}

    @staticmethod
    def compute(class_samples: List[np.ndarray], num_classes: int,
                inner_products: List[np.ndarray], **kwargs) -> float:
        class_means = [np.mean(fids) for fids in inner_products if len(fids) > 0]
        return 1.0 - np.mean(class_means) if class_means else 1.0


class LogLikelihoodMacroLoss(LossStrategy):
    @staticmethod
    def prepare_targets(labels: np.ndarray, class_patterns: Optional[Dict],
                        num_qubits: int) -> Dict:
        if class_patterns is None:
            raise ValueError("class_patterns must be set for log_likelihood_loss")
        return {label: class_patterns[label] for label in labels}

    @staticmethod
    def compute(class_samples: List[np.ndarray], num_classes: int,
                inner_products: List[np.ndarray], **kwargs) -> float:
        total_loss = [
            np.mean(-np.log(np.clip(fids, FIDELITY_CLIP_MIN, FIDELITY_CLIP_MAX)))
            for fids in inner_products if len(fids) > 0
        ]
        return np.mean(total_loss) if total_loss else 0.0


LOSS_STRATEGIES = {
    'hilbert_schmidt': HilbertSchmidtLoss,
    'pre_determine_inner': InnerLoss,
    'inner_product': InnerLoss,
    'log_likelihood_macro': LogLikelihoodMacroLoss,
}


class TrainableQuantumFeatureMap:
    """Trainable Quantum Feature Map with shot-based simulation only."""

    def __init__(self, depth: int = 1, type_ansatz: str = "RealAmplitudes",
                 type_loss: str = "hilbert_schmidt", warmup_iterations: int = 200,
                 sampler=None, shots: int = 1024, noise_model=None,
                 accuracy_check_interval: int = 50):
        self.depth = depth
        self.ansatz_type = type_ansatz
        self.optimizer = None
        self.type_loss = type_loss
        self.warmup_iterations = warmup_iterations
        self.shots = shots
        self.noise_model = noise_model
        self.sampler = sampler if sampler is not None else create_sampler(shots, noise_model=noise_model)
        # How often (in optimizer evaluations) to run the expensive validation-accuracy
        # check that drives best_params selection. Smaller = finer early-stopping grid,
        # but each check computes shot-based train+val kernels. Must be >= 1.
        self.accuracy_check_interval = max(1, int(accuracy_check_interval))

        self.num_qubits = None
        self.init_theta = None
        self.num_classes = None
        self.unique_labels = None
        self.circuit = None
        self.class_patterns = None
        self.loss_strategy = None
        self.target_states = None

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        self.optimal_params = None
        self.optimal_value = None
        self.best_params = None
        self.best_val_acc = 0.0

        self.best_params_qc_trace = None
        self.best_val_acc_qc_trace = 0.0
        self.best_rhos_qc_trace = None
        self.best_params_qc_hs = None
        self.best_val_acc_qc_hs = 0.0
        self.best_rhos_qc_hs = None
        self.best_params_qc_overlap = None
        self.best_val_acc_qc_overlap = 0.0
        self.best_rhos_qc_overlap = None

        self._reset_history()

    def choose_class_patterns(self, X: np.ndarray, y: np.ndarray) -> Dict[int, str]:
        unique_labels = np.unique(y)
        num_basis_states = 2 ** self.num_qubits

        if len(unique_labels) > num_basis_states:
            raise ValueError(
                f"Number of classes ({len(unique_labels)}) exceeds available basis states"
            )

        basis_states = [format(n, f'0{self.num_qubits}b') for n in range(num_basis_states)]
        class_indices = {label: np.where(y == label)[0] for label in unique_labels}
        class_samples = {label: X[idx] for label, idx in class_indices.items()}

        data_params = self.circuit.parameters[:X.shape[1]]
        theta_params = self.circuit.parameters[X.shape[1]:]
        circuit_with_theta = self.circuit.assign_parameters(dict(zip(theta_params, self.init_theta)))

        inner_products = np.zeros((len(unique_labels), num_basis_states))

        for label_idx, label in enumerate(unique_labels):
            quasi_dists = batch_measure_probabilities(
                circuit_with_theta, data_params, class_samples[label], self.sampler
            )
            prob_sum = np.zeros(num_basis_states)
            for quasi_dist in quasi_dists:
                for state_idx, prob in quasi_dist.items():
                    prob_sum[int(state_idx)] += prob
            inner_products[label_idx, :] = prob_sum / len(class_samples[label])

        final_assignment = {}
        for _ in range(len(unique_labels)):
            row_idx, col_idx = np.unravel_index(np.argmax(inner_products), inner_products.shape)
            label = unique_labels[row_idx]
            final_assignment[label] = basis_states[col_idx]
            inner_products[row_idx, :] = -np.inf
            inner_products[:, col_idx] = -np.inf

        return dict(sorted(final_assignment.items()))

    def _compute_class_data(
        self, X: np.ndarray, y: np.ndarray, circuit_with_theta: QuantumCircuit,
        data_params: List, target_patterns: Optional[Dict] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        class_samples = []
        all_fidelities = []
        class_indices = {label: np.where(y == label)[0] for label in self.unique_labels}

        for label in self.unique_labels:
            samples = X[class_indices[label]]
            class_fids = []

            if target_patterns is not None:
                quasi_dists = batch_measure_probabilities(
                    circuit_with_theta, data_params, samples, self.sampler
                )
                basis_label = target_patterns[label]
                class_fids = [fidelity_to_basis_state(qd, basis_label) for qd in quasi_dists]

            class_samples.append(samples)
            all_fidelities.append(np.array(class_fids))

        return class_samples, all_fidelities

    def _class_representatives(self, class_samples: List[np.ndarray]) -> List[np.ndarray]:
        """One representative training sample per class (first sample)."""
        return [samples[0].copy() for samples in class_samples]

    @staticmethod
    def _statevectors_for_samples(
        circuit_with_theta: QuantumCircuit, data_params: List, samples: np.ndarray,
    ) -> np.ndarray:
        """Exact statevectors |psi(x, theta)> for each data sample (no shots/noise)."""
        return np.array([
            Statevector.from_instruction(
                bind_data_circuit(circuit_with_theta, data_params, sample)
            ).data
            for sample in samples
        ])

    @staticmethod
    def _mean_pairwise_overlap(sv1: np.ndarray, sv2: np.ndarray) -> float:
        """Mean |<psi_i|psi_j>|^2 over all cross pairs, computed from statevectors."""
        if sv1.size == 0 or sv2.size == 0:
            return 0.0
        inner = sv1 @ sv2.conj().T
        return float(np.mean(np.abs(inner) ** 2))

    def _store_history(
        self, class_samples: List[np.ndarray], circuit_with_theta: QuantumCircuit,
        data_params: List, loss: float,
    ) -> None:
        # self_overlaps = [
        #     ensemble_overlap_samples(s, s, circuit_with_theta, data_params, self.sampler)
        #     for s in class_samples
        # ]
        # Pairwise class overlaps are a logged diagnostic only (not used for training
        # or model selection), so compute them exactly from statevectors instead of
        # shot-based inversion tests: same quantity, O(n) statevector sims + a matmul
        # per class pair instead of O(n^2) circuit executions.
        class_statevectors = [
            self._statevectors_for_samples(circuit_with_theta, data_params, samples)
            for samples in class_samples
        ]
        pairwise_overlaps_dict = {
            f"{self.unique_labels[i]}_{self.unique_labels[j]}": self._mean_pairwise_overlap(
                class_statevectors[i], class_statevectors[j]
            )
            for i in range(self.num_classes - 1) for j in range(i + 1, self.num_classes)
        }
        # pairwise_distances_dict = {
        #     f"{self.unique_labels[i]}_{self.unique_labels[j]}": ensemble_trace_distance_samples(
        #         class_samples[i], class_samples[j],
        #         circuit_with_theta, data_params, self.sampler,
        #     )
        #     for i in range(self.num_classes - 1) for j in range(i + 1, self.num_classes)
        # }

        # self.class_samples_history.append(class_samples)
        # self.class_representatives.append(self._class_representatives(class_samples))
        # self.self_overlaps.append(self_overlaps)
        self.pairwise_overlaps.append(pairwise_overlaps_dict)
        # self.pairwise_distances.append(pairwise_distances_dict)
        self.loss_history.append(loss)

    def _update_best_params(
        self, theta: np.ndarray, val_acc: float, metric_name: str,
        class_representatives: List[np.ndarray],
    ) -> None:
        val_attr = f'best_val_acc_qc_{metric_name}'
        param_attr = f'best_params_qc_{metric_name}'
        rep_attr = f'best_rhos_qc_{metric_name}'

        if val_acc > getattr(self, val_attr):
            setattr(self, val_attr, val_acc)
            setattr(self, param_attr, theta.copy())
            setattr(self, rep_attr, [s.copy() for s in class_representatives])

    def _track_accuracies(
        self, X: np.ndarray, y: np.ndarray, theta: np.ndarray,
        circuit_template: QuantumCircuit, class_samples: List[np.ndarray],
        class_representatives: List[np.ndarray],
    ) -> None:
        from src.classifier import ClassicalClassifier

        kernel_train = compute_kernel_matrix_shots(X, X, theta, circuit_template, self.sampler)
        kernel_val = compute_kernel_matrix_shots(self.X_val, X, theta, circuit_template, self.sampler)
        train_acc, val_acc = ClassicalClassifier.calculate_accuracy_fix(
            kernel_train, kernel_val, y, self.y_val
        )

        self.train_accuracy_history.append(train_acc)
        self.val_accuracy_history.append(val_acc)

        if self.iteration_counter >= self.warmup_iterations and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_params = theta.copy()

    def _loss(self, theta: np.ndarray, store_history: bool = True) -> float:
        is_batch_probe = theta.ndim > 1
        theta = np.ravel(theta)

        theta_params = self.circuit.parameters[self.X_train.shape[1]:]
        theta_dict = {theta_params[k]: theta[k] for k in range(len(theta_params))}
        circuit_with_theta = self.circuit.assign_parameters(theta_dict)
        data_params = self.circuit.parameters[:self.X_train.shape[1]]

        class_samples, fidelities = self._compute_class_data(
            self.X_train, self.y_train, circuit_with_theta, data_params, self.target_states
        )
        class_representatives = self._class_representatives(class_samples)

        loss = self.loss_strategy.compute(
            class_samples, self.num_classes,
            inner_products=fidelities,
            target_states=self.target_states,
            labels=self.unique_labels,
            sampler=self.sampler,
            circuit_with_theta=circuit_with_theta,
            data_params=data_params,
        )
        loss = float(np.real(loss))

        if store_history and not is_batch_probe:
            self._store_history(class_samples, circuit_with_theta, data_params, loss)
            if self.iteration_counter % 100 == 0:
                self._track_accuracies(
                    self.X_train, self.y_train, theta, self.circuit,
                    class_samples, class_representatives,
                )
                self.accuracy_iteration_indices.append(self.iteration_counter)
            self.iteration_counter += 1
            print(f"Iteration {self.iteration_counter}: loss = {loss:.6f}, best_val_acc = {self.best_val_acc:.4f}")

        return loss

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, optimizer,
            init_theta: Optional[np.ndarray] = None,
            circuit: Optional[QuantumCircuit] = None) -> None:
        self.X_train = X_train
        self.y_train = np.ravel(y_train).astype(int)
        self.X_val = X_val
        self.y_val = np.ravel(y_val).astype(int)
        self.optimizer = optimizer

        self.num_qubits = X_train.shape[1]
        self.num_classes = len(np.unique(self.y_train))
        self.unique_labels = np.unique(self.y_train)
        print(f"Number of qubits: {self.num_qubits}, Number of classes: {self.num_classes}")
        print(f"Shot-based simulation: {self.shots} shots")

        if circuit is None:
            self._set_circuit(type=self.ansatz_type)
        else:
            self.circuit = circuit

        if init_theta is None:
            num_params = len(self.circuit.parameters) - self.num_qubits
            self.init_theta = np.random.uniform(-np.pi, np.pi, num_params)
        else:
            self.init_theta = init_theta

        if self.type_loss not in LOSS_STRATEGIES:
            raise ValueError(f"Unknown loss type: {self.type_loss}")
        print(f"Using loss strategy: {LOSS_STRATEGIES[self.type_loss].__name__}")
        self.loss_strategy = LOSS_STRATEGIES[self.type_loss]

        if self.type_loss in {"pre_determine_inner", "log_likelihood_macro"}:
            self.class_patterns = self.choose_class_patterns(X_train, self.y_train)
            print(f"Class patterns: {self.class_patterns}")
        elif self.type_loss == "inner_product":
            if self.num_classes == 2:
                self.class_patterns = {self.unique_labels[0]: '00', self.unique_labels[1]: '01'}
            elif self.num_classes == 3:
                self.class_patterns = {
                    self.unique_labels[0]: '00', self.unique_labels[1]: '01', self.unique_labels[2]: '10'
                }
            elif self.num_classes == 4:
                self.class_patterns = {
                    self.unique_labels[0]: '00', self.unique_labels[1]: '01',
                    self.unique_labels[2]: '10', self.unique_labels[3]: '11',
                }
            else:
                raise ValueError(f"inner_product only supports 2-4 classes, got {self.num_classes}")
            print(f"Class patterns: {self.class_patterns}")
        else:
            self.class_patterns = None

        self.target_states = self.loss_strategy.prepare_targets(
            self.unique_labels, self.class_patterns, self.num_qubits
        )

        self._reset_history()
        result = self.optimizer.minimize(self._loss, x0=self.init_theta)

        self.optimal_params = result.x
        self.optimal_value = result.fun

        print(f"\nOptimization completed!")
        print(f"Optimal value: {self.optimal_value:.6f}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

    def _reset_history(self) -> None:
        self.loss_history = []
        self.class_samples_history = []
        self.class_representatives = []
        self.self_overlaps = []
        self.pairwise_overlaps = []
        self.pairwise_distances = []
        self.train_accuracy_history = []
        self.accuracy_iteration_indices = []
        self.val_accuracy_history = []
        self.train_accuracy_qc_trace_history = []
        self.val_accuracy_qc_trace_history = []
        self.train_accuracy_qc_hs_history = []
        self.val_accuracy_qc_hs_history = []
        self.train_accuracy_qc_overlap_history = []
        self.val_accuracy_qc_overlap_history = []
        self.best_val_acc = 0.0
        self.best_params = None
        self.best_val_acc_qc_trace = 0.0
        self.best_params_qc_trace = None
        self.best_rhos_qc_trace = None
        self.best_val_acc_qc_hs = 0.0
        self.best_params_qc_hs = None
        self.best_rhos_qc_hs = None
        self.best_val_acc_qc_overlap = 0.0
        self.best_params_qc_overlap = None
        self.best_rhos_qc_overlap = None
        self.iteration_counter = 0

    def _set_circuit(self, type: str = 'EfficientSU2') -> None:
        from src.paramertrized_circuit import ParametrizedCircuit
        if type == 'Universal':
            self.circuit = ParametrizedCircuit.Universal_circuit(self.num_qubits, self.depth)
        elif type == 'RealAmplitudes':
            self.circuit = ParametrizedCircuit.RealAmplitudes_circuit(self.num_qubits, self.depth)
        elif type == 'EfficientSU2':
            self.circuit = ParametrizedCircuit.EfficientSU2_circuit(self.num_qubits, self.depth)
        else:
            raise ValueError(f"Unknown circuit type: {type}")

    def get_optimal_params(self) -> np.ndarray:
        return self.optimal_params

    def get_optimal_value(self) -> float:
        return self.optimal_value

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit

    def get_class_samples(self) -> Optional[List[np.ndarray]]:
        if not self.class_samples_history:
            return None
        return self.class_samples_history[-1]

    def get_class_representatives(self) -> Optional[List[np.ndarray]]:
        if not self.class_representatives:
            return None
        return self.class_representatives[-1]

    def get_class_states(self) -> Optional[List[np.ndarray]]:
        """Backward-compatible alias for class representative samples."""
        return self.get_class_representatives()

    def get_rhos(self) -> Optional[List[np.ndarray]]:
        """Backward-compatible alias for class representative samples."""
        return self.get_class_representatives()

    def get_best_rhos_qc_trace(self) -> Optional[List[np.ndarray]]:
        return self.best_rhos_qc_trace

    def get_best_rhos_qc_hs(self) -> Optional[List[np.ndarray]]:
        return self.best_rhos_qc_hs

    def get_best_rhos_qc_overlap(self) -> Optional[List[np.ndarray]]:
        return self.best_rhos_qc_overlap

    def set_optimizer(self, optimizer) -> None:
        self.optimizer = optimizer

