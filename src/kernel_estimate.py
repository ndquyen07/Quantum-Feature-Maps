import numpy as np
from qiskit.quantum_info import Statevector

from src.shot_simulation import create_sampler, compute_kernel_matrix_shots


class KernelMatrix:
    """Kernel matrix computation (shot-based by default)."""

    @staticmethod
    def _get_statevectors(X, theta_params, circuit):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        data_params = list(circuit.parameters)[:n_features]
        theta_params_list = list(circuit.parameters)[n_features:]

        theta_dict = {}
        for param, val in zip(theta_params_list, theta_params):
            theta_dict[param] = float(val.item() if hasattr(val, 'item') else val)

        statevectors = []
        for i in range(n_samples):
            param_dict = theta_dict.copy()
            for k, param in enumerate(data_params):
                param_dict[param] = X[i, k]
            statevectors.append(Statevector.from_instruction(circuit.assign_parameters(param_dict)).data)

        return np.array(statevectors)

    @staticmethod
    def compute_kernel_matrix_with_inner_products(
        X1, X2, theta_params, circuit, sampler=None, shots=1024,
    ):
        """
        Compute K_ij = |<psi(x_i, theta)|psi(x_j, theta)>|^2.
        Uses shot-based inversion tests when sampler is provided (default).
        """
        if sampler is None:
            sampler = create_sampler(shots)
        return compute_kernel_matrix_shots(X1, X2, theta_params, circuit, sampler)

    @staticmethod
    def compute_kernel_matrix_exact(X1, X2, theta_params, circuit):
        """Exact statevector kernel (legacy helper)."""
        is_symmetric = X1 is X2 or np.array_equal(X1, X2)
        statevectors_1 = KernelMatrix._get_statevectors(X1, theta_params, circuit)
        statevectors_2 = statevectors_1 if is_symmetric else KernelMatrix._get_statevectors(X2, theta_params, circuit)

        inner_products = statevectors_1 @ statevectors_2.conj().T
        kernel_matrix = np.real(inner_products * inner_products.conj())

        if is_symmetric:
            kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2.0
            np.fill_diagonal(kernel_matrix, 1.0)

        return kernel_matrix

    @staticmethod
    def compute_kernel_matrix_with_inner_products_legacy(X1, X2, theta_params, circuit):
        return KernelMatrix.compute_kernel_matrix_exact(X1, X2, theta_params, circuit)


if __name__ == "__main__":
    pass
