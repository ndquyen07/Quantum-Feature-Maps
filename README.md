# Benchmarking Loss Functions for Trainable Quantum Feature Maps

Code for the paper [Benchmarking loss functions for trainable quantum feature maps](https://arxiv.org/abs/2607.12487) (arXiv:2607.12487).

A trainable quantum feature map (TQFM) is a parametrized quantum circuit that encodes classical data into quantum states. We train the circuit so that each class maps to well-separated states, then use the resulting quantum kernel $K(x, x') = |\langle\psi(x)|\psi(x')\rangle|^2$ in a classical SVM.

We compare three loss functions for training the circuit:

| Loss | Name in code | Cost per step |
|---|---|---|
| Distance Loss (DL) | `hilbert_schmidt` | $O(M^2)$ |
| Measurement Loss (ML) | `pre_determine_inner` | $O(M)$ |
| Log-Likelihood Loss (LLL) | `log_likelihood_macro` | $O(M)$ |

Here $M$ is the number of training samples. The experiments use three scikit-learn datasets (circles, moons and a 3-class Gaussian) on 2 qubits with the EfficientSU2 ansatz, at depths 1–6, with and without simulated noise.

The code simulates circuits with measurement shots (Qiskit Aer). The paper's noiseless results were computed with exact statevector simulation.

## Installation

```bash
git clone https://github.com/ndquyen07/Quantum-Feature-Map.git
cd Quantum-Feature-Map
pip install -r requirements.txt
```

Developed with Python 3.13 and Qiskit 1.4.

## Project structure

```
src/          feature map, loss functions, quantum kernel, SVM
hpc/          training scripts and SLURM job files
experiment/   analysis notebooks and plotting script
```

## Usage

**1. Create the initial parameters.** Every run starts from a saved file `shared_init_theta_depth{D}_ansatz{A}_run{R}.npy`, shared by all three losses so they are compared fairly. For EfficientSU2 on 2 qubits it holds $8(D+1)$ values, which is 24 for depth 2. From the repository root:

```bash
python -c "import os, numpy as np; os.makedirs('init_theta', exist_ok=True); np.save('init_theta/shared_init_theta_depth2_ansatzEfficientSU2_run1.npy', np.random.uniform(-np.pi, np.pi, 24))"
```

**2. Train.** This trains all three losses in parallel on the moons dataset:

```bash
cd hpc
python parallel_losses_moon_from_Eff.py --run_id 1 --depth 2 --ansatz EfficientSU2 \
    --maxiter 2000 --shots 1024 --n_train 100 --n_val 50 --n_test 100 \
    --results_dir ../results_moon --init_theta_source_dir ../init_theta
```

- **Other datasets:** use `parallel_losses_circle_from_Eff.py` or `parallel_losses_gauss_3class_from_Eff.py`.
- **Noise:** add `--depol_1q 0.001 --depol_2q 0.01 --readout_error 0.01`.
- **SLURM cluster:** run `sbatch run_tqfm_moon.sh` from `hpc/`, after editing the paths in the script.

Each run saves a trained model (`.pkl`) for each loss and a JSON file with the train and test accuracies to `--results_dir`.

## Citation

```bibtex
@misc{quyen2026benchmarking,
  title         = {Benchmarking loss functions for trainable quantum feature maps},
  author        = {Nguyen Dinh Quyen and Vu Tuan Hai and Quoc Chuong Nguyen and Le Bin Ho and Lan Nguyen Tran},
  year          = {2026},
  eprint        = {2607.12487},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2607.12487},
  url           = {https://arxiv.org/abs/2607.12487}
}
```
