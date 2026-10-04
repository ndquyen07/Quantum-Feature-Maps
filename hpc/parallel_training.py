"""
Shared utilities for parallel TQFM training on HPC.
Uses shot-based simulation with inversion-test overlaps (src.feature_map).
"""
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit_algorithms.optimizers import ADAM, COBYLA, SPSA

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is importable when scripts run from hpc/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.shot_simulation import create_sampler  # noqa: E402

LOSS_FUNCTIONS = ["pre_determine_inner", "hilbert_schmidt", "log_likelihood_macro"]
VALID_ANSATZ = ["Universal", "RealAmplitudes", "EfficientSU2"]
VALID_OPTIMIZERS = ["COBYLA", "SPSA", "ADAM"]


@dataclass
class TrainingConfig:
    run_id: int = 0
    depth: int = 1
    type_ansatz: str = "RealAmplitudes"
    optimizer_name: str = "COBYLA"
    maxiter: int = 1000
    warmup_iterations: int = 200
    accuracy_check_interval: int = 50
    shots: int = 1024
    n_train: int = 200
    n_val: int = 100
    n_test: int = 200
    # noise parameters (all zero/None = noiseless)
    depol_1q: float = 0.0
    depol_2q: float = 0.0
    t1: Optional[float] = None
    t2: Optional[float] = None
    readout_error: float = 0.0
    results_dir: str = "results"
    init_theta_source_dir: str = ""

    def validate(self) -> None:
        checks = [
            (self.depth >= 1, "depth must be >= 1"),
            (self.maxiter >= 1, "maxiter must be >= 1"),
            (self.warmup_iterations >= 0, "warmup_iterations must be >= 0"),
            (self.warmup_iterations < self.maxiter, "warmup_iterations must be < maxiter"),
            (self.accuracy_check_interval >= 1, "accuracy_check_interval must be >= 1"),
            (self.shots >= 1, "shots must be >= 1"),
            (self.type_ansatz in VALID_ANSATZ, f"ansatz must be one of {VALID_ANSATZ}"),
            (self.optimizer_name in VALID_OPTIMIZERS, f"optimizer must be one of {VALID_OPTIMIZERS}"),
            (self.n_train >= 10, "n_train must be >= 10"),
            (self.n_val >= 10, "n_val must be >= 10"),
            (self.n_test >= 10, "n_test must be >= 10"),
        ]
        for condition, message in checks:
            if not condition:
                raise ValueError(message)


def noise_model_from_config(config: "TrainingConfig"):
    """Build a NoiseModel from TrainingConfig noise fields, or return None if noiseless."""
    from src.shot_simulation import create_noise_model
    return create_noise_model(
        depol_1q=config.depol_1q,
        depol_2q=config.depol_2q,
        t1=config.t1,
        t2=config.t2,
        readout_error=config.readout_error,
    )


def create_optimizer(name: str, maxiter: int):
    optimizers = {
        # tol maps to COBYLA's rhoend; set tiny so it doesn't self-terminate early on
        # the noisy objective and instead runs the full maxiter budget.
        "COBYLA": lambda: COBYLA(maxiter=maxiter, tol=1e-12),
        "SPSA": lambda: SPSA(maxiter=maxiter, learning_rate=0.05, perturbation=0.01),
        "ADAM": lambda: ADAM(maxiter=maxiter, lr=0.05),
    }
    return optimizers[name]()


def train_single_loss(args: Dict[str, Any]) -> Dict[str, Any]:
    from src.feature_map import TrainableQuantumFeatureMap

    loss_type = args["loss_type"]
    process_name = mp.current_process().name
    print(f"[{process_name}] Starting {loss_type} training ({args['shots']} shots)")

    try:
        sampler = create_sampler(args["shots"], noise_model=args.get("noise_model"))
        tqfm = TrainableQuantumFeatureMap(
            depth=args["depth"],
            type_ansatz=args["type_ansatz"],
            type_loss=loss_type,
            warmup_iterations=args["warmup_iterations"],
            accuracy_check_interval=args["accuracy_check_interval"],
            shots=args["shots"],
            sampler=sampler,
        )

        start_time = time.time()
        optimizer = create_optimizer(args["optimizer_name"], args["maxiter"])
        tqfm.fit(
            args["X_train"], args["y_train"],
            args["X_val"], args["y_val"],
            optimizer=optimizer,
            init_theta=args["init_theta"].copy(),
        )
        training_time = time.time() - start_time
        print(f"[{process_name}] {loss_type} completed in {training_time:.2f}s")

        model_filename = (
            f"{args['results_dir']}/tqfm_{loss_type}_"
            f"depth{args['depth']}_ansatz{args['type_ansatz']}_"
            f"iter{args['maxiter']}_run{args['run_id']}.pkl"
        )
        with open(model_filename, "wb") as f:
            pickle.dump(tqfm, f)
        print(f"[{process_name}] Saved to {model_filename}")

        return {
            "loss_type": loss_type,
            "tqfm": tqfm,
            "training_time": training_time,
            "optimal_value": tqfm.optimal_value,
            "model_filename": model_filename,
            "success": True,
        }
    except Exception as e:
        print(f"[{process_name}] Error in {loss_type}: {e}")
        import traceback
        traceback.print_exc()
        return {"loss_type": loss_type, "error": str(e), "success": False}


def load_init_theta(config: TrainingConfig) -> np.ndarray:
    init_theta_file = (
        f"{config.init_theta_source_dir}/shared_init_theta_"
        f"depth{config.depth}_ansatz{config.type_ansatz}_run{config.run_id}.npy"
    )
    if not os.path.exists(init_theta_file):
        raise FileNotFoundError(
            f"shared_init_theta file not found: {init_theta_file}"
        )
    init_theta = np.load(init_theta_file)
    print(f"Loaded shared_init_theta from: {init_theta_file}  (shape={init_theta.shape})")
    return init_theta


def evaluate_test_accuracies(
    trained_models: Dict[str, Any],
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    from src.kernel_estimate import KernelMatrix
    from src.classifier import ClassicalClassifier

    test_accuracies = {}
    train_accuracies = {}

    for loss_type, tqfm in trained_models.items():
        # Prefer best_params (best post-warmup validation accuracy). If the optimizer
        # converged before any post-warmup accuracy check landed, best_params is None —
        # fall back to optimal_params so the model is still evaluated, not silently dropped.
        eval_params = tqfm.best_params if tqfm.best_params is not None else tqfm.optimal_params
        if eval_params is None:
            print(f"✗ {loss_type:25} | Skipped (no params available)")
            test_accuracies[loss_type] = None
            train_accuracies[loss_type] = None
            continue
        params_source = "best_params" if tqfm.best_params is not None else "optimal_params (fallback)"

        train_acc = tqfm.train_accuracy_history[-1] if tqfm.train_accuracy_history else None
        sampler = tqfm.sampler

        kernel_train_val = KernelMatrix.compute_kernel_matrix_with_inner_products(
            X_train_val, X_train_val, eval_params, tqfm.circuit,
            sampler=sampler, shots=tqfm.shots,
        )
        kernel_test = KernelMatrix.compute_kernel_matrix_with_inner_products(
            X_test, X_train_val, eval_params, tqfm.circuit,
            sampler=sampler, shots=tqfm.shots,
        )

        _, test_acc, _, best_c = ClassicalClassifier.evaluate_model(
            kernel_train_val, kernel_test, y_train_val, y_test
        )

        test_accuracies[loss_type] = test_acc
        train_accuracies[loss_type] = train_acc
        train_acc_str = f"{train_acc * 100:.2f}%" if train_acc is not None else "N/A"
        print(
            f"✓ {loss_type:25} | Train Acc: {train_acc_str} | "
            f"Test Acc: {test_acc * 100:.2f}% | Best C: {best_c} | using {params_source}"
        )

    return test_accuracies, train_accuracies


def run_parallel_training(
    config: TrainingConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    loss_functions: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Optional[float]]]:
    config.validate()
    loss_functions = loss_functions or LOSS_FUNCTIONS
    os.makedirs(config.results_dir, exist_ok=True)

    shared_init_theta = load_init_theta(config)

    print(f"\n{'=' * 80}")
    print("PARALLEL TRAINING CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"Run ID:              {config.run_id}")
    print(f"Depth:               {config.depth}")
    print(f"Ansatz:              {config.type_ansatz}")
    print(f"Optimizer:           {config.optimizer_name}")
    print(f"Max iterations:      {config.maxiter}")
    print(f"Warmup iterations:   {config.warmup_iterations}")
    print(f"Shots:               {config.shots}")
    noise_on = (
        config.depol_1q > 0 or config.depol_2q > 0
        or (config.t1 is not None and config.t1 > 0)
        or config.readout_error > 0
    )
    print(f"Noise model:         {'ON' if noise_on else 'OFF (noiseless)'}")
    if noise_on:
        print(f"  depol_1q / 2q:     {config.depol_1q} / {config.depol_2q}")
        print(f"  t1 / t2:           {config.t1} / {config.t2}")
        print(f"  readout_error:     {config.readout_error}")
    print(f"Training samples:    {len(X_train)}")
    print(f"Validation samples:  {len(X_val)}")
    print(f"Test samples:        {len(X_test)}")
    print(f"Loss functions:      {loss_functions}")
    print(f"{'=' * 80}\n")

    base_args = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "init_theta": shared_init_theta,
        "depth": config.depth,
        "type_ansatz": config.type_ansatz,
        "optimizer_name": config.optimizer_name,
        "maxiter": config.maxiter,
        "warmup_iterations": config.warmup_iterations,
        "accuracy_check_interval": config.accuracy_check_interval,
        "shots": config.shots,
        "noise_model": noise_model_from_config(config),
        "results_dir": config.results_dir,
        "run_id": config.run_id,
    }
    training_args = [{**base_args, "loss_type": loss} for loss in loss_functions]

    trained_models = {}
    training_times = {}
    start_time = time.time()
    num_workers = min(len(loss_functions), mp.cpu_count())
    print(f"Starting parallel training with {num_workers} workers...\n")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_loss = {
            executor.submit(train_single_loss, args): args["loss_type"]
            for args in training_args
        }
        for future in as_completed(future_to_loss):
            loss_type = future_to_loss[future]
            try:
                result = future.result()
                if result["success"]:
                    trained_models[loss_type] = result["tqfm"]
                    training_times[loss_type] = result["training_time"]
                    print(
                        f"✓ {loss_type:25} | optimal={result['optimal_value']:.6f} | "
                        f"time={result['training_time']:.2f}s"
                    )
                else:
                    print(f"✗ {loss_type:25} | FAILED: {result.get('error', 'Unknown')}")
            except Exception as exc:
                print(f"✗ {loss_type:25} | EXCEPTION: {exc}")

    total_time = time.time() - start_time

    init_filename = (
        f"{config.results_dir}/shared_init_theta_"
        f"depth{config.depth}_ansatz{config.type_ansatz}_run{config.run_id}.npy"
    )
    np.save(init_filename, shared_init_theta)

    print(f"\n{'=' * 80}")
    print("EVALUATING TEST ACCURACY")
    print(f"{'=' * 80}")

    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([np.ravel(y_train), np.ravel(y_val)], axis=0).astype(int)
    test_accuracies, train_accuracies = evaluate_test_accuracies(
        trained_models, X_train_val, y_train_val, X_test, y_test
    )

    timing_data = {
        "total_parallel_time": total_time,
        "individual_training_times": training_times,
        "test_accuracies": test_accuracies,
        "train_accuracies": train_accuracies,
        "configuration": {
            "run_id": config.run_id,
            "depth": config.depth,
            "type_ansatz": config.type_ansatz,
            "optimizer_name": config.optimizer_name,
            "maxiter": config.maxiter,
            "warmup_iterations": config.warmup_iterations,
            "shots": config.shots,
            "n_train": config.n_train,
            "n_val": config.n_val,
            "n_test": config.n_test,
            "init_theta_source": (
                f"{config.init_theta_source_dir}/shared_init_theta_"
                f"depth{config.depth}_ansatz{config.type_ansatz}_run{config.run_id}.npy"
            ),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    timing_filename = (
        f"{config.results_dir}/training_times_"
        f"depth{config.depth}_ansatz{config.type_ansatz}_run{config.run_id}.json"
    )
    with open(timing_filename, "w") as f:
        json.dump(timing_data, f, indent=2)
    print(f"Training times saved to: {timing_filename}")

    print(f"\n{'=' * 80}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total time:          {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"Successful:          {len(trained_models)}/{len(loss_functions)}")
    print(f"Results saved in:    {config.results_dir}/")
    print("\nTest Accuracy Summary:")
    for loss_type in loss_functions:
        acc = test_accuracies.get(loss_type)
        print(f"  {loss_type:25} | {acc * 100:.2f}%" if acc is not None else f"  {loss_type:25} | N/A")
    print(f"{'=' * 80}\n")

    return trained_models, shared_init_theta, test_accuracies


def add_common_cli_args(
    parser,
    n_train: int = 200,
    n_val: int = 100,
    n_test: int = 200,
    results_dir: Optional[str] = None,
    init_theta_source_dir: Optional[str] = None,
) -> None:
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--ansatz", type=str, default="RealAmplitudes")
    parser.add_argument("--optimizer", type=str, default="COBYLA")
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--accuracy_check_interval", type=int, default=50,
                        help="Optimizer evals between validation-accuracy checks "
                             "(drives best_params selection; larger=faster, coarser)")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--depol_1q", type=float, default=0.0,
                        help="Depolarizing error prob for single-qubit gates (0=noiseless)")
    parser.add_argument("--depol_2q", type=float, default=0.0,
                        help="Depolarizing error prob for two-qubit gates (0=noiseless)")
    parser.add_argument("--t1", type=float, default=None,
                        help="T1 relaxation time in seconds, e.g. 50e-6")
    parser.add_argument("--t2", type=float, default=None,
                        help="T2 dephasing time in seconds, e.g. 30e-6")
    parser.add_argument("--readout_error", type=float, default=0.0,
                        help="Symmetric readout error probability (0=noiseless)")
    parser.add_argument("--n_train", type=int, default=n_train)
    parser.add_argument("--n_val", type=int, default=n_val)
    parser.add_argument("--n_test", type=int, default=n_test)
    parser.add_argument("--results_dir", type=str, required=False, default=results_dir)
    parser.add_argument("--init_theta_source_dir", type=str, required=False,
                        default=init_theta_source_dir)


def config_from_args(args, default_results_dir: str, default_init_dir: str) -> TrainingConfig:
    return TrainingConfig(
        run_id=args.run_id,
        depth=args.depth,
        type_ansatz=args.ansatz,
        optimizer_name=args.optimizer,
        maxiter=args.maxiter,
        warmup_iterations=args.warmup,
        accuracy_check_interval=args.accuracy_check_interval,
        shots=args.shots,
        depol_1q=args.depol_1q,
        depol_2q=args.depol_2q,
        t1=args.t1,
        t2=args.t2,
        readout_error=args.readout_error,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        results_dir=args.results_dir or default_results_dir,
        init_theta_source_dir=args.init_theta_source_dir or default_init_dir,
    )
