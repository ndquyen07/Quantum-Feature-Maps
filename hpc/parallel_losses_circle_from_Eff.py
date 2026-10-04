"""
Parallel TQFM training on circle dataset.
Loads init_theta from results_losses_circle_Eff; uses shot-based inversion-test simulation.
"""
import argparse

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from parallel_training import (
    TrainingConfig,
    add_common_cli_args,
    config_from_args,
    run_parallel_training,
)


def load_circle_data(config: TrainingConfig):
    n_total = config.n_train + config.n_val + config.n_test
    X_all, y_all = make_circles(
        n_samples=n_total, noise=0.1, factor=0.5, random_state=config.run_id
    )

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=config.n_test, random_state=42, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.n_val, random_state=42, stratify=y_temp
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    parser = argparse.ArgumentParser(
        description="Parallel TQFM training on circle dataset (shot-based, init from Eff results)"
    )
    add_common_cli_args(
        parser,
        n_train=100, n_val=50, n_test=100,
        results_dir="/data/ndquyen/qfm/results_losses_circle_from_Eff",
        init_theta_source_dir="/data/ndquyen/qfm/results_losses_circle_Eff",
    )
    args = parser.parse_args()

    config = config_from_args(
        args,
        default_results_dir="/data/ndquyen/qfm/results_losses_circle_from_Eff",
        default_init_dir="/data/ndquyen/qfm/results_losses_circle_Eff",
    )
    data = load_circle_data(config)
    run_parallel_training(config, *data)


if __name__ == "__main__":
    main()
