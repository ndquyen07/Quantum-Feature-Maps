"""
Parallel TQFM training on moon dataset.
Loads init_theta from results_losses_moon_Eff; uses shot-based inversion-test simulation.
"""
import argparse

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from parallel_training import (
    TrainingConfig,
    add_common_cli_args,
    config_from_args,
    run_parallel_training,
)


def load_moon_data(config: TrainingConfig):
    n_total = config.n_train + config.n_val + config.n_test
    X_all, y_all = make_moons(n_samples=n_total, noise=0.1, random_state=config.run_id)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=config.n_test, random_state=42, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.n_val, random_state=42, stratify=y_temp
    )

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    parser = argparse.ArgumentParser(
        description="Parallel TQFM training on moon dataset (shot-based, init from Eff results)"
    )
    add_common_cli_args(
        parser,
        results_dir="/data/ndquyen/qfm/results_losses_moon_from_Eff",
        init_theta_source_dir="/data/ndquyen/qfm/results_losses_moon_Eff",
    )
    args = parser.parse_args()

    config = config_from_args(
        args,
        default_results_dir="/data/ndquyen/qfm/results_losses_moon_from_Eff",
        default_init_dir="/data/ndquyen/qfm/results_losses_moon_Eff",
    )
    data = load_moon_data(config)
    run_parallel_training(config, *data)


if __name__ == "__main__":
    main()
