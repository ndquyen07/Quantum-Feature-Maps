#!/bin/bash
#SBATCH --job-name=tqfm_1gauss
#SBATCH --output=/data/%u/logs/tqfm_gauss_%j.out
#SBATCH --error=/data/%u/logs/tqfm_gauss_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16384
#SBATCH --array=1-20

# Activate conda environment

export JOB_SCARCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCARCH_PATH"


# ---- hardware noise model (set any value to 0 to disable that channel) ----
DEPOL_1Q=0.001      # depolarizing error prob, 1-qubit gates (ry, rz)
DEPOL_2Q=0.01       # depolarizing error prob, 2-qubit gates (cx)
READOUT=0.01        # symmetric readout error prob
# thermal relaxation is off; add e.g.  --t1 50e-6 --t2 30e-6  below to enable it

python3.13 parallel_losses_gauss_3class_from_Eff.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz EfficientSU2 --optimizer SPSA --maxiter 1000 --shots 1024 --n_train 200 --n_val 100 --n_test 200 \
    --depol_1q ${DEPOL_1Q} --depol_2q ${DEPOL_2Q} --readout_error ${READOUT} \
    --results_dir /data/ndquyen/qfm/results_losses_gauss_3class_from_Eff_noisy \
    --init_theta_source_dir /data/ndquyen/qfm/results_losses_gauss_3class_Eff


# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz TwoLocal --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --ansatz TwoLocal --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --ansatz TwoLocal --optimizer COBYLA --maxiter 50000

# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz RealAmplitudes --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --ansatz RealAmplitudes --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --ansatz RealAmplitudes --optimizer COBYLA --maxiter 50000

# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz EfficientSU2 --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --ansatz EfficientSU2 --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --ansatz EfficientSU2 --optimizer COBYLA --maxiter 50000

