#!/bin/bash
#SBATCH --job-name=model_testing
#SBATCH --output=/home/li.yu/code/scripts/v112_200k_pt1010_1108_test.txt
#SBATCH --partition=gpu_gen4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=60G
#SBATCH --time=8:00:00
# #SBATCH --nodelist=stc01sppamxnl003
# #SBATCH --nodelist=stc01spplmdanl004,stc01sppamxnl002

# activate virtual env
eval "$(/home/li.yu/anaconda3/bin/conda shell.bash hook)"
conda activate torch221

# add working dir
REPO_ROOT=/home/li.yu/code/JupiterCVML
# REPO_ROOT=/mnt/sandbox1/li.yu/code/JupiterCVML
export EUROPA_DIR=${REPO_ROOT}/europa/base/src/europa
export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/europa/base/src/europa

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export COLUMNS=150

# experiment name
EXP='11_3_rev1_train_human_test_dean_multires_ignore_trees_1p25_u_p5_h_p2_v_ft_from_rev1_22kdust_ft_p15dust_h_fresh_dust11'

# check if best.ckpt exists and create symlink if not (trained on kore)
# BEST_CHECKPOINT_PATH=/data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/checkpoints/best.ckpt
# BEST_CHECKPOINT_PATH=/mnt/sandbox1/ben.cline/logs/bc_sandbox_2024_q4/${EXP}/checkpoints/best.ckpt
BEST_CHECKPOINT_PATH=/mnt/sandbox1/ben.cline/logs/bc_sandbox_2024_q4/dust113/${EXP}/checkpoints/best.ckpt

# field dust test
cd ${REPO_ROOT}
# DATASET='halo_failure_case_of_box_in_dust'
# DATASET='halo_fps_in_dust_candidates_stereo'
# DATASET='dust_datasets/Jupiter_bedrock_40013_20240617_dust_sequences'
# DATASET='dust_datasets/Jupiter_bedrock_40013_20240617_212424_box_in_dust_seq'
# DATASET='dust_datasets/Jupiter_bedrock_40013_20240617_214449_lying_manny_in_dust_seq'
DATASET='dust_datasets/halo_buildup_medium_dust_with_human_lo_gilroy_july29'
# DATASET='halo_productivity_fps_in_dust_bedrock411_20240627'
# DATASET='large_datasets/halo_data_pool_pt9_2024_month09_stereo'
srun --kill-on-bad-exit python kore/scripts/predict_seg.py \
    --config_path /home/li.yu/code/scripts/test_dust.yml \
    --data.test_set.dataset_name ${DATASET} \
    --data.test_set.csv master_annotations.csv \
    --data.test_set.dataset_path /data2/jupiter/datasets/${DATASET} \
    --inputs.with_semantic_label false \
    --metrics.run_productivity_metrics true \
    --metrics.gt_stop_classes_to_consider Non-driveable Trees_Weeds Humans Vehicles Unharvested_field \
    --metrics.pred_stop_classes_to_consider Non-driveable Humans Vehicles \
    --states_to_save '' \
    --run_id ${EXP} \
    --ckpt_path ${BEST_CHECKPOINT_PATH} \
    --output_dir /data/jupiter/li.yu/exps/driveable_terrain_model/${EXP}/${DATASET} \


# deactivate virtual env
conda deactivate
conda deactivate

# leave working directory
cd /home/li.yu/code/scripts
