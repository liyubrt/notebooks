#!/usr/bin/env bash
#SBATCH --job-name=pack_perception
#SBATCH --output=/home/li.yu/code/scripts/pp_halo_human_w_corn_stubble_0812_oct.txt
#SBATCH --error=/home/li.yu/code/scripts/pp_halo_human_w_corn_stubble_0812_oct.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=48000
#SBATCH --time=20:00:00
#SBATCH --partition=gpu_gen3
#SBATCH --requeue

# If you wish, change the number of tasks per node, but using 8 GPUs is
# probably the most optimal.

# TODO: Change the data folder and dataset that you are trying to run PP on
# Name or ID of the dataset that we are saving.
DATASET_NAME_OR_ID=halo_human_w_corn_stubble_0812_oct
# DATA_FOLDER=/data/jupiter/datasets
# DATA_FOLDER=/data2/jupiter/datasets
# DATA_FOLDER=/data3/jupiter/datasets
# DATA_FOLDER=/data/jupiter/datasets/safety_datasets/hazards/on_path_forward
# DATA_FOLDER=/data/jupiter/datasets/dust_datasets
# DATA_FOLDER=/data2/jupiter/datasets/dust_datasets
DATA_FOLDER=/data3/jupiter/datasets/model_positive
# DATA_FOLDER=/data/jupiter/li.yu/data
# Name of the folder that we store the dataset in. 'auto' to name it based on the dataset
# name of the specified id.
FOLDER_NAME='auto'

# TODO: Rename this variable or symlink this directory to a path to your JupiterCVML folder
JUPITERCVML_DIR=/home/li.yu/code/JupiterCVML
# JUPITERCVML_DIR=/mnt/sandbox1/li.yu/code/JupiterCVML
# export EUROPA_DIR=${JUPITERCVML_DIR}/europa/base/src/europa
# export PYTHONPATH=${JUPITERCVML_DIR}:${JUPITERCVML_DIR}/europa/base/src/europa
# Possible values for RESUME_MODE:
#   fresh: delete existing partitioning and restart all of PP
#   redo-ocal: delete ocal results and restart all of PP
#   redo-depth: keep ocal results, delete PP artifacts, and restart depth inference
#   existing: continue running PP with the current partitioning and partial results
RESUME_MODE=existing
# "yes" to Download processed folder from AWS if available
AWS_DOWNLOAD=yes
# "yes" to upload data to AWS when done.
AWS_UPLOAD=no
# "yes" to Remove files after PP is done
CLEAN_WHEN_DONE=yes
# TODO: Choose the appropriate command and modify to your tastes/needs
#  - divide the batch size (and maybe workers) to prevent OOMs. This includes dividing
#    the batch size by 4 if you are using a full resolution model.

PP_PARAMETERS="""
    --batch-size 12 --multiprocess-workers 24 --pandarallel-workers 24
    --model-path /data2/jupiter/models/20240816_depth_w_lr.ptp
"""
    # --model-path /data2/jupiter/models/20240423_halo_depth_ext_2xDL_SW3_11_lite_max_fov_ep38.ptp
    # --model-path /data2/jupiter/models/20240622_depth_model_master_1cyc.ptp
    # --model-path /data2/jupiter/models/20240625_depth_w_lr_probs.ptp
    # --run-oc
    # --no-drop-bad-data
    # --image-only
    # --backend adk
    # --debayer mhc_stcompand

PP_CONTAINER=/data2/jupiter/singularity/jupiter-pack-perception/main.sif

#############################
# End of configuration.     #
# Main script starts below. #
#############################

# Everything below this point must pass (except for deletion of old files) for
# PP to run smoothly.
set -e

module load anaconda aws-cli apptainer

# Change your conda environment if you want to use a different one, but there
# generally isn't any need to. This is only used for downloading and paritioning
# a dataset.
conda activate /mnt/sandbox1/anirudh.vegesana/conda_envs/pp/


# Set important environmental variables
export BRT_ENV=prod
export AWS_PROFILE=jupiter_prod_engineer-425642425116
export AWS_DEFAULT_REGION=us-west-2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Get useful locations inside of JupiterCVML if the variable is set
if [ -n "$JUPITERCVML_DIR" ]
then
    EUROPA_DIR=$JUPITERCVML_DIR/europa/base/src/europa/
    export PYTHONPATH=$EUROPA_DIR:$PYTHONPATH
    FILES_DIR=$JUPITERCVML_DIR/europa/base/files/
    EXECUTABLES_DIR=$JUPITERCVML_DIR/europa/base/executables
    _ADDITIONAL_BINDS=$_ADDITIONAL_BINDS,$JUPITERCVML_DIR:/src/brt/autonomy/jupiter/cvml,$FILES_DIR:/files,$EXECUTABLES_DIR:/executables
    DOWNLOAD_DATASET_CMD="python3 $EUROPA_DIR/dl/dataset/download.py"
    PARITION_DATASET_CMD="python3 $EUROPA_DIR/dl/dataset/pack_perception/partition_dataset.py"
else
    DOWNLOAD_DATASET_CMD=/data2/jupiter/download_ds
    PARITION_DATASET_CMD=/data2/jupiter/partition_dataset
fi

APPTAINER_COMMAND="apptainer run --bind /data,/data2,/mnt/sandbox1$_ADDITIONAL_BINDS --nv $PP_CONTAINER"
NUM_PARTITIONS=$SLURM_NTASKS


# Get the hash with the existing apptainer container + parameters
$APPTAINER_COMMAND python3 -m dl.dataset.pack_perception.pack_perception_parameters \
	 --dataset-id $DATASET_NAME_OR_ID $PP_PARAMETERS

HASH=$(cat ~/.local/pack_perception/$DATASET_NAME_OR_ID/pp_hash.txt)
DATASET_ID=$(cat ~/.local/pack_perception/$DATASET_NAME_OR_ID/dset_id.txt)
DATASET_NAME=$(cat ~/.local/pack_perception/$DATASET_NAME_OR_ID/dset_name.txt)

if [[ "$FOLDER_NAME" == "auto" ]]; then
    FOLDER_NAME=$DATASET_NAME
    echo "Setting folder name to the name of the dataset; $FOLDER_NAME"
fi

DATA=$DATA_FOLDER/$FOLDER_NAME

echo Data folder: $DATA
mkdir -p $DATA

# Try to download the artifacts from AWS
AWS_OUTPUT_PATH="s3://blueriver-jupiter-data/pack_perception/ml/${HASH}"
AWS_S3_MASTER_ANNO_URI="${AWS_OUTPUT_PATH}/${DATASET_ID}_master_annotations.csv"
if [[ -e "$DATA/processed" ]]; then
    echo "Processed folder already exists. Rerunning to generate the results."
fi
if [[ "$AWS_DOWNLOAD" == "yes" ]]; then
    if [[ $(aws s3 ls $AWS_S3_MASTER_ANNO_URI) ]]; then
        echo "Skipping PP and getting processed folder from the cache"
        python $JUPITERCVML_DIR/europa/base/src/europa/dl/dataset/fetch_pp_artifacts.py --output-path $DATA --master-csv-s3-uri $AWS_S3_MASTER_ANNO_URI
        ln -s ${DATASET_ID}_master_annotations.csv $DATA/master_annotations.csv
        exit 0
    else
        echo "Starting PP as images are not cached in  $AWS_S3_MASTER_ANNO_URI"
    fi
else
    echo "Not attempting to download PP artifacts from AWS cache"
    echo "If you wish to download from cache set AWS_DOWNLOAD to yes and/or delete existing artifacts in $DATA"
fi


# Download the dataset.
if [ ! -e "$DATA/annotations.csv" ] || [ ! -d "$DATA/images" ]
then
    $DOWNLOAD_DATASET_CMD $DATASET_ID -d $DATA_FOLDER --folder_name $FOLDER_NAME
fi

# Download supporting ocal data.
if [[ $PP_PARAMETERS == *"--run-oc"* && ! -d "$DATA/online_calibration_data/images" ]]
then
    mkdir -p $DATA/online_calibration_data/images

    if [ -n "$JUPITERCVML_DIR" ]
    then
        python3 $EUROPA_DIR/dl/dataset/pack_perception/download_ocal_data.py $DATA
    fi
fi

# Delete PP outputs if requested.
set +e
if [ -n "$SLURM_RESTART_COUNT" ] && [ $SLURM_RESTART_COUNT -ne 0 ]
then
    echo SLURM reset. Ignoring resume mode and continuing run.
elif [ "$RESUME_MODE" = "fresh" ]
then
    echo Deleting existing partitions.
    rm -r \
        $DATA/partitions \
        $DATA/processed \
        $DATA/*master_annotations.csv
elif [ "$RESUME_MODE" = "redo-ocal" ]
then
    echo Deleting ocal results.
    rm -r \
        $DATA/partitions/*/*master_annotations.csv \
        $DATA/partitions/*/annotations_ocal.csv \
        $DATA/partitions/*/online_calibration_data/ocal_df.csv \
        $DATA/processed \
        $DATA/*master_annotations.csv
elif [ "$RESUME_MODE" = "redo-depth" ]
then
    echo Deleting depth inference results.
    rm -r \
        $DATA/partitions/*/*master_annotations.csv \
        $DATA/processed \
        $DATA/*master_annotations.csv
elif [ "$RESUME_MODE" = "existing" ]
then
    if [ -d "$DATA/partitions" ]
    then
        echo Resuming existing PP run.
    else
        echo Starting new PP run.
    fi
else
    echo Unknown resume mode $RESUME_MODE. Using existing.
fi
set -e

# Create processed folder if it doesn't exist.
mkdir -p $DATA/processed

# Copy the calibration data into the right place if it isn't there already.
if [ -n "$JUPITERCVML_DIR" -a ! -d "$DATA/processed/calibration" ]
then
    cp -r $FILES_DIR/calibration $DATA/processed/
fi

# Parition the dataset.
if [ ! -d "$DATA/partitions" ]
then
    $PARITION_DATASET_CMD \
        --dataset-folder $DATA \
        --partitions-folder $DATA/partitions \
        --num-partitions $NUM_PARTITIONS \
        partition \
        --use-relative-symlinks false
else
    echo Using existing partitioning.

    $PARITION_DATASET_CMD \
        --dataset-folder $DATA \
        --partitions-folder $DATA/partitions \
        --num-partitions $NUM_PARTITIONS \
        verify
fi

# Actually run the main pack perception script.
# TODO: You may wish to add `--kill-on-bad-exit` to the srun command if you
# are debugging. This flag will terminate the job early if any partition
# fails.
srun \
    --output=/home/li.yu/code/scripts/pp_${DATASET_NAME_OR_ID}.txt \
    --error=/home/li.yu/code/scripts/pp_${DATASET_NAME_OR_ID}.txt \
    --unbuffered \
$APPTAINER_COMMAND python3 -m dl.dataset.pack_perception.ml_pack_perception \
    --data-dir $DATA/partitions/\$SLURM_PROCID --csv-path \\\$DATA_DIR/annotations.csv \
    --ignore-slurm-variables --gpu 0 --dataset-id $DATASET_ID \
    $PP_PARAMETERS

# Combine the partitions back into the master_annotations.csv
$PARITION_DATASET_CMD \
    --dataset-folder $DATA \
    --partitions-folder $DATA/partitions \
    --num-partitions $NUM_PARTITIONS \
    combine

# Give the "jupiter" group read/write access to the dataset
chmod -R g+w "$DATA"

# Upload combined data to AWS
if [[ -e "$DATA/master_annotations.csv" ]]; then
    echo "GREAT SUCCESS, created a master annotations.csv!"
    printf "id:\n${DATASET_ID}\nname:\n${DATASET_NAME}\naws_path:\n${AWS_S3_MASTER_ANNO_URI}" >> $DATA/dataset_info.txt
    echo "Saved $DATASET_NAME images to ${AWS_S3_MASTER_ANNO_URI}"
    if [[ "$CLEAN_WHEN_DONE" == "yes" ]]; then
        echo "Deleting other folders"
        rm -r $DATA/partitions
        rm -r $DATA/online_calibration_data
        rm -r $DATA/images
        rm -r $DATA/processed/calibration
        echo "Cleanup complete!"
    fi
    # Note that using the --upload-to-s3 flag when we call pack perception does not work because it will only upload 1 partition
    if [[ "$AWS_UPLOAD" == "yes" ]]; then
        echo "Saved $DATASET_NAME images to ${AWS_S3_MASTER_ANNO_URI}"
        echo "Uploading to AWS..."
        aws s3 cp --recursive $DATA/processed $AWS_OUTPUT_PATH/processed
        aws s3 cp $DATA/master_annotations.csv $AWS_S3_MASTER_ANNO_URI
        echo "$AWS_S3_MASTER_ANNO_URI" >> "$DATA/aws_path.txt"
        echo "Success: Saved images to $AWS_S3_MASTER_ANNO_URI"
    fi
else
    echo "SHOCKING FAILURE: no master csv was created"
fi
