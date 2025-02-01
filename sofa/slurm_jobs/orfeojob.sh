#!/bin/bash
#SBATCH --job-name=sofa_grasp        # Job name
#SBATCH --output=sofa_grasp_%j.log   # Standard output and error log (%j expands to jobID)
#SBATCH --error=sofa_grasp_%j.err    # Error log
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks per node
#SBATCH --cpus-per-task=32           # Number of CPU cores per task
#SBATCH --mem=100GB                  # Memory limit
#SBATCH --partition=DGX              # Specify the GPU partition/queue
#SBATCH --gres=gpu:1                 # Request 1 GPU

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "Submit dir: $SLURM_SUBMIT_DIR"

# Define paths (adjust these according to your setup)
CONTAINER_PATH="$HOME/xvbf-hub/"
OUTPUT_DIR="$SLURM_SUBMIT_DIR/output"

mkdir -p "$OUTPUT_DIR"

# Run the Singularity container
apptainer run \
    --nv --fakeroot \
    --bind $HOME:/u/dssc/mnunzi00 \
    $CONTAINER_PATH \
    bash -c "\
    source /opt/miniconda/bin/activate && \
    conda activate sofa && \
    cd $CONTAINER_PATH/app/sofa_zoo/ && \
    python3 sofa_zoo/envs/grasp_lift_touch/ppo.py"
