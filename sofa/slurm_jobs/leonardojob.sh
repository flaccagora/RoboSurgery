#!/bin/bash
#SBATCH --job-name=sofa_grasp        # Job name
#SBATCH --output=sofa_grasp_%j.log   # Standard output and error log (%j expands to jobID)
#SBATCH --error=sofa_grasp_%j.err    # Error log
#SBATCH --time=00:30:00              # Time limit hrs:min:sec
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks per node
#SBATCH --cpus-per-task=32           # Number of CPU cores per task
#SBATCH --mem=100GB                  # Memory limit
#SBATCH -p boost_usr_prod              # Specify the GPU partition/queue
#SBATCH -A ICT24_DSSC_GPU            # Account
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --qos=boost_qos_dbg

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "Submit dir: $SLURM_SUBMIT_DIR"

# Define paths (adjust these according to your setup)
CONTAINER_PATH="$SCRATCH/xvbf-hub"

# Run the Singularity container
singularity run \
    --nv \
    --bind /leonardo_scratch/large/userexternal/mnunzian:$SCRATCH \
    $CONTAINER_PATH \
    bash -c "\
    source /opt/miniconda/bin/activate && \
    conda activate sofa && \
    cd $CONTAINER_PATH/app/sofa_zoo/ && \
    python3 sofa_zoo/envs/grasp_lift_touch/ppo.py"

# /leonardo_scratch/large/userexternal/mnunzian
# /leonardo/home/userexternal/mnunzian