#!/bin/bash
#!
#! Dask job script for CSD3
#! Full step scaling test, inputting queue sizes and fixing configuration
#! so submit via "sbatch run_distr_single_csd3.slurm $queue_size
#!
#!#############################################################
#! Name of the job:
#SBATCH -J DFT
#! Which project should be charged:
#SBATCH -A DIRAC-TP001-CPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=9
#! How many tasks will there be in total? (Add a scheduler)
#SBATCH --ntasks=10
#! Memory limit (can experiment with this)
#SBATCH --mem 300000
#! How much wallclock time will be required?
#SBATCH --time=08:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
#SBATCH --no-requeue
#! Using high memory nodes
#SBATCH -p icelake-himem
#! Ask for exclusive access
#SBATCH --exclusive

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-icl 
conda info --envs
source activate python39


if [ -z "$SSMROOT" ]
then
    SSMROOT=${HOME}/Code/ska-sdp-distributed-fourier-transform/
fi
echo "SSMROOT : $SSMROOT"
export PYTHONPATH=$SSMROOT:$PYTHONPATH

cd ${SSMROOT}

echo -e "Running python: `which python`"
echo -e "Running dask-scheduler: `which dask-scheduler`"

echo -e "Changed directory to `pwd`.\n"

JOBID=${SLURM_JOB_ID}
echo ${SLURM_JOB_NODELIST}

#! Create a hostfile:
scontrol show hostname $SLURM_JOB_NODELIST | uniq > hostfile.$JOBID

# Run the scheduler on a large memory node
scheduler=$(head -1 hostfile.$JOBID)

echo "run dask-scheduler"
ssh ${scheduler} dask-scheduler --port=8786 &
sleep 5

# Start the scheduler
while ! nc -z $scheduler 8786; do sleep 0.5; done

for host in `cat hostfile.$JOBID`; do

   echo "Working on $host ...."
   echo "run dask-worker"
   ssh $host dask-worker --host ${host} --nworkers 1 --nthreads 38  \
   --memory-limit 512GiB $scheduler:8786 & 
   sleep 1
done
echo "Scheduler and workers now running"

# Run command
# Input the queue size
config=32k[1]-n16k-1k
CMD="DASK_SCHEDULER=${scheduler}:8786 python scripts/performance_full_steps.py --swift_config $config --queue_size $1 | tee -a distr_fft_single_$1.log"

echo "About to execute $CMD"

eval $CMD
