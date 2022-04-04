d#!/bin/bash

#! Job name
#SBATCH -J dft
#! Which project should be charged:
#SBATCH -A SKA-SDP
#! How many whole nodes should be allocated?
#SBATCH --nodes=8
#! How many (MPI) tasks will there be in total? (<= nodes*16)
#SBATCH --ntasks=64
#! Memory limit
##SBATCH --mem 128000
#! How much wallclock time will be required?
#SBATCH --time=48:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#! Do not change:
#SBATCH -p full

module purge   

DFT_PATH=${HOME}/Code/ska-sdp-distributed-fourier-transform
echo "source code path : $DFT_PATH"
export PYTHONPATH=$DFT_PATH:$PYTHONPATH

#! Set up python
echo -e "[main]Running python: `which python`"
echo -e "[main]which main in node: `hostname`"
echo -e "[main]Running dask-scheduler: `which dask-scheduler`"

cd $SLURM_SUBMIT_DIR
echo -e "[main]Changed directory to `pwd`.\n"

JOBID=${SLURM_JOB_ID}
echo ${SLURM_JOB_NODELIST}

#! Create a hostfile:
scontrol show hostnames $SLURM_JOB_NODELIST | uniq > hostfile.$JOBID

# Run the scheduler on a large memory node
scheduler=$(head -1 hostfile.$JOBID)

echo "run dask-scheduler"
ssh ${scheduler} dask-scheduler --port=8786 &
sleep 5

for host in `cat hostfile.$JOBID`; do
    if [ $host != $scheduler ]; then
       echo "Working on $host ...."
       echo "run dask-worker"
       ssh $host dask-worker --host ${host} --nprocs 8 --nthreads 1  \
       --memory-limit 0 $scheduler:8786 &
       sleep 1
    fi
done
echo "Scheduler and workers now running"

CMD="MALLOC_TRIM_THRESHOLD_=0 RASCIL_DASK_SCHEDULER=${scheduler}:8786 python ${DFT_PATH}/src/fourier_transform_2d_dask.py | tee -a dft_test.log"

echo "About to execute $CMD"

eval $CMD

