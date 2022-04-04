#!/bin/bash
#!
#! Dask job script for Astrolab HPC Cluster
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH --job-name distft
#! Which project should be charged:
#SBATCH --account astro
#! How many whole nodes should be allocated?
#SBATCH --nodes=13
#! How many (MPI) tasks will there be in total? (<= nodes*16)
#SBATCH --ntasks=13
#! Memory limit: ASTROLAB has roughly 1000GB per node
#SBATCH --mem 100GB
#! How much wallclock time will be required?
#SBATCH --time=01:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL,END
#! Where to send email messages
#SBATCH --mail-user=wangfeng@cnlab.net
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#! Do not change:
#! Partitions
#SBATCH --partition astro-cpu
#! Cluster name
#SBATCH --clusters astrolab

#! Same switch
#SBATCH --switches=1
#! Included nodes
#SBATCH --nodelist astrolab-hpc-1
#! Excluded nodes
##SBATCH --exclude astrolab-hpc-[2-13]
#! Output
#SBATCH --output ./slurm_out/slurm-sim-%A.out
#! Exclusive
#SBATCH --exclusive

module purge

module load miniconda3-4.9.2-gcc-9.3.0-fihj225
source /home/${USER}/.bashrc

conda activate ${HOME}/python/rascil

#! spack load ucx /xo4jd7b

#！ export DASK_CONFIG=./dask_config_distrim_hight_split_mini.yaml
#！setup distributed fourier transform
DFTPATH=${HOME}/work/ska-sdp-distributed-fourier-transform
echo "D-FT path : $DFTPATH"
export PYTHONPATH=$DFTPATH:$PYTHONPATH

#! Set up python
echo -e "[main]Running python: `which python`"
echo -e "[main]which main in node: `hostname`"
echo -e "[main]Running dask-scheduler: `which dask-scheduler`"

#! cd $SLURM_SUBMIT_DIR
cd $DFTPATH
echo -e "[main]Changed directory to `pwd`.\n"

JOBID=${SLURM_JOB_ID}
echo ${SLURM_JOB_NODELIST}

#! Run Scheduler
localdir=/tmp
protool=tcp
scheduler=192.168.99.201
port=9786
dashboardport=9787
export DASK_SCHEDULER=${protool}://${scheduler}:${port}
dask-scheduler --dashboard-address $dashboardport --protocol ${protool} --interface ib0 --port $port &
echo dask-scheduler started on ${scheduler}:${port}
sleep 5

#! Run workers
srun -o ./slurm_out/srun_%x_d%j_worker_%n.out -e ./slurm_out/srun_error_%x_%j_worker_%n.out dask-worker --nprocs 2 --nthreads 1 --interface ib0 --memory-limit 115GiB --local-directory ${localdir} ${protool}://${scheduler}:${port} &
echo dask-worker started on all nodes

sleep 1
echo "[main]Scheduler and workers now running"
python src/fourier_transform_2d_dask.py

sleep 20
echo "Done"