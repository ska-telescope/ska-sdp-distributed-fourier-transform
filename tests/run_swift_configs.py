import sys
import os
import logging
import numpy
import dask
import time
from distributed import performance_report

from src.swift_configs import SWIFT_CONFIGS
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask
from src.fourier_transform_2d_dask import main

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def run_swift_params():
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    scheduler = os.environ.get("DASK_SCHEDULER", None)
    log.info("Scheduler: %s", scheduler)

    client = set_up_dask(scheduler_address=scheduler)

    for k, v in SWIFT_CONFIGS.items():
        log.info("Testing configuration:", k)
        test_conf = SWIFT_CONFIGS[k]
        with performance_report(filename="dask-report-"+k+".html"):
            main(test_conf, to_plot=False, use_dask=True)

        log.info("Finished test.")

    tear_down_dask(client)

if __name__ == '__main__':

    run_swift_params()
