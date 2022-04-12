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


def run_swift_params(k):

    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    test_conf = SWIFT_CONFIGS[k]
    # with performance_report(filename="dask-report-" + k + ".html"):
    main(test_conf, to_plot=False, use_dask=True)


if __name__ == "__main__":

    scheduler = os.environ.get("DASK_SCHEDULER", None)
    log.info("Scheduler: %s", scheduler)

    client = set_up_dask(scheduler_address=scheduler)

    for k, v in SWIFT_CONFIGS.items():
        log.info("Testing configuration: {}".format(k))
        run_swift_params(k)
        log.info("Finished test.")

    tear_down_dask(client)
