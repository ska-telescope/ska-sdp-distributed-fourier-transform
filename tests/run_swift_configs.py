import sys
import os
import logging
import numpy
import dask
import time
from distributed import performance_report

from src.fourier_transform.algorithm_parameters import BaseArrays
from src.swift_configs import SWIFT_CONFIGS
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask
from src.fourier_transform_2d_dask import main

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def run_swift_params(k, client_dask):

    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    test_conf = SWIFT_CONFIGS[k]

    base_arrays_class = BaseArrays(**test_conf)

    _ = base_arrays_class.pswf

    with performance_report(filename="dask-report-2d.html"):
        main(
            base_arrays_class,
            test_conf,
            to_plot=False,
            use_dask=True,
            client=client_dask,
        )


if __name__ == "__main__":

    scheduler = os.environ.get("DASK_SCHEDULER", None)
    log.info("Scheduler: %s", scheduler)

    client_dask = set_up_dask(scheduler_address=scheduler)

    for k, v in SWIFT_CONFIGS.items():
        log.info("Testing configuration: {}".format(k))
        run_swift_params(k, client_dask)
        log.info("Finished test.")

    tear_down_dask(client_dask)
