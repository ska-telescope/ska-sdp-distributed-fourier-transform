"""
The dask wrapper.

We wrap the Dask delayed implementation in a dask
wrapper where if use_dask is set to True, the dask wrapper
function will call the dask.delayed option for the computation.
"""

import functools
import logging

import dask
from distributed import Client

log = logging.getLogger("fourier-logger")


def dask_wrapper(func):
    """
    The Dask wrapper function

    If arg use_dask = True and arg nout is provided,
    then function is run with dask.delayed.
    Else, it is executed in serial as normal.
    """

    @functools.wraps(func)  # preserves information about the original function
    def wrapper(*args, **kwargs):
        try:
            use_dask = kwargs["use_dask"]
            nout = kwargs["nout"]
        except KeyError:
            use_dask = False
            nout = None

        if use_dask:
            result = dask.delayed(func, nout=nout)(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        return result

    return wrapper


def set_up_dask(scheduler_address=None):
    """
    Set up the Dask Client

    :param scheduler_address: IP_address:PORT of scheduler
                if None, a local cluster is created with machine resources
    :return: Dask client
    """
    client = Client(scheduler_address)
    log.info(client.dashboard_link)
    return client


def tear_down_dask(client):
    """
    Close the Dask Client
    """
    client.close()
