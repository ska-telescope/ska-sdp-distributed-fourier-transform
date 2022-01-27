import os
import functools
import dask
from distributed import Client


def dask_wrapper(use_dask=True):
    def decorator_dask(func):
        @functools.wraps(func)  # preserves information about the original function
        def wrapper(*args, **kwargs):
            if use_dask:
                result = dask.delayed(func)(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator_dask


def set_up_dask():
    current_env_var = os.getenv("USE_DASK")
    os.environ["USE_DASK"] = "True"
    client = Client()  # set up local cluster on your laptop
    print(client.dashboard_link)
    return client, current_env_var


def tear_down_dask(client, original_env_var):
    client.close()
    if original_env_var is None:
        os.environ.pop("USE_DASK")
    else:
        os.environ["USE_DASK"] = original_env_var
