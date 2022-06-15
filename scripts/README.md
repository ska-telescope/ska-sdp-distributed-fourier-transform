## Running evaluation scripts

Two scripts are provided in the `scripts` directory to investigate the performance of Distributed FT algorithm 
using `class StreamingDistributedFFT`

`memory_comsumption.py` can be used for memory consumption evaluation. The example is shown as follows:

```bash
python scripts/memory_consumption.py --swift_config 8k[1]-n4k-512
```

`performance_queue.py` can be used to evaluate the memory consumption and performance of Dask execution using queue 
and batch mode optimization. The command is :
```bash
python scripts/performance_queue.py --swift_config 8k[1]-n4k-512 --hdf5_prefix path/to/data
```
If you want to verify and output approximate values of subgrid in hdf5 format, you need to use the parameter `--check_results=True`

Parameters `--max_work_tasks`, `--max_NMBF_BF_waiting_task` and `--batch_i1_number_task` can adjust the number of batches to improve execution efficiency