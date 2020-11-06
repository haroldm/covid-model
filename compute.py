#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('..')

import os
import multiprocessing
import multiprocessing.pool
import pandas as pd
import time

from covid.models.generative import GenerativeModel
from covid.data import summarize_inference_data

from covid.data_fr import get_and_process_covidtracking_data

# Each model runs 4 processes.
# This variable specifies how many models run in parallel.
# With 4 there is a total of 4 * 4 = 16 processes.
# Set this value such as number of processes is less than number of threads
# of your machine.
PARALLEL = 4

# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def process_data(data):
    (dep, model_data) = data
    print("Computing model for " + dep)
    gm = GenerativeModel(dep, model_data)
    gm.sample()

    result = summarize_inference_data(gm.inference_data)
    return (dep, result)

# Timer
start_time = time.time()

# Download and process Covid tracking data
df = get_and_process_covidtracking_data(run_date=pd.Timestamp.today()-pd.Timedelta(days=1))

# Build a set of all deps
deps = set([])
for (dep, date) in df.index.values:
    deps.add(dep)

# Sort deps
deps = list(deps)
deps.sort()

# Split `df` by deps
df_by_deps = []
for dep in deps:
    df_by_deps.append((dep, df.loc[dep]))

# Speed up computation for testing
# df_by_deps = df_by_deps[:4]

# Compute 4 deps at once (16 threads)
pool = MyPool(PARALLEL)
results = pool.map(process_data, df_by_deps)
pool.close()
pool.join()

# Concatenate results
results_dep = [res[0] for res in results]
results_rt = [res[1] for res in results]
results = pd.concat(results_rt, keys=results_dep)

# Save output
results.to_json("results.json", orient="split")

# Show elapsed time
print("--- %s seconds ---" % (time.time() - start_time))