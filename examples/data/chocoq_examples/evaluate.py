import os
import time
import csv
import signal
import itertools
import sys
sys.path.append('../../..')
import logging
logging.basicConfig(level=logging.WARN)
import pandas as pd
pd.set_option('display.max_rows', None)  # display all rows
pd.set_option('display.max_columns', None)  # display all columns.
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from janusq.application.chocoq.chocoq.problems.facility_location_problem import generate_flp
from janusq.application.chocoq.chocoq.problems.graph_coloring_problem import generate_gcp
from janusq.application.chocoq.chocoq.problems.k_partition_problem import generate_kpp
from janusq.application.chocoq.chocoq.problems.job_scheduling_problem import generate_jsp
from janusq.application.chocoq.chocoq.problems.traveling_salesman_problem import generate_tsp
from janusq.application.chocoq.chocoq.problems.set_cover_problem import generate_scp
from janusq.application.chocoq.chocoq.solvers.optimizers import CobylaOptimizer, AdamOptimizer
from janusq.application.chocoq.chocoq.solvers.qiskit import (
    PenaltySolver, CyclicSolver, HeaSolver, ChocoSolver, 
    AerGpuProvider, AerProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, DdsimProvider,
)

# ----------------- Generate problems -----------------

num_cases = 10 # The number of cases in each benchmark
problem_scale = 4 # The problem scale
# 1 is the minimal scale like F1,K1,G1 in Table 1 of paper, 2 means F2 K2 ... 
# In CPU version, benchmarks with higher scale execute much slower when solving with baselines.

file_path = f"./scale_{problem_scale}"
os.makedirs(file_path, exist_ok=True)

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2), (2, 3), (3, 3), (3, 4)][:problem_scale], 1, 20)
gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1), (3, 2), (4, 2), (4, 3)][:problem_scale])
kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3), (6, 3, 5), (8, 3, 7), (9, 3, 8)][:problem_scale], 1, 20)

configs_pkg = flp_configs_pkg + gcp_configs_pkg + kpp_configs_pkg
with open(f"{file_path}/problem.config", "w") as file:
    for pkid, configs in enumerate(configs_pkg):
        for problem in configs:
            file.write(f'{pkid}: {problem}\n')

# ----------------- Evaluate depth -----------------

problems_pkg = flp_problems_pkg + gcp_problems_pkg + kpp_problems_pkg
metrics_lst = ['culled_depth', 'num_params']
solvers = [PenaltySolver, CyclicSolver, HeaSolver, ChocoSolver]
headers = ["pkid", 'method', 'layers'] + metrics_lst

def process_layer(prb, num_layers, solver, metrics_lst):
    opt = CobylaOptimizer(max_iter=200)
    ddsim = DdsimProvider()
    cpu = AerProvider()
    gpu = AerGpuProvider()
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        # Select CPU or GPU simulator
        provider = cpu if solver in [PenaltySolver, CyclicSolver, HeaSolver] else ddsim,
        # provider = gpu if solver in [PenaltySolver, CyclicSolver, HeaSolver] else ddsim,
        num_layers = num_layers,
        shots = 1024,
    )
    metrics = used_solver.circuit_analyze(metrics_lst)
    return metrics

set_timeout = 60 * 60 * 24 # Set timeout duration
num_complete = 0
with open(f"{file_path}/evaluate_depth.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write headers once

num_processes_cpu = os.cpu_count() // 2
with ProcessPoolExecutor(max_workers=num_processes_cpu) as executor:
    futures = []
    for solver in solvers:
        for pkid, problems in enumerate(problems_pkg):
            for problem in problems:
                if solver == ChocoSolver:
                    num_layers = 1
                else:
                    num_layers = 5
                future = executor.submit(process_layer, problem, num_layers, solver, metrics_lst)
                futures.append((future, pkid, solver.__name__, num_layers))

    start_time = time.perf_counter()
    for future, pkid, solver, num_layers in futures:
        current_time = time.perf_counter()
        remaining_time = max(set_timeout - (current_time - start_time), 0)
        diff = []
        try:
            result = future.result(timeout=remaining_time)
            diff.extend(result)
            # print(f"Task for problem {pkid}, num_layers {num_layers} executed successfully.")
        except MemoryError:
            diff.append('memory_error')
            print(f"Task for problem {pkid}, num_layers {num_layers} encountered a MemoryError.")
        except TimeoutError:
            diff.append('timeout')
            print(f"Task for problem {pkid}, num_layers {num_layers} timed out.")
        finally:
            row = [pkid, solver, num_layers] + diff
            with open(f"{file_path}/evaluate_depth.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)  # Write row immediately
            num_complete += 1
            if num_complete == len(futures):
                print(f'Data has been written to {file_path}/evaluate_depth.csv')
                for process in executor._processes.values():
                    os.kill(process.pid, signal.SIGTERM)

# ----------------- Evaluate other metrics -----------------

problems_pkg = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
        enumerate(gcp_problems_pkg),
        enumerate(kpp_problems_pkg),
    )
)
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
solvers = [PenaltySolver, CyclicSolver, HeaSolver, ChocoSolver]
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics

def process_layer(prb, num_layers, solver):
    opt = CobylaOptimizer(max_iter=200)
    ddsim = DdsimProvider()
    cpu = AerProvider()
    gpu = AerGpuProvider()
    prb.set_penalty_lambda(400)
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        # Select CPU or GPU simulator
        provider = cpu if solver in [PenaltySolver, CyclicSolver, HeaSolver] else ddsim,
        # provider = gpu if solver in [PenaltySolver, CyclicSolver, HeaSolver] else ddsim,
        num_layers = num_layers,
        shots = 1024,
    )
    used_solver.solve()
    eval = used_solver.evaluation()
    time = list(used_solver.time_analyze())
    run_times = used_solver.run_counts()
    return eval + time + [run_times]

all_start_time = time.perf_counter()
set_timeout = 60 * 60 * 2 # Set timeout duration
num_complete = 0

with open(f"{file_path}/evaluate_other.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

num_processes_cpu = os.cpu_count()
# pkid-pbid: problem package id - problem id
for pkid, (diff_level, problems) in enumerate(problems_pkg):
    for solver in solvers:
        if solver in [PenaltySolver, CyclicSolver, HeaSolver]:
            num_processes = 2**(4 - diff_level) + 1
        else:
            num_processes = 100

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            if solver == ChocoSolver:
                layer = 1
            else:
                layer = 5

            for pbid, prb in enumerate(problems):
                print(f'{pkid}-{pbid}, {layer}, {solver} build')
                future = executor.submit(process_layer, prb, layer, solver)
                futures.append((future, prb, pkid, pbid, layer, solver.__name__))

            start_time = time.perf_counter()
            for future, prb, pkid, pbid, layer, solver in futures:
                current_time = time.perf_counter()
                remaining_time = max(set_timeout - (current_time - start_time), 0)
                diff = []
                try:
                    metrics = future.result(timeout=remaining_time)
                    diff.extend(metrics)
                    # print(f"Task for problem {pkid}-{pbid} L={layer} {solver} executed successfully.")
                except MemoryError:
                    print(f"Task for problem {pkid}-{pbid} L={layer} {solver} encountered a MemoryError.")
                    for dict_term in evaluation_metrics:
                        diff.append('memory_error')
                except TimeoutError:
                    print(f"Task for problem {pkid}-{pbid} L={layer} {solver} timed out.")
                    for dict_term in evaluation_metrics:
                        diff.append('timeout')
                except Exception as e:
                    print(f"An error occurred: {e}")
                finally:
                    row = [pkid, pbid, layer, len(prb.variables), len(prb.lin_constr_mtx), solver] + diff
                    with open(f"{file_path}/evaluate_other.csv", mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)  # Write row immediately
                    num_complete += 1
                    if num_complete == len(futures):
                        print(f'problem_pkg_{pkid} has finished')
                        for process in executor._processes.values():
                            os.kill(process.pid, signal.SIGTERM)

print(f'Data has been written to {file_path}/evaluate_other.csv')
print(time.perf_counter() - all_start_time)
