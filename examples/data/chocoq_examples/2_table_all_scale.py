# nohup python3 2_table_all_scale.py > 2_table_all_scale.output 2>&1 &
import os
import time
import csv
import signal
import random
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from chocoq.problems.facility_location_problem import generate_flp
from chocoq.problems.graph_coloring_problem import generate_gcp
from chocoq.problems.k_partition_problem import generate_kpp
from chocoq.problems.job_scheduling_problem import generate_jsp
from chocoq.problems.traveling_salesman_problem import generate_tsp
from chocoq.problems.set_cover_problem import generate_scp
from chocoq.solvers.optimizers import CobylaOptimizer, AdamOptimizer
from chocoq.solvers.qiskit import (
    PenaltySolver, CyclicSolver, HeaSolver, ChocoSolver, 
    AerGpuProvider, AerProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, DdsimProvider,
)

import pandas as pd
pd.set_option('display.max_rows', None)  # display all rows
pd.set_option('display.max_columns', None)  # display all columns.

num_cases = 10  # The number of cases in each benchmark
problem_scale = 4 # The problem scale, 1 is the minimal scale like F1,K1,G1 in Table 1 of paper
#2 means F2 K2 ... In CPU version, this benchmarks with higher scale is much slower when solving with baselines.

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2), (2, 3), (3, 3), (3, 4)][:problem_scale], 1, 20)
gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1), (3, 2), (4, 2), (4, 3)][:problem_scale])
kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3), (6, 3, 5), (8, 3, 7), (9, 3, 8)][:problem_scale], 1, 20)

configs_pkg = flp_configs_pkg + gcp_configs_pkg + kpp_configs_pkg
with open(f"2_table_all_scale.config", "w") as file:
    for pkid, configs in enumerate(configs_pkg):
        for problem in configs:
            file.write(f'{pkid}: {problem}\n')

new_path = '2_table_depth_all_scale'

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
        # cpu simulator, comment it when use GPU
        # provider = cpu if solver in [PenaltySolver, CyclicSolver, HeaSolver] else ddsim,
        # uncomment the line below to use GPU simulator
        provider = gpu if solver in [PenaltySolver, CyclicSolver, HeaSolver] else ddsim,
        num_layers = num_layers,
        shots = 1024,
    )
    metrics = used_solver.circuit_analyze(metrics_lst)
    return metrics

if __name__ == '__main__':
    set_timeout = 60 * 60 * 24 # Set timeout duration
    num_complete = 0
    with open(f'{new_path}.csv', mode='w', newline='') as file:
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
                print(f"Task for problem {pkid}, num_layers {num_layers} executed successfully.")
            except MemoryError:
                diff.append('memory_error')
                print(f"Task for problem {pkid}, num_layers {num_layers} encountered a MemoryError.")
            except TimeoutError:
                diff.append('timeout')
                print(f"Task for problem {pkid}, num_layers {num_layers} timed out.")
            finally:
                row = [pkid, solver, num_layers] + diff
                with open(f'{new_path}.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)  # Write row immediately
                num_complete += 1
                if num_complete == len(futures):
                    print(f'Data has been written to {new_path}.csv')
                    for process in executor._processes.values():
                        os.kill(process.pid, signal.SIGTERM)

file_path = '2_table_depth_all_scale.csv'

df = pd.read_csv(file_path)

grouped_df = df.groupby(['pkid', 'layers', 'method'], as_index=False).agg({
    "culled_depth": 'mean',
})

values = ["culled_depth"]
pivot_df = grouped_df.pivot(index =['pkid'], columns='method', values=values)

method_order = ['PenaltySolver', 'CyclicSolver', 'HeaSolver', 'ChocoSolver']
pivot_df = pivot_df.reindex(columns=pd.MultiIndex.from_product([values, method_order]))

print(pivot_df)

new_path = '2_table_evaluate_all_scale'

problems_pkg = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
        enumerate(gcp_problems_pkg),
        enumerate(kpp_problems_pkg),
    )
)

solvers = [PenaltySolver, CyclicSolver, HeaSolver, ChocoSolver]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
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
        # 根据配置的环境选择CPU或GPU
        # provider = cpu if solver in [PenaltySolver, CyclicSolver, HeaSolver] else ddsim,
        provider = gpu if solver in [PenaltySolver, CyclicSolver, HeaSolver] else ddsim,
        num_layers = num_layers,
        shots = 1024,
    )
    used_solver.solve()
    eval = used_solver.evaluation()
    time = list(used_solver.time_analyze())
    run_times = used_solver.run_counts()
    return eval + time + [run_times]

if __name__ == '__main__':
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 2 # Set timeout duration
    num_complete = 0

    with open(f'{new_path}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # 写入标题

    num_processes_cpu = os.cpu_count()
    # pkid-pbid: 问题包序-包内序号
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
                        print(f"Task for problem {pkid}-{pbid} L={layer} {solver} executed successfully.")
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
                        with open(f'{new_path}.csv', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)  # Write row immediately
                        num_complete += 1
                        if num_complete == len(futures):
                            print(f'problem_pkg_{pkid} has finished')
                            for process in executor._processes.values():
                                os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {new_path}.csv')
    print(time.perf_counter()- all_start_time)


import pandas as pd

file_path1 = '2_table_depth_all_scale.csv'
df1 = pd.read_csv(file_path1, encoding='utf-8')

grouped_df1 = df1.groupby(['pkid', 'layers', 'method'], as_index=False).agg({
    "culled_depth": 'mean',
})

pivot_df1 = grouped_df1.pivot(index=['pkid'], columns='method', values=["culled_depth"])

method_order1 = ['PenaltySolver', 'CyclicSolver', 'HeaSolver', 'ChocoSolver']
pivot_df1 = pivot_df1.reindex(columns=pd.MultiIndex.from_product([["culled_depth"], method_order1]))

file_path2 = '2_table_evaluate_all_scale.csv'
df2 = pd.read_csv(file_path2)
df2 = df2.drop(columns=['pbid','ARG'])


df2[['best_solution_probs', 'in_constraints_probs', 'iteration_count',
     'classcial', 'quantum', 'run_times']] = df2[['best_solution_probs', 'in_constraints_probs', 'iteration_count',
                                                  'classcial', 'quantum', 'run_times']].apply(pd.to_numeric, errors='coerce')
grouped_df2 = df2.groupby(['pkid', 'layers', 'variables', 'constraints', 'method'], as_index=False).agg({
    # "ARG": 'mean',
    'in_constraints_probs': 'mean',
    'best_solution_probs': 'mean',
    'iteration_count': 'mean',
    'classcial': 'mean',
    'run_times': 'mean',
})

pivot_df2 = grouped_df2.pivot(index=['pkid', 'variables', 'constraints'], columns='method', values=["best_solution_probs", 'in_constraints_probs'])

method_order2 = ['PenaltySolver', 'CyclicSolver', 'HeaSolver', 'ChocoSolver']
pivot_df2 = pivot_df2.reindex(columns=pd.MultiIndex.from_product([["best_solution_probs", 'in_constraints_probs'], method_order2]))

merged_df = pd.merge(pivot_df1, pivot_df2, on='pkid', how='inner')

merged_df = merged_df[['culled_depth', 'best_solution_probs', 'in_constraints_probs']]
merged_df = merged_df.rename(columns={
    'culled_depth': 'Circuit depth',
    'best_solution_probs': 'Success rate (%)',
    'in_constraints_probs': 'In-constraints rate (%)'
})

merged_df = merged_df.rename(columns={
    'PenaltySolver': 'Penalty',
    'CyclicSolver': 'Cyclic',
    'HeaSolver': 'HEA',
    'ChocoSolver': 'Choco-Q'
})

merged_df.index = ['F1', 'F2', 'F3', 'F4', 'G1', 'G2', 'G3', 'G4', 'K1', 'K2', 'K3', 'K4']

print(merged_df)

import pandas as pd

# Assuming 'merged_df' already contains the necessary data (after the previous steps)

# Calculate the improvement for each row

# Circuit depth improvement: cyclic / Choco-Q
merged_df['Circuit_depth_improvement'] = merged_df[('Circuit depth', 'Cyclic')] / merged_df[('Circuit depth', 'Choco-Q')]

# Success rate improvement: Choco-Q / cyclic
merged_df['Success_rate_improvement'] = merged_df[('Success rate (%)', 'Choco-Q')] / merged_df[('Success rate (%)', 'Cyclic')]

# In-constraints rate improvement: Choco-Q / cyclic
merged_df['In_constraints_rate_improvement'] = merged_df[('In-constraints rate (%)', 'Choco-Q')] / merged_df[('In-constraints rate (%)', 'Cyclic')]

# Filter out rows where any improvement column has a zero denominator or zero numerator (to avoid division by zero)
valid_rows = merged_df[(merged_df[('Circuit depth', 'Cyclic')] != 0) & (merged_df[('Circuit depth', 'Choco-Q')] != 0) &
                       (merged_df[('Success rate (%)', 'Cyclic')] != 0) & (merged_df[('Success rate (%)', 'Choco-Q')] != 0) &
                       (merged_df[('In-constraints rate (%)', 'Cyclic')] != 0) & (merged_df[('In-constraints rate (%)', 'Choco-Q')] != 0)]

# Calculate the average improvement for each metric
avg_circuit_depth_improvement = valid_rows['Circuit_depth_improvement'].mean()
avg_success_rate_improvement = valid_rows['Success_rate_improvement'].mean()
avg_in_constraints_rate_improvement = valid_rows['In_constraints_rate_improvement'].mean()

improvement_table = pd.DataFrame({
    'Circuit Depth': [avg_circuit_depth_improvement],
    'Success Rate': [avg_success_rate_improvement],
    'In-constraints Rate': [avg_in_constraints_rate_improvement]
}, index=['Improvement relative to Cyclic'])

print(improvement_table)
