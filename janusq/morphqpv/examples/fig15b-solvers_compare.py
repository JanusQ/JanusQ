import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


import numpy as np
from copy import deepcopy
from data.Qbenchmark import layer_circuit_generator
from morphQPV.assume_guarantee.inference import infer_process_by_statevector
from scipy.stats import unitary_group
from morphQPV.execute_engine.excute import ExcuteEngine
from tqdm import tqdm
from time import perf_counter
import ray


def get_run_time(circuit, n_qubits, method="random", base_num=4):
    U = unitary_group.rvs(2**n_qubits)
    all_qubits = list(range(n_qubits))
    layer_circuit = deepcopy(circuit)
    state0 = np.zeros((2**n_qubits), dtype=np.complex128)
    state0[0] = 1
    input_state = U @ state0
    layer_circuit.insert(0, [{"name": "unitary", "params": U, "qubits": all_qubits}])
    for optimizer in ["descent", "quadratic", "annealing"]:
        start = perf_counter()
        build_input, build_output = infer_process_by_statevector(
            layer_circuit[1:],
            input_state,
            optimizer=optimizer,
            method=method,
            base_num=base_num,
            target="input",
        )
        end = perf_counter()
        yield optimizer, end - start


@ray.remote
def get_data(name, n_qubits):

    layer_circuit = layer_circuit_generator(name, n_qubits)
    n_qubits = ExcuteEngine.get_qubits(layer_circuit)[-1] + 1
    base_nums = 2**2
    time_dict = {}
    for opt, time in get_run_time(layer_circuit, n_qubits, base_num=base_nums):
        time_dict[opt] = time
    with open(csvpath, "a") as f:
        f.write(
            f'{name},{n_qubits},{time_dict["annealing"]*5},{time_dict["descent"]},{time_dict["quadratic"]}\n'
        )


def get_solver_time(csvpath,earlystop=False):
    if earlystop:
        qubit_range = range(2, 12, 2)
    else:
        qubit_range = range(2, 13, 1)
    with open(csvpath, "w") as f:
        f.write("name,qubits,annealing,descent,quadratic\n")
    ray.get(
        [
            get_data.remote(name, qubits)
            for name in ["qknn", "qft", "bv"]
            for qubits in qubit_range
        ]
    )


def plot_solver_time(csvpath):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pandas as pd

    runtime = pd.read_csv(csvpath)
    runtime
    scale = 0.0105
    from tqdm import tqdm

    fig = plt.figure(figsize=(1161.75 * scale, 848.98 * scale))
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["font.size"] = 62
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Times New Roman"
    mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
    mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"
    # set axes linewidth
    linewidth = 5
    # set axes linewidth
    mpl.rcParams["axes.linewidth"] = linewidth
    # set ticks linewidth
    mpl.rcParams["xtick.major.size"] = 20 * scale / 0.01
    mpl.rcParams["xtick.major.width"] = linewidth
    mpl.rcParams["ytick.major.size"] = 20 * scale / 0.01
    mpl.rcParams["ytick.major.width"] = linewidth
    mpl.rcParams["xtick.minor.size"] = 10 * scale / 0.01
    mpl.rcParams["xtick.minor.width"] = linewidth
    mpl.rcParams["ytick.minor.size"] = 10 * scale / 0.01
    mpl.rcParams["ytick.minor.width"] = linewidth
    markersize = 20 * scale / 0.01
    ax = plt.axes()
    colors = ["#E8E131", "#FF6F45", "#3274A1"]
    qubits = sorted(runtime["qubits"].unique())
    runtime_mean = (
        runtime.groupby("qubits", as_index=False).mean().sort_values(by="qubits")
    )
    runtime_std = (
        runtime.groupby("qubits", as_index=False).std().sort_values(by="qubits")
    )
    names = ["annealing", "descent", "descent"]
    for m, col in tqdm(enumerate(runtime.columns[2:])):
        ax.plot(
            runtime_mean["qubits"],
            runtime_mean[col] * 1e3,
            "o-",
            markersize=markersize,
            color=colors[m],
            linewidth=linewidth,
        )
        ax.errorbar(
            runtime_mean["qubits"],
            runtime_mean[col] * 1e3,
            yerr=runtime_std[col] * 1e3,
            fmt="o",
            markersize=markersize,
            color=colors[m],
            linewidth=linewidth,
            label=names[m],
        )
    # ax.legend( fontsize=62,frameon=False,bbox_to_anchor=(0.5, 1.15), loc='center', ncol=2, borderaxespad=0.)
    ax.set_yscale("log")
    ax.set_xticks(range(2, 13, 2))
    ax.set_xticks(range(2, 13), minor=True)
    ax.set_xticklabels(range(2, 13, 2), fontsize=62)
    ax.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6])
    # ax.set_yticks(np.arange(1e2,1e7,1e2),minor=True)
    # ax.minorticks_on()
    # set y axis  minor ticks on
    ax.yaxis.set_tick_params(which="major", left=False)
    # ax.set_ylim(0, 1e7)
    ax.tick_params(axis="y", which="major", left=False)
    ax.set_xlabel("# qubits")
    ax.set_ylabel("runtime (ms)")
    plt.legend(frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="y", linestyle="--", linewidth=2, which="major")
    # ax.grid(axis='y', linestyle='--', linewidth=1,which='minor')
    plt.savefig(f"{resultspath}runtime.svg", bbox_inches="tight")


if __name__ == "__main__":
    resultspath = "examples/fig12(b)-solvers_compare/"
    import argparse

    parser = argparse.ArgumentParser(
        description="profiling the validation time  for solvers"
    )
    parser.add_argument(
        "--earlystop",
        action="store_true",
        default=False,
        help="reduce the sampling time",
    )
    s = perf_counter()
    args = parser.parse_args()
    if not os.path.exists(resultspath):
        os.mkdir(resultspath)
    csvpath = f"{resultspath}time.csv"
    get_solver_time(csvpath, earlystop=args.earlystop)
    plot_solver_time(csvpath)
    e = perf_counter()
    print("totol speed time:", e - s)
