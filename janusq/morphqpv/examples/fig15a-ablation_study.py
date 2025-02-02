import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import numpy as np
import ray
from morphQPV.execute_engine.metric import fidelity
from copy import deepcopy
from data.Qbenchmark import layer_circuit_generator
from morphQPV.assume_guarantee.inference import infer_process_by_statevector
from scipy.stats import unitary_group
from morphQPV.execute_engine.excute import convert_state_to_density
from tqdm import tqdm



@ray.remote
def build_accuracy(idx, circuit, N_qubit, method="random", base_num=4):
    U = unitary_group.rvs(2**N_qubit)
    all_qubits = list(range(N_qubit))
    layer_circuit = deepcopy(circuit)
    state0 = np.zeros((2**N_qubit), dtype=np.complex128)
    state0[0] = 1
    input_state = U @ state0
    layer_circuit.insert(0, [{"name": "unitary", "params": U, "qubits": all_qubits}])
    # real_output_state = ExcuteEngine.excute_on_pennylane(layer_circuit,type='statevector')
    build_input, build_output = infer_process_by_statevector(
        layer_circuit[1:],
        input_state,
        method=method,
        base_num=base_num,
        target="input",
        optimizer="descent",
        learning_rate=0.01,
        steps=5000,
        jobname="build_accuracy",
        device="simulate",
    )
    return idx, method, fidelity(
        convert_state_to_density(input_state), convert_state_to_density(build_input)
    )


@ray.remote
def get_mean_fidelity(
    name, circuit, n_qubits, base_num=4, test_num=8, resultspath=None
):
    acc_dict = [{} for _ in range(test_num)]
    fids = ray.get(
            [
                build_accuracy.remote(
                    idx, circuit, n_qubits, method=method, base_num=base_num
                )
                for idx in tqdm(
                    range(test_num), desc=f"{name}_{n_qubits}_{base_num} testing"
                )
                for method in ["random", "base"]
            ]
        )
    for i,method,fid in fids:
        acc_dict[i][method] = fid

    with open(resultspath, "a") as f:
        for i in range(test_num):
            f.write(
                f'{name},{n_qubits},{base_num},{acc_dict[i]["base"]},{acc_dict[i]["random"]}\n'
            )


def get_results(name, n_qubits, csvpath, test_num=8, earlystop=True):
    with open(csvpath, "w") as f:
        f.write("name,qubits,samples,basis_gate,clliford\n")
    layer_circuit = layer_circuit_generator(name, n_qubits)
    if earlystop:
        base_nums = [2 ** (i) for i in range(n_qubits - 3, n_qubits + 4, 2)]
    else:
        base_nums = [2 ** (i) for i in range(n_qubits - 3, 2 * n_qubits, 2)]
    ray.get(
        [
            get_mean_fidelity.remote(
                name,
                layer_circuit,
                n_qubits,
                base_num=b,
                test_num=test_num,
                resultspath=csvpath,
            )
            for b in base_nums
        ]
    )


def plot_results(csvpath):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    data = pd.read_csv(csvpath)
    data.sort_values(by=["samples"], inplace=True)
    scale = 0.01
    fig = plt.figure(figsize=(2500 * scale, 600 * scale))
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
    mpl.rcParams["axes.linewidth"] = 5
    ## set ticks linewidth
    mpl.rcParams["xtick.major.size"] = 20
    mpl.rcParams["xtick.major.width"] = 5
    ax = plt.axes([0, 0, 0.38, 1])
    colors = ["#FF7F0E", "#3274A1"]
    data = data.groupby("samples").mean().reset_index()
    for m, col in enumerate(data.columns[-2:]):
        ax.plot(
            data.samples*2**2,
            data[col] * 100,
            marker="o",
            label=col,
            linewidth=3,
            markeredgewidth=5,
            markersize=25,
            color=colors[m],
        )
    ax.legend(loc="upper left", fontsize=60)
    ax.set_xticks(data.samples)
    ax.set_xscale("log", base=2)
    # ax.set_xticklabels(r"$2^{" + f"{int(np.log2(i))+2}" + "}$" for i in data.samples)
    ax.set_xlabel("# samples")
    ax.set_ylabel(r"accuracy (%)")
    ax.tick_params(axis="y", labelsize=60, pad=15)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 1.1, 0.2) * 100)
    ax.grid(axis="y", color="gray", linestyle="--", linewidth=5)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    plt.savefig(resultspath + "fig12(a)-ablation_study.pdf", bbox_inches="tight")


if __name__ == "__main__":
    directory = os.path.abspath(__file__).split("/")[-1].split(".")[0]
    resultspath = os.path.join(os.path.dirname(__file__), f"{directory}/")
    if not os.path.exists(resultspath):
        os.mkdir(resultspath)
    csvpath = f"{resultspath}accuracy.csv"
    import argparse

    parser = argparse.ArgumentParser(
        description="profiling the ablation study for fig 12, which compare the accuary with clliford gates and basis gates"
    )
    parser.add_argument(
        "--earlystop",
        action='store_true',
        default=False,
        help="run the program untill the clliford gates converge.",
    )
    parser.add_argument(
        "--numqubit",
        type=bool,
        default=7,
        help="run the program untill the clliford gates converge.",
    )
    args = parser.parse_args()
    get_results("qft", args.numqubit, csvpath, test_num=2, earlystop=args.earlystop)
    plot_results(csvpath)
