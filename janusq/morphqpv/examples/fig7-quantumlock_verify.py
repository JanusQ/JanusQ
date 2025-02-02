import numpy as np
import pennylane as qml
import ray
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from morphQPV.cases.quantum_lock import (
    generate_qml_lock_circuit,
    quantum_backdoor_attack,
    get_bit_string,
)

np.random.seed(np.random.randint(0, 1000))


# @ray.remote
def profilling_qubit(n_qubit, method):
    lock = get_bit_string(n_qubit)
    hidden = get_bit_string(n_qubit)
    if lock == hidden:
        return profilling_qubit(n_qubit, method)
    inference_circuit = generate_qml_lock_circuit(n_qubit, lock, hidden_key=hidden)
    num = quantum_backdoor_attack(inference_circuit, lock, hidden, n_qubit, method=method)
    return num

def scale_profilling(qubit_range, method):
    # ray.init(ignore_reinit_error=True, n_cpus=4)
    with open(f"{resultspath}quantumlock.csv", "w") as f:
        f.write("n_qubits,bases\n")
    # nimimal_nums = ray.get(
    #     [profilling_qubit.remote(qubit_num, method) for qubit_num in qubit_range]
    # )
    nimimal_nums =  [profilling_qubit(qubit_num, method) for qubit_num in qubit_range]
    for i, qubit_num in enumerate(qubit_range):
        with open(f"{resultspath}quantumlock.csv", "a") as f:
            f.write(f"{qubit_num},{nimimal_nums[i]}\n")
    # nimimal_nums = [profilling_qubit(qubit_num,method) for qubit_num in qubit_range]
    return nimimal_nums


def plot_scale_profilling():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    data = pd.read_csv(f"{resultspath}quantumlock.csv")
    data.sort_values(by="n_qubits", inplace=True)
    n_qubits = data.n_qubits
    Ns = 2**n_qubits
    ## qutio uses a fixed number of test cases generated in advance, that is, Ns
    qutio = Ns
    ## state assertion uses random selection of the ground state until the backdoor is found, so the expected number of test cases is in line with the hypergeometric distribution
    ndd_num = (1 - ((Ns - 1) / Ns) ** Ns) * Ns
    scale = 0.01
    fig = plt.figure(figsize=(2500 * scale, 1200 * scale))
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
    mpl.rcParams["xtick.minor.size"] = 10
    mpl.rcParams["xtick.minor.width"] = 5
    mpl.rcParams["ytick.major.size"] = 20
    mpl.rcParams["ytick.major.width"] = 5
    mpl.rcParams["ytick.minor.size"] = 10
    mpl.rcParams["ytick.minor.width"] = 5
    axes = plt.axes()
    axes.plot(
        n_qubits,
        qutio,
        "^",
        color="#7EB57E",
        label="Quito",
        linestyle="-",
        linewidth=5,
        markersize=30,
    )
    axes.plot(
        n_qubits,
        ndd_num,
        "^",
        color="#FF8663",
        label="State Assertion",
        linestyle="dashed",
        linewidth=5,
        markersize=30,
    )
    axes.plot(
        n_qubits,
        data.bases,
        "^",
        color="#4885B0",
        label="MORPH",
        linestyle="-",
        linewidth=5,
        markersize=30,
    )
    for i, q in enumerate(n_qubits):
        speed = round(ndd_num[i] / data.bases.iloc[i], 1)
        axes.plot(
            [q, q], [ndd_num[i], data.bases.iloc[i]], color="grey", linewidth=4, ls="-"
        )
        axes.text(q, data.bases.iloc[i] * 2, f"{speed}" + r"$\times$")
    axes.grid(which="major", linestyle="--", axis="y", linewidth=5)
    axes.set_xticks(n_qubits )
    axes.set_xlabel("# qubits")
    axes.set_ylabel("# samples")
    axes.set_yscale("log")
    axes.legend(frameon=False)
    fig.savefig(
        f"{resultspath}quantumlock.svg", dpi=600, format="svg", bbox_inches="tight"
    )


if __name__ == "__main__":
    ## build the arguments by command line
    import argparse

    parser = argparse.ArgumentParser(
        description="profiling the backdoor of quantum lock, and compare the number of samples required for the backdoor to be found"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="random",
        help="the method for sampling, random for clliford gate, base for the basis gate",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=21,
        help="the maximum number of qubits for profiling, default is 10. The evaluation will be performed from 4 to the maximum number",
    )
    parser.add_argument(
        "--resultspath",
        type=str,
        default="examples/fig7-quantumlock/",
        help="the path to save the results",
    )
    args = parser.parse_args()
    if not os.path.exists(args.resultspath):
        os.makedirs(args.resultspath)
    resultspath = args.resultspath
    scale_profilling(range(11, args.qubits + 1, 2), args.method)
    plot_scale_profilling()
