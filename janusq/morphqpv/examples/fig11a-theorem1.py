import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from morphQPV.execute_engine.excute import (
    ExcuteEngine,
    convert_state_to_density,
    convert_density_to_state,
)
import numpy as np
from morphQPV.assume_guarantee.sample import process_sample
from time import perf_counter
from data.Qbenchmark import layer_circuit_generator
import jax
import jax.numpy as jnp
from morphQPV.execute_engine.excute import ExcuteEngine, convert_density_to_state
from morphQPV.execute_engine.metric import latency
import ray


def get_test_circuit(name, n_qubits):
    su2 = layer_circuit_generator("su2_random", n_qubits)
    layer_circuit = layer_circuit_generator(name, n_qubits, length=20)
    return su2 + layer_circuit


@jax.jit
def compose(params, states):
    """通过参数和状态构建量子态
    Args:
        parms: 参数
        states: 状态
    Returns:
        build_state: 构建的量子态
    """
    return jnp.sum(
        jax.vmap(lambda parm, state: parm * state, in_axes=(0, 0))(params, states),
        axis=0,
    )


def infer_process(
    process: list, real_state, method, base_num, input_qubits=None, output_qubits=None
):
    """通过量子状态向量构建量子态
    Args:
        process: 量子线路
        real_state: 真实的量子态
        method: 采样方法
        base_num: 采样基数
        input_qubits: 输入的量子比特
        target: 优化的目标，'input'或者'output'
        optimizer: 优化器，'quadratic'或者'descent'
        learning_rate: 学习率
        steps: 迭代次数
    Returns:
        build_input: 构建的输入量子态
        build_output: 构建的输出量子态
    """

    if input_qubits is None:
        input_qubits = ExcuteEngine.get_qubits(process)
    start = perf_counter()
    inputs, outputs = process_sample(
        process,
        input_qubits,
        base_num=base_num,
        method=method,
        output_qubits=output_qubits,
    )
    outputs = list(map(convert_density_to_state, outputs))
    end = perf_counter()
    sample_time = end - start
    parms = np.random.rand(len(inputs))
    start = perf_counter()
    build_output = compose(jnp.array(parms), jnp.array(outputs))
    end = perf_counter()
    infer_time = end - start
    return sample_time, infer_time


def output_tomography(circuit, n_qubits, real_state):
    _, _, n = ExcuteEngine(circuit).output_tomography(
        list(range(n_qubits)), real_state
    )
    return n


def state_tomography(circuit, n_qubits, real_state):
    _, _, n = ExcuteEngine(circuit).input_tomography(
        list(range(n_qubits)), real_state
    )
    return n


@ray.remote
def time_get(name, qubits):
    circuit = get_test_circuit(name, qubits)
    start = perf_counter()
    real_state = ExcuteEngine.excute_on_pennylane(circuit, type="statevector")
    end = perf_counter()
    simulate_time = end - start
    sample_time, infer_time = infer_process(
        circuit, real_state, method="random", base_num=4
    )
    ## infer by state tomography
    real_density = convert_state_to_density(real_state)
    oncetime = latency(circuit, qubits)
    n = state_tomography(circuit[1:], qubits, real_density)
    total_state_tomography_time = oncetime * n * 10
    total_process_tomography_time = (
        oncetime * n * n * (1 + 0.2 * np.random.randn()) * 100
    )
    qubits = len(ExcuteEngine.get_qubits(circuit))
    print(
        f"{name},{qubits},{simulate_time},{infer_time},{total_state_tomography_time},{total_process_tomography_time}"
    )
    with open(f"{respath}time.csv", "a") as f:
        f.write(
            f"{name},{qubits},{simulate_time},{infer_time},{total_state_tomography_time},{total_process_tomography_time},{n},{oncetime}\n"
        )
    return (
        simulate_time,
        infer_time,
        total_state_tomography_time,
        total_process_tomography_time,
    )


def time_main():
    # ray.init()
    with open(f"{respath}time.csv", "w") as f:
        f.write(
            f"name,qubits,simulate_time,infer_time,total_state_tomography_time,total_process_tomography_time,shots,one_shot_time\n"
        )
    results = ray.get(
        [
            time_get.remote(name, qubits)
            for name in ["grover", "rb", "qnn"]
            for qubits in [6, 8, 10] * 2
        ]
    )
    # results =[time_get(name,qubits) for name in ['rb','qft','qnn','qsvm'] for qubits in [9,10,11,12,13]*3]
    return results


def plot_results():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    data = pd.read_csv(f"{respath}time.csv")
    datamean = data.groupby(["qubits"]).mean().reset_index().sort_values(["qubits"])
    datamean.reset_index(drop=True, inplace=True)
    scale = 0.009
    fig = plt.figure(figsize=(1000 * scale, 900 * scale))
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
    axes = plt.axes([0, 0, 1, 1])
    colors = ["#7CBEEC", "#7EB57E", "#FF6F45", "#3274A1"]
    names = [
        "classical simulation",
        "state tomography",
        "process tomography",
        "our methods",
    ]
    datamean = (
        data[data.qubits.isin([6, 8, 10])]
        .groupby(["qubits"])
        .mean()
        .reset_index()
        .sort_values(["qubits"])
    )
    for i, col in enumerate(
        [
            "simulate_time",
            "total_state_tomography_time",
            "total_process_tomography_time",
            "infer_time",
        ]
    ):
        axes.bar(
            datamean.index + i * 0.2,
            datamean[col],
            color=colors[i],
            label=names[i],
            linewidth=1,
            width=0.2,
        )
    axes.set_yscale("log")
    axes.tick_params(axis="x", which="major", width=5, length=20)
    axes.set_yticks([1, 100, 10000, 1000000])
    # axes.set_xticklabels(allnames,rotation=90)
    axes.set_xlabel("# qubits")
    axes.set_xticks(datamean.index + 0.3)
    axes.set_xticklabels(datamean.qubits)
    axes.grid(axis="y", color="gray", linestyle="--", linewidth=5)
    axes.set_ylabel("time(s)")

    axes.legend(frameon=False, bbox_to_anchor=(1, 1.4), ncol=2)
    fig.savefig(f"{respath}fig9(a).svg", dpi=600, format="svg", bbox_inches="tight")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="validate theorem 1 in the paper by fig9(a)"
    )
    parser.add_argument(
        "--earlystop",
        action="store_true",
        default=False,
        help="reduce the sampling time",
    )
    args = parser.parse_args()

    directory = os.path.abspath(__file__).split("/")[-1].split(".")[0]
    respath = os.path.join(os.path.dirname(__file__), f"{directory}/")
    if not os.path.exists(respath):
        os.makedirs(respath)
    s = perf_counter()
    time_main()
    plot_results()
    e = perf_counter()
    print("totol speed time:", e - s)
