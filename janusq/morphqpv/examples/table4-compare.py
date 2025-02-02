import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from data.Qbenchmark import layer_circuit_generator
from morphQPV.execute_engine.excute import (
    ExcuteEngine,
    qiskit_circuit_to_layer_cirucit,
    convert_state_to_density,
)
from morphQPV.execute_engine.metric import fidelity
import numpy as np
from copy import deepcopy
from morphQPV.baselines import proj, stat, ndd
from qiskit.quantum_info import random_clifford

## ignore warning
import ray
from time import perf_counter
import warnings

warnings.filterwarnings("ignore")
np.random.seed(0)


def gate_delete_generator(right_layer_circuit, delete_gate_num=1):
    bad_layer_circuit = deepcopy(right_layer_circuit)
    for i in range(delete_gate_num):
        delete_gate_layer_idx = np.random.randint(0, len(bad_layer_circuit))
        delete_gate_layer = bad_layer_circuit[delete_gate_layer_idx]
        ## random choose a gate in this layer
        bad_layer_circuit[delete_gate_layer_idx].pop(
            np.random.randint(0, len(delete_gate_layer))
        )
        if len(delete_gate_layer) == 0:
            bad_layer_circuit.pop(delete_gate_layer_idx)
    return bad_layer_circuit


def gate_add_generator(right_layer_circuit, n_qubits, _gate_num=1):
    bad_layer_circuit = deepcopy(right_layer_circuit)
    for i in range(_gate_num):
        _layer_idx = np.random.randint(0, len(bad_layer_circuit))
        _layer = bad_layer_circuit[_layer_idx]
        ## random choose a gate in this layer
        bad_layer_circuit[_layer_idx].insert(
            np.random.randint(0, len(_layer)),
            {"name": "z", "qubits": [np.random.choice(n_qubits, size=1)[0]]},
        )

    return bad_layer_circuit


def standard_state_prepare(n_qubits):
    gates = [
        [
            {
                "name": "x",
                "qubits": list(
                    np.random.choice(
                        n_qubits, size=np.random.randint(0, n_qubits), replace=False
                    )
                ),
            }
        ]
    ]
    # gates = [[{'name':'x','qubits':list(np.random.choice(n_qubits-1,size=1,replace=False))}]]
    return gates


def clliford_prepare(n_qubits):
    gates = random_clifford(n_qubits).to_circuit()
    gates = qiskit_circuit_to_layer_cirucit(gates)
    return gates


def standard_test(
    assertionfunc, state_prepare, right_circuit, bad_circuit, n_qubits, input_num=10
):
    for j in range(input_num):
        heads = state_prepare(n_qubits)
        real_output_state = ExcuteEngine.excute_on_pennylane(
            heads + right_circuit, type="statevector"
        )
        verify_right, spend_gates = assertionfunc(
            heads, right_circuit, real_output_state, n_qubits
        )
        if not verify_right:
            return 0, spend_gates
        is_right, spend_gates = assertionfunc(
            heads, bad_circuit, real_output_state, n_qubits
        )
        if not is_right:
            return 1, spend_gates
    return 0, spend_gates


def standard_mean_test(
    assertionfunc,
    state_prepare,
    right_circuit,
    bad_circuit,
    n_qubits,
    total_round=10,
    input_num=3,
):
    res = np.mean(
        np.array(
            [
                standard_test(
                    assertionfunc,
                    state_prepare,
                    right_circuit,
                    bad_circuit,
                    n_qubits,
                    input_num,
                )
                for i in range(total_round)
            ]
        ),
        axis=0,
    )
    return res


def proj_assertion(heads, layer_circuit, real_output_state, n_qubits):
    ## proj test
    proj_circuit, fake_gates = proj.assertion(
        real_output_state, list(range(n_qubits))
    )
    circuit_obj = ExcuteEngine(fake_gates)
    proj_gates = circuit_obj.gates_num
    proj_measurements = ExcuteEngine.excute_on_pennylane(
        heads + layer_circuit + proj_circuit, type="distribution", shots=10000
    )

    return abs(1 - proj_measurements[0]) < 1e-2, proj_gates * 5


def ndd_assertion(heads, layer_circuit, real_output_state, n_qubits):
    ## proj test
    proj_circuit, fake_gates, ancliqubit = ndd.assertion(
        real_output_state, list(range(n_qubits))
    )
    circuit_obj = ExcuteEngine(fake_gates)
    proj_gates = circuit_obj.gates_num
    proj_measurements = ExcuteEngine.excute_on_pennylane(
        heads + layer_circuit + proj_circuit,
        type="distribution",
        shots=1000,
        output_qubits=ancliqubit,
    )
    return abs(1 - proj_measurements[0]) < 1e-2, proj_gates * 5


def morph_assertion(heads, layer_circuit, real_output_state, n_qubits):
    output_state = ExcuteEngine.excute_on_pennylane(
        heads + layer_circuit, type="density"
    )
    fid = fidelity(convert_state_to_density(real_output_state), output_state)
    return fid > 0.98, ExcuteEngine(heads).gates_num * 5


def quito_assertion(heads, layer_circuit, real_output_state, n_qubits):
    real_distribution = np.abs(real_output_state) ** 2
    stat_shots = stat.assertion(real_output_state)
    stat_measurements = ExcuteEngine.excute_on_pennylane(
        heads + layer_circuit, type="distribution", shots=stat_shots
    )
    stat_verify = stat.chi_square_test(
        stat_measurements * stat_shots, real_distribution * stat_shots
    )
    return stat_verify, 5


def stat_assertion(heads, layer_circuit, real_output_state, n_qubits):
    real_distribution = np.abs(real_output_state) ** 2
    stat_shots = stat.assertion(real_output_state)
    stat_measurements = ExcuteEngine.excute_on_pennylane(
        heads + layer_circuit, type="distribution", shots=stat_shots
    )
    stat_verify = stat.chi_square_test(
        stat_measurements * stat_shots, real_distribution * stat_shots
    )
    return stat_verify, 5


def test_assertion(name, n_qubits, assertionfunc=proj_assertion):
    right_layer_circuit = layer_circuit_generator(name, n_qubits)
    heads = standard_state_prepare(n_qubits)
    if assertionfunc(
        heads,
        right_layer_circuit,
        ExcuteEngine.excute_on_pennylane(
            heads + right_layer_circuit, type="statevector"
        ),
        n_qubits,
    )[0]:
        print("test pass")
    else:
        print("test fail")


@ray.remote
def profilling_methods(name, n_qubits, earlystop=True):
    right_layer_circuit = layer_circuit_generator(name, n_qubits)
    bad_layer_circuit = gate_add_generator(right_layer_circuit, n_qubits)
    minimal_gates_num = ExcuteEngine(bad_layer_circuit).gates_num
    quito_confidence, quito_gates = standard_mean_test(
        quito_assertion,
        standard_state_prepare,
        right_layer_circuit,
        bad_layer_circuit,
        n_qubits,
        total_round=50
    )
    print(f"quito confidence {quito_confidence}  gates num {quito_gates}")
    ## ndd test
    if name == "qnn":
        ndd_confidence = "/"
        ndd_gates = "/"
    else:
        if earlystop and n_qubits > 5:
            ndd_confidence = 100
            ndd_gates = "over $10^5$"
        else:
            ndd_confidence, ndd_gates = standard_mean_test(
                ndd_assertion,
                standard_state_prepare,
                right_layer_circuit,
                bad_layer_circuit,
                n_qubits,
                total_round=5
            )
            ndd_confidence *= 100
    print(f"ndd confidence {ndd_confidence}  gates num {ndd_gates}")
    ## our test
    morph_confidence, morph_gates = standard_mean_test(
        morph_assertion,
        clliford_prepare,
        right_layer_circuit,
        bad_layer_circuit,
        n_qubits,
    )
    print(f"morph confidence {morph_confidence}  gates num {morph_gates}")
    with open(f"{resultspath}overhead.csv", "a") as f:
        f.write(
            f"{name},{n_qubits},{quito_confidence*100},{ndd_confidence},{morph_confidence*100},{quito_gates},{ndd_gates},{morph_gates}\n"
        )


if __name__ == "__main__":
    directory = os.path.abspath(__file__).split("/")[-1].split(".")[0]
    resultspath = os.path.join(os.path.dirname(__file__), f"{directory}/")
    if not os.path.exists(resultspath):
        os.mkdir(resultspath)
    with open(f"{resultspath}overhead.csv", "w") as f:
        f.write(
            "name,n_qubits,quito_confidence,ndd_confidence,morph_confidence,quito_gates_num,ndd_gates_num,morph_gates_num\n"
        )
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
    args = parser.parse_args()
    s = perf_counter()
    ray.get(
        [
            profilling_methods.remote(algo, qubit, earlystop=args.earlystop)
            for algo in ["qnn", "qec", "shor", "xeb"]
            for qubit in range(3, 10, 2)
        ]
    )
    e = perf_counter()
    print("take time:", e - s)
