import numpy as np
from itertools import combinations
from qiskit import QuantumCircuit
from typing import Dict
from .hdi_decompose import driver_component
from qiskit import transpile
from chocoq.utils.linear_system import get_circ_unitary
from ..provider import Provider

def obj_compnt(qc: QuantumCircuit, param, obj_dct: Dict):
    """https://quantumcomputing.stackexchange.com/questions/5567/circuit-construction-for-hamiltonian-simulation"""

    for pow, terms in obj_dct.items():
        for vars_tuple, coeff in terms:
            for k in range(1, pow + 1):
                fianl_theta = (1 / 2) ** pow * 2 * coeff * (-1) ** k
                for combo in combinations(range(pow), k):
                    for i in range(len(combo) - 1):
                        qc.cx(vars_tuple[combo[i]], vars_tuple[combo[i + 1]])
                    qc.rz(fianl_theta * param, vars_tuple[combo[-1]])
                    for i in range(len(combo) - 2, -1, -1):
                        qc.cx(vars_tuple[combo[i]], vars_tuple[combo[i + 1]])

def penalty_decompose(qc:QuantumCircuit, penalty_m: list, param, num_qubits):
    for penalty_mi in penalty_m:
        coeff = np.sum(penalty_mi[:-1]) / 2 - penalty_mi[-1]
        for i in range(num_qubits):
            qc.rz(-2 * coeff * penalty_mi[i] * param, i)
        for i in range(num_qubits - 1):
            for j in range(i + 1, num_qubits):
                if penalty_mi[i] != 0 and penalty_mi[j] != 0:
                    qc.cx(i, j)
                    coeff = penalty_mi[i] * penalty_mi[j]
                    qc.rz(coeff * param, j)
                    qc.cx(i, j)

def search_evolution_space_low_cost(qc: QuantumCircuit, params, Hd_bitstr_list, anc_idx, mcx_mode, num_qubits, shots, provider:Provider):
    num_basis_list = []
    set_basis_list = []
    depth_list = []
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit.quantum_info import Statevector

    CORE_BASIS_GATES = ["measure", "cx", "id", "rz", "sx", "x"]
    generate_preset_pass_manager(optimization_level=2, basis_gates=CORE_BASIS_GATES,)
    
    for i,hdi_vct in enumerate(Hd_bitstr_list):
        nonzero_indices = np.nonzero(hdi_vct)[0].tolist()
        hdi_bitstr = [0 if x == -1 else 1 for x in hdi_vct if x != 0]
        driver_component(qc, nonzero_indices, anc_idx, hdi_bitstr, params[i], mcx_mode)
        qc_cp:QuantumCircuit = qc.copy()
        qc_cp.measure(range(num_qubits), range(num_qubits)[::-1])
        qc_cp = provider.transpile(qc_cp)
        counts = provider.get_counts_with_time(qc_cp, shots=shots)
        num_basis_list.append(len(counts))
        set_basis_list.append(set(counts.keys()))
        depth_list.append(qc_cp.depth())
    return num_basis_list, set_basis_list, depth_list

# def search_evolution_space_low_cost(qc: QuantumCircuit, param, Hd_bitstr_list, anc_idx, mcx_mode, num_qubits, shots, provider:Provider):
#     num_basis_list = []
#     set_basis_list = []
#     depth_list = []
#     from qiskit_aer import AerSimulator
#     from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
#     from qiskit_ibm_runtime import SamplerV2 as Sampler
#     from qiskit.quantum_info import Statevector

#     CORE_BASIS_GATES = ["measure", "cx", "id", "rz", "sx", "x"]
#     generate_preset_pass_manager(optimization_level=2, basis_gates=CORE_BASIS_GATES,)
    
#     for hdi_vct in Hd_bitstr_list:
#         nonzero_indices = np.nonzero(hdi_vct)[0].tolist()
#         hdi_bitstr = [0 if x == -1 else 1 for x in hdi_vct if x != 0]
#         driver_component(qc, nonzero_indices, anc_idx, hdi_bitstr, param, mcx_mode)
#         qc_cp:QuantumCircuit = qc.copy()
#         qc_cp = transpile(qc_cp, provider.backend)
#         counts = provider.get_counts_with_time(qc_cp, shots=shots)
#         num_basis_list.append(len(counts))
#         set_basis_list.append(set(counts.keys()))
#         depth_list.append(qc_cp.depth())
#     return num_basis_list, set_basis_list, depth_list

def search_evolution_space(qc: QuantumCircuit, params, transpiled_hlist, anc_idx, mcx_mode, num_qubits, shots, provider:Provider):
    num_basis_list = []
    set_basis_list = []
    depth_list = []
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit.quantum_info import Statevector

    CORE_BASIS_GATES = ["measure", "cx", "id", "rz", "sx", "x"]
    generate_preset_pass_manager(optimization_level=2, basis_gates=CORE_BASIS_GATES,)
    for i, transpiled_h in enumerate(transpiled_hlist):
        transpiled_h = transpiled_h.assign_parameters([params[i]])
        qc.compose(transpiled_h, inplace=True)
        qc_cp:QuantumCircuit = qc.copy()
        qc_cp.measure(range(num_qubits), range(num_qubits)[::-1])
        counts = provider.get_counts_with_time(qc_cp, shots=shots)
        num_basis_list.append(len(counts))
        set_basis_list.append(set(counts.keys()))
        depth_list.append(qc_cp.depth())
    return num_basis_list, set_basis_list, depth_list


        # statevector = Statevector.from_instruction(qc_cp)
        # probabilities = statevector.probabilities()
        # print(np.count_nonzero(probabilities))

def commute_compnt(qc: QuantumCircuit, param, Hd_bitstr_list, anc_idx, mcx_mode):
    for hdi_vct in Hd_bitstr_list:
        nonzero_indices = np.nonzero(hdi_vct)[0].tolist()
        hdi_bitstr = [0 if x == -1 else 1 for x in hdi_vct if x != 0]
        driver_component(qc, nonzero_indices, anc_idx, hdi_bitstr, param, mcx_mode)

def new_compnt(qc: QuantumCircuit, params, Hd_bitstr_list, anc_idx, mcx_mode):
    for idx, hdi_vct in enumerate(Hd_bitstr_list):
        nonzero_indices = np.nonzero(hdi_vct)[0].tolist()
        hdi_bitstr = [0 if x == -1 else 1 for x in hdi_vct if x != 0]
        # qc.h(hdi_bitstr[0])
        driver_component(qc, nonzero_indices, anc_idx, hdi_bitstr, params[idx], mcx_mode)
        # qc.h(hdi_bitstr[0])

def new_compnt_with_measure(qc: QuantumCircuit, params, Hd_bitstr_list, anc_idx, mcx_mode, num_qubits):
    for idx, hdi_vct in enumerate(Hd_bitstr_list):
        nonzero_indices = np.nonzero(hdi_vct)[0].tolist()
        hdi_bitstr = [0 if x == -1 else 1 for x in hdi_vct if x != 0]
        # qc.h(hdi_bitstr[0])
        driver_component(qc, nonzero_indices, anc_idx, hdi_bitstr, params[idx], mcx_mode)
        qc.measure(range(num_qubits), range(num_qubits)[::-1])
        # qc.h(hdi_bitstr[0])

def new_x_compnt(qc: QuantumCircuit, params, Hd_bitstr_list):
    for idx, hdi_vct in enumerate(Hd_bitstr_list):
        nonzero_indices = np.nonzero(hdi_vct)[0].tolist()
        qc.barrier()
        for bit in nonzero_indices:
            qc.rx(params[idx], bit)
        # break
    # print(qc.draw())
        # exit()
        # hdi_bitstr = [0 if x == -1 else 1 for x in hdi_vct if x != 0]
        # driver_component(qc, nonzero_indices, anc_idx, hdi_bitstr, params[idx], mcx_mode)

def cyclic_compnt(qc: QuantumCircuit, param, constr_cyclic):
    for constr in constr_cyclic:
        nzlist = np.nonzero(constr[:-1])[0]
        if len(nzlist) < 2:
            continue
        for i in range(len(nzlist)):
            j = (i + 1) % len(nzlist)
            ## gate for X_iX_j
            qc.h(nzlist[j])
            qc.h(nzlist[i])
            qc.cx(nzlist[i], nzlist[j])
            qc.rz(2 * param, nzlist[j])
            qc.cx(nzlist[i], nzlist[j])
            qc.h(nzlist[j])
            qc.h(nzlist[i])
            ## gate for Y_iY_j
            qc.u(np.pi / 2, np.pi / 2, np.pi / 2, nzlist[j])
            qc.u(np.pi / 2, np.pi / 2, np.pi / 2, nzlist[i])
            qc.cx(nzlist[i], nzlist[j])
            qc.rz(2 * param, nzlist[j])
            qc.cx(nzlist[i], nzlist[j])
            qc.u(np.pi / 2, np.pi / 2, np.pi / 2, nzlist[j])
            qc.u(np.pi / 2, np.pi / 2, np.pi / 2, nzlist[i])
