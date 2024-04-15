# Import general libraries (needed for functions)
import numpy as np
# Import the RB Functions
import sys
import os
from pathlib import Path
# sys.path.append(str(Path(os.getcwd())))

import qiskit.ignis.verification.randomized_benchmarking as rb

from qiskit import execute, transpile, Aer
import ray
from janusq.simulator.noisy_simulator import NoisySimulator
from janusq.data_objects.circuit import  Circuit, qiskit_to_circuit
from janusq.tools.ray_func import wait, map

class RBModel():
    
    def __init__(self, simulator: NoisySimulator, multi_process = True):
        backend = simulator.vec_model.backend
        qubits = backend.involvod_qubits
        couplers = list(backend.coupling_map)

        qubit_errors = map(get_error_1q, qubits, multi_process = multi_process, show_progress= True, simulator = simulator)
        coupler_errors = map(get_error_2q, couplers, multi_process = multi_process, show_progress= True, error_1qs = qubit_errors, simulator = simulator)

        # may raise ValueError: `x0` is infeasible.
        self.qubits = qubits
        self.couplers = couplers
        self.qubit_errors =  [error['u3'] for error in qubit_errors]
        self.coupler_errors = coupler_errors


    def predict_circuit_fidelity(self, circuit: Circuit):
        fidelity = 1
        for layer in circuit:
            for gate in layer:
                if len(gate['qubits']) == 1:
                    fidelity *= 1-self.qubit_errors[gate['qubits'][0]]
                else:
                    fidelity  *= 1- self.coupler_errors[self.couplers.index(list(gate['qubits']))]
        return fidelity
    
    
    
    @staticmethod
    def get_rb_fidelity(circuit, single_average_error_rb, couple_average_error_rb):
        fidelity = 1
        for gate in circuit.gates:
            from  janusq.analysis.vectorization import extract_device
            device = extract_device(gate)
            if isinstance(device,tuple):
                fidelity = fidelity * couple_average_error_rb[device]
            else:
                fidelity = fidelity * single_average_error_rb[device]
        return fidelity 
















def run_rb(simulator: NoisySimulator, rb_pattern, rb_circs, xdata, target_qubits):
    # if upstream_model is not None:
    #     assert upstream_model.backend == backend
    backend = simulator.backend
    basis_gates = backend.basis_gates #['u1', 'u2', 'u3', 'cx']
    transpiled_circs_list = []
    
    rb_fit = rb.RBFitter(None, xdata, rb_pattern)
    shots = 500
    
    jobs = []
    for rb_index, rb_circ in enumerate(rb_circs):
        # 其实是一组电路, 应该是同一个电路的不同长度
        
        fit_rb_circ = transpile(rb_circ, basis_gates=['u2', 'u3', 'cx'])  # ibm 只能有u1, u2, u3和 cx 垃圾玩意
        real_rb_circ = transpile(rb_circ, basis_gates=basis_gates)  # 实际执行的电路

        error_rb_circ = []
        for index, elm in enumerate(real_rb_circ):
            elm, n_error = simulator._inject_context_error(qiskit_to_circuit(elm))
            elm = simulator.to_qiskit(elm)
            elm2 = fit_rb_circ[index]
            
            elm.name = elm2.name
            new_creg = elm._create_creg(len(target_qubits), "cr")
            elm.add_register(new_creg)
            for cbit, qubit in enumerate(target_qubits):
                elm.barrier()
                elm.measure(cbit, cbit)
            
            error_rb_circ.append(elm)


        transpiled_circs_list.append(fit_rb_circ)
        
        '''TODO: 看下很多比特的时候的时间'''
        qasm_simulator = Aer.get_backend('qasm_simulator')
        job = execute(error_rb_circ, qasm_simulator, noise_model =simulator.get_noise_model(target_qubits) , basis_gates=basis_gates, shots= shots, optimization_level=0)
        jobs.append(job)


        # Add data to the fitter
    rb_fit.add_data([job.result() for job in jobs])


    gpc = rb.rb_utils.gates_per_clifford(
        transpiled_circuits_list=transpiled_circs_list,
        clifford_lengths=xdata[0],
        basis=['u2', 'u3', 'cx'],
        qubits=target_qubits)

    epc = rb_fit.fit[0]['epc']
    return gpc, epc

def get_error_1q(target_qubit, simulator, length_range = [20, 1500]):
    # 先搞单比特的rb
    rb_pattern = [[target_qubit]]
    target_qubits = [target_qubit]
    rb_circs, xdata = rb.randomized_benchmarking_seq(
        rb_pattern=rb_pattern, nseeds=5, length_vector=np.arange(length_range[0], length_range[1], (length_range[1] - length_range[0]) //10 ), )  # seed 越多跑的越久

    # print(rb_circs[0][-1])

    gpc, epc = run_rb(simulator, rb_pattern, rb_circs, xdata, target_qubits)

    # calculate 1Q EPGs
    epg = rb.calculate_1q_epg(gate_per_cliff=gpc, epc_1q=epc, qubit=target_qubit)

    return epg  # epg['u3'] 作为单比特门误差    #sum(epg.values()) / len(epg)


# TODO: 现在的噪音太小了
def get_error_2q(target_qubits, error_1qs,  simulator, length_range = [20, 600]):
    error_1qs = [error_1qs[qubit] for qubit in target_qubits]
    assert len(target_qubits) == 2  and len(error_1qs) == 2
    
    # 先搞单比特的rb
    rb_pattern = [target_qubits]
    target_qubits = target_qubits
    
    # 这个好像特别慢
    rb_circs, xdata = rb.randomized_benchmarking_seq(
        rb_pattern=rb_pattern, nseeds=5, length_vector=np.arange(length_range[0], length_range[1], (length_range[1] - length_range[0]) //10), )  # seed 越多跑的越久

    # print(rb_circs[0][-1])

    gpc, epc = run_rb(simulator, rb_pattern, rb_circs, xdata, target_qubits)

    # calculate 1Q EPGs
    epg = rb.calculate_2q_epg(
        gate_per_cliff=gpc,
        epc_2q=epc,
        qubit_pair=target_qubits,
        list_epgs_1q=error_1qs)
    
    return epg

@ray.remote
def get_error_2q_remote(target_qubits, error_1qs, simulator):
    return get_error_2q(target_qubits, error_1qs, simulator)

@ray.remote
def get_error_1q_remote(qubit, simulator):
    return get_error_1q(qubit, simulator)


    