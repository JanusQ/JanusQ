'''生成用于测量M的电路'''
import logging
import random
import numpy as np
from qiskit import QuantumCircuit

from janusq.data_objects.backend import Backend
from janusq.data_objects.circuit import Circuit

from janusq.optimizations.readout_mitigation.fem.tools import all_bitstrings, decimal, expand, statuscnt_to_npformat
from janusq.tools.ray_func import map

from janusq.simulator.noisy_simulator import NoisySimulator
from janusq.simulator.readout_error_model import ReadoutErrorModel

def gen_benchmarking_circuit(real: np.ndarray) -> Circuit:              
    measured_qubits = [
        qubit
        for qubit, measurement in enumerate(real)
        if '2' != measurement
    ]
    
    layer = []
    for qubit, bit  in enumerate(real):
        if bit == '1' or bit == 1:
            layer.append({
                'name': 'rx',
                'qubits': [qubit],
                'params': [np.pi]
            })
    n_qubits = len(real)
    
    return Circuit([layer], n_qubits, measured_qubits = measured_qubits, operated_qubits=list(range(n_qubits)))

                 
class EnumeratedProtocol():
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        
    def gen_circuits(self) -> tuple[list[str], list[QuantumCircuit]]:
        circuits = []
        
        reals = []
        for real in all_bitstrings(self.n_qubits, base = 3):
            if all(real ==  2): continue  # 没有测量
            

            real_rev = real[::-1]
            circuit = gen_benchmarking_circuit(real_rev)
            circuits.append(circuit)
            
            reals.append(real)
            
        return np.array(reals, dtype=np.int8), circuits



'''TODO: 还跑不了'''
class IterativeSamplingProtocol():
    def __init__(self, backend: Backend, hyper = 1, n_samples_iter = 1, threshold = 1e-3):
        '''TODO: hyper是啥'''
        self.backend = backend
        self.n_qubits = backend.n_qubits
        self.hyper = hyper
        self.n_samples_iter = n_samples_iter
        self.threshold = threshold
        self.cnt = self.hyper * self.n_qubits


        self.benchmarking_results: list[dict] = None
        
        self.executed_reals = []
        self.executed_statuscntss= []
    
    @property
    def all_results(self): return self.ideals, self.all_results
    
    def get_iter_protocol_results(self, machine_data, cnt, filter = None ):
        
        if filter is None:
            iter_res = machine_data[:cnt]
            machine_data =  machine_data[cnt:]
        else:
            iter_res, new_machine_data = [], []
            for ele in machine_data:
                real_bitstring, status_count = ele 
                if real_bitstring[filter[0]] != filter[2] or real_bitstring[filter[1]] != filter[3]:
                    new_machine_data.append(ele)
                else:
                    iter_res.append(ele)
            new_machine_data += iter_res[cnt:]
            iter_res = iter_res[:cnt]
            machine_data = new_machine_data
        return iter_res, machine_data

    def gen_random_circuits(self, cnt, bitstring_dataset: list,  filter = None)-> tuple[np.ndarray, Circuit]:
        n_qubits = self.n_qubits
            
        reals, circuits = [], []
    
        i = 0
        while i < cnt:
            if len(bitstring_dataset) == 3**n_qubits - 1:
                break
            
            max = 3**self.n_qubits
            value = random.randint(0, max - 1)
            bitstring = decimal(value, 'str', base = 3)
            bitstring = '0' * (n_qubits - len(bitstring)) + bitstring
            
            if bitstring == '2' * self.n_qubits or bitstring in bitstring_dataset:
                continue  # 没有测量
            if filter is not None:
                if bitstring[filter[0]] != filter[2] or bitstring[filter[1]] != filter[3]:
                    continue
            i += 1
            reals.append(np.array(list(bitstring)).astype(np.int8))
            bitstring_dataset.append(bitstring)
                            
            measured_qubits = [
                qubit
                for qubit, measurement in enumerate(bitstring)
                if '2' != measurement
            ]
            
            layer = []
            for qubit, bit  in enumerate(bitstring):
                if bit == '1':
                    layer.append({
                        'name': 'rx',
                        'qubits': [qubit],
                        'params': [np.pi]
                    })
            circuit = Circuit([layer], n_qubits, measured_qubits = measured_qubits, operated_qubits=list(range(n_qubits)))
            circuits.append(circuit)        
        
        return np.array(reals, dtype=np.int8), circuits
    


    # def gen_circuits(self):
    #     n_qubits = self.n_qubits
    #     threshold = self.threshold
    #     n_samples_iter = self.n_samples_iter

    #     bitstring_dataset =  []
        
    #     qubit_errors = np.zeros(shape=n_qubits)
    #     qubit_count = 0
    #     states_error = np.zeros((n_qubits, n_qubits, 2, 3))
    #     states_count = np.zeros((n_qubits, n_qubits, 2, 3))
    #     states_datasize = np.zeros((n_qubits, n_qubits, 2, 3))

    #     filter = None
        
    #     kth_max = 0
    #     while True:
    #         reals, protocol_circuits = self.gen_random_circuits(self.cnt, bitstring_dataset, filter = filter)
            
    #         if len(reals) == 0:
    #             break
            
    #         # yield reals, protocol_circuits
    #         yield  protocol_circuits
            
    #         # benchmarking_statuscntss = self.benchmarking_results


    #         benchmarking_statuscntss = [
    #             self.simulator.execute(circuit, 1000)
    #             for circuit in protocol_circuits
    #         ]


    #         format_benchmarking_results = []
    #         for (bitstrings, probs), real in zip(self.benchmarking_results, reals):
    #             measured_qubits = [qubit for qubit in range(n_qubits) if real[qubit] != 2]
    #             bitstrings = np.array([expand(bitstring, measured_qubits, n_qubits) for bitstring in bitstrings], dtype=np.int8)
    #             format_benchmarking_results.append((bitstrings, probs))
            
    #         benchmarking_statuscntss = format_benchmarking_results
            
    #         self.executed_statuscntss += benchmarking_statuscntss
    #         if len(self.executed_reals) == 0:
    #             self.executed_reals = reals
    #         else:
    #             self.executed_reals = np.concatenate([self.executed_reals, reals], dtype=np.int8)
    #         # print(len(self.executed_statuscntss), len(self.executed_reals))
            
    #         for real, (meas_np, cnt_np) in zip(reals, benchmarking_statuscntss):
    #             qubit_count += np.sum(cnt_np)  # 固定值
                
    #             for q0 in range(n_qubits):
    #                 if real[q0] == 2:
    #                      continue
    #                 for q1 in range(n_qubits):
    #                     states_count[q0][q1][real[q0]][real[q1]] += qubit_count
    #                     states_datasize[q0][q1][real[q0]][real[q1]] += 1

    #                 error_index = meas_np[:,q0] != real[q0]
    #                 error_cnt_np = cnt_np[error_index]
                    
    #                 total_error_cnt_np = np.sum(error_cnt_np)
    #                 for q1 in range(n_qubits):
    #                     states_error[q0][q1][real[q0]][real[q1]] += total_error_cnt_np
                    
                
    #         iter_qubit_errors = qubit_errors / qubit_count               
    #         iter_states_error = states_error / states_count

    #         for qubit in range(n_qubits):
    #             iter_states_error[qubit] -= iter_qubit_errors[qubit]
                
    #         eq6 = np.abs(iter_states_error)/states_datasize
                
    #         if np.nanmax(eq6) < threshold:
    #             break
            
    #         nan_count = np.sum(np.isnan(eq6))
    #         filter = np.argsort(eq6, axis = None )[-1-nan_count-kth_max]
    #         filter = np.unravel_index(filter, eq6.shape)


    def get_data(self, machine_data):

        hyper = self.hyper
        threshold = self.threshold
        n_qubits = self.n_qubits
    
        bitstring_dataset =  []
        protocol_results_dataset = []

        cnt = hyper * n_qubits
        
        qubit_errors = np.zeros(shape=n_qubits)
        qubit_count = 0
        states_error = np.zeros((n_qubits, n_qubits, 2, 3))
        states_count = np.zeros((n_qubits, n_qubits, 2, 3))
        states_datasize = np.zeros((n_qubits, n_qubits, 2, 3))

        filter = None
        
        step = 1
        kth_max = 0
        while True:
            
            if machine_data is not None:
                protocol_results, machine_data = self.get_iter_protocol_results(machine_data, 1000 if step == 0 else cnt, filter = filter)

                while len(protocol_results) == 0:
                    if len(machine_data) == 0:
                        logging.warning(f'当 threshold = ', threshold, '被薅空了')
                        return protocol_results_dataset
                    kth_max += 1

                    nan_count = np.sum(np.isnan(eq6))
                    filter = np.argsort(eq6,  axis = None )[-1-nan_count-kth_max]
                    filter = np.unravel_index(filter, eq6.shape)
                    logging.info('new filter: ', filter)
                    protocol_results, machine_data = self.get_iter_protocol_results(machine_data, cnt, filter = filter)

            protocol_results_dataset += protocol_results
            
            # for real_bitstring, status_count in protocol_results:
            for ele in protocol_results:
                real_bitstring, status_count  = ele
                meas_np, cnt_np = status_count
                qubit_count += np.sum(cnt_np)  # 固定值
                

                for q0 in range(n_qubits):
                    if real_bitstring[q0] == 2:
                         continue
                    for q1 in range(n_qubits):
                        states_count[q0][q1][real_bitstring[q0]][real_bitstring[q1]] += qubit_count
                        states_datasize[q0][q1][real_bitstring[q0]][real_bitstring[q1]] += 1

                    error_index = meas_np[:,q0] != real_bitstring[q0]
                    # error_meas_np = meas_np[error_index]
                    error_cnt_np = cnt_np[error_index]
                    
                    total_error_cnt_np = np.sum(error_cnt_np)
                    for q1 in range(n_qubits):
                        states_error[q0][q1][real_bitstring[q0]][real_bitstring[q1]] += total_error_cnt_np
                    
                
                        
            
            iter_qubit_errors = qubit_errors / qubit_count               
            iter_states_error = states_error / states_count

            for qubit in range(n_qubits):
                iter_states_error[qubit] -= iter_qubit_errors[qubit]
                
            eq6 = np.abs(iter_states_error)/states_datasize
                
            if np.nanmax(eq6) < threshold:
                break
            
            nan_count = np.sum(np.isnan(eq6))
            filter = np.argsort(eq6, axis = None )[-1-nan_count-kth_max]
            filter = np.unravel_index(filter, eq6.shape)
        
        return  protocol_results_dataset

    
    def update(self, results):
        # ideal_ops: list[np.ndarray], execution_results: list[tuple[np.ndarray, np.ndarray]]
        self.benchmarking_results: list[tuple[np.ndarray, np.ndarray]] = map(statuscnt_to_npformat, results)
        # benchmarking_result_to_np_format(results)
        # 可以变成完全矩阵计算

    