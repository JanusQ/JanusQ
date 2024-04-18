'''
Author: name/jxhhhh� 2071379252@qq.com
Date: 2024-04-17 06:05:51
LastEditors: name/jxhhhh� 2071379252@qq.com
LastEditTime: 2024-04-18 09:03:13
FilePath: /JanusQ/janusq/optimizations/mapping/mapping_ct.py
Description: 

Copyright (c) 2024 by name/jxhhhh� 2071379252@qq.com, All Rights Reserved. 
'''
import logging
import numpy as np
from janusq.analysis.fidelity_prediction import FidelityModel
from janusq.data_objects.backend import Backend

from janusq.data_objects.circuit import Circuit, qiskit_to_circuit
from qiskit import transpile

class Mapper():
    def __init__(self, fidelity_model: FidelityModel) -> None:
        self.candidates = []
        self.fidelity_model = fidelity_model

    def run(self, circuit: Circuit, backend: Backend, return_candidates = False) -> Circuit:
        '''
        description: 
        param {Circuit} circuit: circuit need to optimize
        param {Backend} backend: mapper optimize run base on specific backend 
        param {bool} return_candidates: weather to return candidates
        '''
        qiskit_circuit = circuit.to_qiskit()
        
        transpile_results = [
            qiskit_to_circuit(transpile(qiskit_circuit, basis_gates=backend.basis_gates,
                                    coupling_map=backend.coupling_map, optimization_level=0,
                                    routing_method='sabre'))
            for _ in range(20)
        ]
        
        fidelities = [
            self.fidelity_model.predict_circuit_fidelity(circuit)
            for circuit in transpile_results
        ]
        
        max_index = np.argmax(fidelities)
        
        logging.info('before selection:', sum(fidelities)/len(fidelities), 'after selection:', max(fidelities))
        
        if not return_candidates:
            return transpile_results[max_index]
        else:
            return transpile_results[max_index], transpile_results
        
        # TODO: 返回mapping的顺序之类的
        
        
