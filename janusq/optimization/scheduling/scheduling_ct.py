'''
Author: name/jxhhhh� 2071379252@qq.com
Date: 2024-04-17 06:06:07
LastEditors: name/jxhhhh� 2071379252@qq.com
LastEditTime: 2024-04-18 09:07:26
FilePath: /JanusQ/janusq/optimizations/scheduling/scheduling_ct.py
Description: 

Copyright (c) 2024 by name/jxhhhh� 2071379252@qq.com, All Rights Reserved. 
'''
import copy
import random
import numpy as np
from janusq.analysis.fidelity_prediction import FidelityModel

from janusq.objects.circuit import Circuit
import time




class Scheduler():
    def __init__(self, fidelity_model: FidelityModel) -> None:
        self.candidates = []
        self.fidelity_model = fidelity_model

    def run(self, target_circuit: Circuit, timeout = 60):
        '''
        description: use method of local search to optimize circuit
        param {Circuit} target_circuit: target_circuit
        param {int} timeout: break if out of time
        '''
        # 采用local search的方式
        
        # for 10个最好的candidate的
            # for 5 step
                # 每次随机选一个门 （fidelity低的或者随机的）
                # 移动记录candidate
            # 再挑10个最好的candidate
        explore_ratio = .3
        topk = 10
            
        fidelity_model = self.fidelity_model
        candidates: list[Circuit] = [target_circuit]
        fidelities = np.array([fidelity_model.predict_circuit_fidelity(target_circuit)])
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # find topk candidates
            fidelities_indices = np.argsort(-fidelities)
            fidelities_indices = list(fidelities_indices[:topk])
            if len(fidelities_indices[topk:]) > 0:
                fidelities_indices += random.choices(fidelities_indices[topk:], k = min([topk//2, len(fidelities_indices[topk:])]))
            candidates =  [candidates[index] for index in fidelities_indices]
            fidelities = [fidelities[index] for index in fidelities_indices]
            
            # print('fidelities:', fidelities)
            
            new_candidates: list[Circuit] = []
            new_fidelities = []
            for candidate in candidates:
                if random.random() < explore_ratio:
                    moving_gate = random.choice(candidate.gates)
                else:
                    gate_fidelities = fidelity_model.predict_gate_fidelities(candidate)
                    error_gates = [candidate.gates[index] for index in np.argsort(-gate_fidelities)[:10]]
                    moving_gate = random.choice(error_gates)
                    
                moving_space = candidate.get_available_space(moving_gate)
                if len(moving_space) == 1:
                    continue
                for new_layer in moving_space:
                    if new_layer == moving_gate.layer_index:
                        continue
                    new_circuit = candidate.move(moving_gate, new_layer)
                    moving_fidelity = fidelity_model.predict_circuit_fidelity(new_circuit)
                    new_candidates.append(new_circuit)
                    new_fidelities.append(moving_fidelity)
                    
            candidates += new_candidates
            fidelities += new_fidelities
            fidelities = np.array(fidelities)
        
        return candidates[np.argmax(fidelities)]
                
                    
                    
            
    
    
        
def scheduling(circuit: Circuit, fidelity_model: FidelityModel, gate_error_threshold) -> Circuit:
    new_circuit = []      

    cur_layer = [0 for i in range(circuit.n_gates)]
    
    pre_fidelity = 1
    id = 0
    cnt = 0
    
    while True:
        gate_fidelities = fidelity_model.predict_gate_fidelities(new_circuit)
        
        
        pass
    
    for layer in circuit:
        for gate in layer:
            gate = copy.deepcopy(gate)
            new_circuit['gates'].append(gate)
            new_circuit['gate2layer'].append(-1)
            gate['id'] = id
            id += 1
            qubits = gate['qubits']
            
            offset = 0
            while True:
                if len(qubits) == 1:
                    qubit = qubits[0]
                    insert_layer = cur_layer[qubit] + offset
                    new_circuit['gate2layer'][-1] = insert_layer
                    if insert_layer >= len(new_circuit['layer2gates']):
                        assert insert_layer == len(new_circuit['layer2gates'])
                        new_circuit['layer2gates'].append([gate])
                    else:
                        new_circuit['layer2gates'][insert_layer].append(gate)
                        
                else:
                    qubit0 = qubits[0]
                    qubit1 = qubits[1]
                    insert_layer = max(cur_layer[qubit0],cur_layer[qubit1]) + offset
                    new_circuit['gate2layer'][-1] = insert_layer
                    if insert_layer >= len(new_circuit['layer2gates']):
                        assert insert_layer == len(new_circuit['layer2gates'])
                        new_circuit['layer2gates'].append([gate])
                    else:
                        new_circuit['layer2gates'][insert_layer].append(gate)
                
                gate_fidelities = fidelity_model.predict_gate_fidelities(new_circuit)
                gate_fidelities = gate_fidelities if gate_fidelities < 1 else 1
                
                if offset > 5 or pre_fidelity - gate_fidelities < threshold:
                    if offset > 5:
                        logging.warning('threshold too small')
                    pre_fidelity = gate_fidelities
                    if len(qubits) == 1:
                        cur_layer[qubit] = insert_layer + 1
                    else:
                        cur_layer[qubit0] = insert_layer+ 1
                        cur_layer[qubit1] = insert_layer+ 1
                    break
                else:
                    new_circuit['layer2gates'][insert_layer].remove(gate)
                    offset += 1
                    cnt += 1
    

    logging.info(new_circuit['id'], 'predict:', fidelity_model.predict_fidelity(circuit_info), '--->', gate_fidelities)
    logging.info(cnt)
    return new_circuit
