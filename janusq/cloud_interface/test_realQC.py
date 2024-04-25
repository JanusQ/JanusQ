import sys
sys.path.append('..')
import time
import sys
sys.path.append('../..')
from janusq.data_objects.circuit import Circuit
import json
import requests
import numpy as np
import time
runUrl = "http://janusq.zju.edu.cn/api1/circuit/runTutorial"
resultUrl = "http://janusq.zju.edu.cn/api1/circuit/result"

def submit(circuit: Circuit=None, label=None, shots=None, chip=None, run_type="simulator", API_TOKEN=None):
    '''
    Run circuits on the janusq backend

    Args:
        circuits: the quantum program in the self-defined circuit type, introduced as a set of quantum gates scheduled in 'qubit' and 'layer'
        run_options (kwargs): additional backend run options
    Returns:
        result id of runtime job. 
    Raises:
        TypeError: If the input parameters has a type mismatch with the run_options or the input circuit is not in the allowed type.
        ValueError: input parameters out of range.
    '''

    # circuit: the input circuit must be transpiled with the native gate sets and arranged in the allowed circuit type described as 'qubit' and 'layer'.
    if circuit is not None:
        if not isinstance(circuit, Circuit):
            raise Exception("Not allowed cirucit type.")

    # shots: the number of shots called in the measurement operation. It should be an integer variable and usually set to 3000 by default.
    if shots is None:
        shots = 3000
    if not isinstance(shots, int):
        raise Exception("shots type must be int")
    
    # run_type: janusq offers a simulator and real quantum devices for quantum circuit runtime. The backend configurations for the simulator and real quantum chips are set properly beforehand.
    if run_type not in ['simulator', 'sqcg']:
        raise Exception("run type must be simulate or sqcg. simulate: simulator runs circuit. sqcg: run circuit with real 0c.")
    
    # chips: the real quantum chip to run the quantum circuit.
    if chip is None:
        chip ='default'
    
    # label: labels of pre-set quantum circuits. It will be 'None' for a new circuit submission.
    if label is None:
        label = 'ghz_state'
    data = {
        "circuit": circuit,
        "shots": shots,
        "run_type": run_type,
        "label": label,
        "is_async": True,
        "n_qubits": circuit.n_qubits if circuit is not None else 4
    }
    max_retries = 5
    responese = None
    for _ in range(max_retries):
        try:
            if API_TOKEN is None:
                responese = requests.post(runUrl+'WithoutToken', data=json.dumps(data)).json()
            else:
                header = {
                    "Authorization": "Bearer " + API_TOKEN,
                    "Content-Type": "application/json"
                }
                responese = requests.post(runUrl, data=json.dumps(data), headers=header).json()
        except requests.ConnectionError:
            continue
        return responese
    return []

def get_result(result_id: str, run_type: str, result_format="sample"):
    '''
    Get circuit result on the janusq backend

    Args:
        result_id: provided by submit function
        run_type: simulator or sqcg, same as submit.
    Returns:
        result of runtime job. 
    '''
    data = {
        "result_id": result_id,
        "type": run_type
    }
    responese = None
    while True:
        try:
            responese = requests.get(resultUrl, params=data).json()
        except requests.ConnectionError:
            if run_type == 'simulator':
                time.sleep(0.01)
            else:
                time.sleep(2)
            continue
        if run_type == 'simulator':
                time.sleep(0.01)
        else:
            time.sleep(2)
        if 'task_status' not in  responese['data']:
            break
    if result_format == 'probs':
        if run_type == 'simulator':
            sample = responese['data']['sample']
            probs = np.zeros(2 ** len(list(sample.keys())[0]))
            sample_count = sum(sample.values())
            for k, v in sample.items():
                probs[int(k, 2)] = v / sample_count
        else:
            probs = responese['data']['probs']
        return probs
    else:
        if run_type == 'simulator':
            sample = responese['data']['sample']
        else:
            sample = {}
            probs = responese['data']['probs']
            for idx, p in enumerate(probs):
                sample[bin(idx)[2:].zfill(np.log2(len(probs) + 1))] = int(p * 3000)
        return sample

for alg in range(1):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    res = submit(circuit=Circuit(qc), label=alg, shots=3000, run_type='simulator', API_TOKEN='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzE0MDI0OTUyLCJpYXQiOjE3MTQwMTQxNTIsImp0aSI6IjA2OThlZGZjOTI3MzQ1NjNiYzQ1MzRhYTgzZWRmMTY0IiwidXNlcl9pZCI6NjAsIm5hbWUiOiJhZG1pbkB6anUuZWR1LmNuIn0.ToNFTVOp-BCVUiIZJNGKceEczdhnQaed-9HsukBZ-cs')
    print(res)
    print(get_result(res['data']['result_id'], run_type='simulator'))