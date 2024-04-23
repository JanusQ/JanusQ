import sys
sys.path.append('../..')
from janusq.data_objects.circuit import Circuit
import json
import requests
import numpy as np
runUrl = "http://janusq.zju.edu.cn/api1/circuit/runTutorial"

def submit(circuit: Circuit=None, label=None, shots=None, chip=None, run_type="simulator", API_TOKEN=None):
    '''
    Run circuits on the janusq backend

    Args:
        circuits: the quantum program in the self-defined circuit type, introduced as a set of quantum gates scheduled in 'qubit' and 'layer'
        run_options (kwargs): additional backend run options
    Returns:
        results of runtime job. For general tasks, it returns the probability distribution of the post quantum circuit. For VQA(QNN) problems, it returns the classification prediction probability and the original grayscale image.
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
        "label": label
    }
    max_retries = 5
    responese = None
    for _ in range(max_retries):
        try:
            # API token: runurl of the api token.
            if API_TOKEN is None:
                responese = requests.post(runUrl+'WithoutToken', data=json.dumps(data)).json()
            else:
                responese = requests.post(runUrl, data=json.dumps(data)).json()
        except requests.ConnectionError:
            continue
        if run_type == 'simulator':
            sample = responese['data']['result']['sample']
            probs = np.zeros(2 ** circuit.n_qubits)
            sample_count = sum(sample.values())
            for k, v in sample.items():
                probs[int(k, 2)] = v / sample_count
        else:
            probs = responese['data']['result']['probs']
        return probs
    return []