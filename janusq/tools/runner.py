import sys
sys.path.append('../..')
from janusq.data_objects.circuit import Circuit
import json
import time
import requests

runUrl = "http://janusq.zju.edu.cn/api1/circuit/runTutorial"

def run4tutorial(circuit: Circuit=None, label=None, shots=None, chip=None, run_type="simulate", API_TOKEN=None):
    if circuit is not None:
        if not isinstance(circuit, Circuit):
            raise Exception("Not allowed cirucit type.")
    if shots is None:
        shots = 3000
    if not isinstance(shots, int):
        raise Exception("shots type must be int")
    if run_type not in ['simulate', 'sqcg']:
        raise Exception("run type must be simulate or sqcg. simulate: simulator runs circuit. sqcg: run circuit with real 0c.")
    if chip is None:
        chip ='default'
    if label is None:
        label = 'ghz_state'
    data = {
        "circuit": "123456",
        "shots": shots,
        "run_type": "sqcg",
        "label": label
    }
    responese = requests.post(runUrl, data=json.dumps(data))
    print(responese.json())
    return 1
for i in range(30):
    for label in ['w_state', 'ghz_state', 'VQA', 'time_crystal']:
        try:
            print(run4tutorial(label=label))
        except requests.exceptions.ConnectionError:
            print('休眠一下')
            time.sleep(600)