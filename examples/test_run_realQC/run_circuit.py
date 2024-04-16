import json
import pickle
import random
import threading
import time
import ray
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import hellinger_fidelity
import sys
sys.path.append('.')
from circuit_operation import parse_circuit, convert_circuit


def one_dimension_chain_coupling_map(qubit_num: int) -> List[List[int]]:
    return [[i, i + 1] for i in range(qubit_num - 1)] + [[i + 1, i] for i in range(qubit_num - 1)]


_SQCG_SERVICE_BASE = 'http://janusq.zju.edu.cn/api1' # 这里应该要改
_SQCG_SERVICE_RUN = _SQCG_SERVICE_BASE + 'run4tutorial'
_SQCG_SERVICE_CHIP_INFO = _SQCG_SERVICE_BASE + 'chip_info'
BASIC_GATE_SET = ['h', 'rx', 'ry', 'rz', 'cz']

def transpile_circuit(circuit: QuantumCircuit, level=3, coupling_map=None):
    if coupling_map is None:
        coupling_map = one_dimension_chain_coupling_map(circuit.num_qubits)
    compiled_qc = transpile(circuit, basis_gates=BASIC_GATE_SET,
                            coupling_map=coupling_map,
                            optimization_level=level)
    return compiled_qc


def pre_process(circuit, chip_name, level=0, devide=False, require_decoupling=False, insert_probs=1.0):
    print('run start parallel ..')
    import warnings
    warnings.filterwarnings("ignore")
    start_transpile_time = time.time()
    compiled_qc = transpile_circuit(circuit, level=level)
    start_optimization_time = time.time()
    circuit_info = parse_circuit(compiled_qc, devide=devide, require_decoupling=require_decoupling, insert_probs=insert_probs)
    # print(circuit_info)
    start_request_time = time.time()
    front_time = {
        'start transpile time': start_transpile_time,
        'start optimize time': start_optimization_time,
        'start request time': start_request_time
    }
    req_data = [{
        "seq": convert_circuit(circuit_info, chip_name),
        "stats": 3000,
        "chip_name": chip_name
    }]
    return req_data, front_time, compiled_qc


@ray.remote
def pre_process_remote(circuit, chip_name, devide=False, require_decoupling=False, level=0, insert_probs=1.0):
    return pre_process(circuit, chip_name, level=level, devide=devide, require_decoupling=require_decoupling, insert_probs=insert_probs)


def handle_time(front_time, back_time):
    return {
        'transpile': front_time['start optimize time'] - front_time['start transpile time'],
        'optimize': front_time['start request time'] - front_time['start optimize time'],
        'old_handle': back_time['enter process time'] - front_time['start request time'],
        'handle': back_time['enter process time'] - back_time['get request time'],
        'enter_process': back_time['backend start time'] - back_time['enter process time'],
        'pulse_construct': back_time['finish envelope construction'] - back_time['backend start time'],
        'pre_processing1': back_time['get runQ time'] - back_time['finish envelope construction'],
        'pre_processing2': back_time['arrive run time'] - back_time['get runQ time'],
        'in_queue': back_time['run start time'] - back_time['arrive run time'],
        'transpile_run': back_time['run end time'] - back_time['run start time'],
        'post_processing': back_time['end post-processing time'] - back_time['start post-processing time'],
        'all_time': back_time['end post-processing time'] - front_time['start transpile time']
    }


def send_request(req_data):
    run_data = {}
    try:
        print('start request .....')
        resp = requests.post(_SQCG_SERVICE_RUN, data=json.dumps(req_data))
        resp_data = resp.json()
        run_data = resp_data['data']
        resp.close()
    except Exception as e:
        return {}
    return run_data

count = 0
lock = threading.Lock()

def post_process(front_time, run_data, return_keys=None, **kwargs):
    global count, lock
    if return_keys is None:
        return_keys = ['times', 'counts', 'extra_data']
    if 'times' in run_data and 'times' in return_keys:
        back_time = run_data['times']
        all_time = handle_time(front_time, back_time)
    else:
        back_time = {}
        all_time = {}
        return {}
    print('run end parallel....')
    post_process_data = {
        'times': {
            'front time': front_time,
            'back time': back_time,
            'all time': all_time,
        },
        'counts': {
            'no readout calibration': run_data['bitstring_cnt_raw'],
            'readout calibration': run_data['probs'],
        },
        'extra_data': kwargs,
    }
    return_data = {key: post_process_data[key] for key in return_keys}
    return return_data


def run_circuit(circuit=None, data=None, chip_name='N36U19', devide=True, require_decoupling=True, level=0, return_keys=None,
                need_ray=False, insert_probs=1):
    if circuit is None:
        circuit = data['circuit']
    if data is not None:
        arg_id = data['id']
    else:
        arg_id = ''
    if need_ray:
        future = pre_process_remote.remote(circuit, chip_name, devide, require_decoupling, level, insert_probs)
        r_future = ray.get(future)
        req_data = r_future[0]
        front_time = r_future[1]
        compiled_qc = r_future[2]
    else:
        req_data, front_time, compiled_qc = pre_process(circuit, chip_name, level=level, devide=devide, require_decoupling=require_decoupling, insert_probs=insert_probs)

    run_data = send_request(req_data)

    return post_process(front_time, run_data, return_keys,
                        id=arg_id, qubit_number=compiled_qc.num_qubits, compiled_qc=compiled_qc)

def save_run_data(res):
    global count, lock
    result = res.result()
    with lock:
        count += 1
        with open(f'result_n/{count}.json', mode='w') as f:
            json.dump(result, f)

def run_circuits(dataset, mode='serial', max_workers=50, virtualization=False, return_keys=None, devide=False, decoupling=False, level=0, time_delay=None, insert_probs=1):
    v_flag = True
    result_list = []

    def _get_chipname(n):
        nonlocal v_flag
        if virtualization:
            if n <= 5:
                v_flag = not v_flag
                if v_flag:
                    return 'N36U19_0'
                else:
                    return 'N36U19_1'
        return 'N36U19'

    if mode == 'serial':
        for data in dataset:
            circuit = data['circuit']
            chipname = _get_chipname(circuit.num_qubits)
            res = run_circuit(data=data, chip_name=chipname, need_ray=False, return_keys=return_keys, insert_probs=insert_probs)
            result_list.append(res)
    elif mode == 'parallel':
        res_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            if time_delay is None:
                for data in dataset:
                    circuit = data['circuit']
                    chipname = _get_chipname(circuit.num_qubits)
                    # pool.submit(run_circuit, data=data, chip_name=chipname, need_ray=True, return_keys=return_keys, devide=devide, require_decoupling=decoupling, level=level).add_done_callback(save_run_data)
                    res = pool.submit(run_circuit, data=data, chip_name=chipname, need_ray=True, return_keys=return_keys, devide=devide, require_decoupling=decoupling, level=level, insert_probs=insert_probs)
                    res_list.append(res)
            else:
                for data in dataset:
                    circuit = data['circuit']
                    chipname = _get_chipname(circuit.num_qubits)
                    res = pool.submit(run_circuit, data=data, chip_name=chipname, need_ray=True, return_keys=return_keys, devide=devide, require_decoupling=decoupling, level=level, insert_probs=insert_probs)
                    res_list.append(res)
                    time.sleep(time_delay)
        for r in as_completed(res_list):
            result = r.result()
            result_list.append(result)
    return result_list


def get_ideal_dist(circuit, reverse=True):
    result_sim = Statevector(circuit).probabilities_dict()
    if reverse:
        c_dict = {}
        for k, v in result_sim.items():
            k = k[::-1]
            c_dict[k] = v
        return c_dict
    else:
        return result_sim


def get_scores(dist_p, dist_q):
    return hellinger_fidelity(dist_p, dist_q)

# 各个数量级的电路运行 虚拟化耗时和不虚拟化的耗时

# 各种负载情况下，单个电路运行的甘特图

# 不同比例时间占比

# 各种准确度

# 与IBM



if __name__ == '__main__':
    from circuit.InternetComputing.timeline.get_timeline_result import random_circuit
    # qc = random_circuit(5, 50)
    # pre_process(qc, 'N36U19', level=3)
    # import pickle
    # with open('timeline/c5.random', mode='rb') as f:
    #     c5 = pickle.load(f)
    # with open('timeline/c10.random', mode='rb') as f:
    #     c10 = pickle.load(f)
    # run_circuit(c5, devide=False, require_decoupling=False, return_keys=['times'])
    #circuit = None, data = None, chip_name = 'N36U19', devide = True, require_decoupling = True, level = 0, return_keys = None,need_ray = False
    # from circuit.InternetComputing.dataset.dataset1.grover import get_cir
    # qc = transpile_circuit(get_cir(5), level=3)
    # print(qc)
    # print(parse_circuit(qc, devide=False, require_decoupling=True, insert_probs=1)['qiskit_circuit'])
    with open('supermarq/circuits7', mode='rb') as f:
        cir = pickle.load(f)
    print(cir)
    cir.pop(9)
    cir.pop(9)
    print(cir)
    result = run_circuits(cir, level=3)
    print(result)
    with open('iz.pkl', mode='wb') as f:
        pickle.dump(result, f)











    # run_circuit(chip_name='N36U19', devide=False, require_decoupling=False, level=0, return_keys=None,
    #             need_ray=False)
    # exp_main(20)

    # with open('circuit.cpkl', mode='rb') as f:
    #     dataset = pickle.load(f)
    #
    # print(len(dataset))
    # print(dataset)
    # return_keys = ['extra_data', 'counts']
    # result1 = run_circuits(dataset[11:12], mode='parallel', virtualization=True, devide=True, decoupling=True, level=3)
    # with open('resultg.pkl', mode='wb') as f:
    #     pickle.dump(result1, f)
    # result2 = run_circuits(dataset[11:12], mode='parallel', virtualization=True, devide=True, decoupling=False, level=3)
    # with open('resultg.pkl', mode='wb') as f:
    #     pickle.dump(result2, f)
    # result3 = run_circuits(dataset[11:12], mode='parallel', virtualization=True, devide=False, decoupling=False, level=3)
    # with open('resultg.pkl', mode='wb') as f:
    #     pickle.dump(result3, f)

    # with open('result3.pkl', mode='rb') as f:
    #     dataset = pickle.load(f)
    # with open('circuit.cpkl', mode='rb') as f:
    #     circuits = pickle.load(f)
    # print(dataset)
    # results = []
    # for data in dataset:
    #     result = {}
    #     result['circuit'] = data['extra_data']['compiled_qc']
    #     result['id'] = data['extra_data']['id']
    #     result['probs'] = data['counts']['readout calibration']
    #     result['nprobs'] = data['counts']['no readout calibration']
    #     results.append(result)
    # for i, result in enumerate(results):
    #     print('*' * 20)
    #     print(result['id'])
    #     d1 = get_ideal_dist(result['circuit'])
    #     print(get_scores(d1, result['probs']))
    #     d1 = get_ideal_dist(result['circuit'], reverse=False)
    #     print(get_scores(d1, result['probs']))
    #     d1 = get_ideal_dist(circuits[i]['circuit'])
    #     print(get_scores(d1, result['probs']))
    #     d1 = get_ideal_dist(result['circuit'])
    #     print(get_scores(d1, result['nprobs']))
    #     d1 = get_ideal_dist(result['circuit'], reverse=False)
    #     print(get_scores(d1, result['nprobs']))
    #     d1 = get_ideal_dist(circuits[i]['circuit'])
    #     print(get_scores(d1, result['nprobs']))