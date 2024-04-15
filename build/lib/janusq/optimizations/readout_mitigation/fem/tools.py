from functools import lru_cache
import logging
import ray
import threading
import inspect
import uuid
import random
import concurrent.futures
from concurrent.futures._base import Future
from collections import defaultdict
import numpy as np


import numpy as np 
import pickle 
import tqdm
import ray

from tools.ray_func import wait, map

def decimal(value, convert_type, base = 2):
    if convert_type == 'str':
        str_value = ''
        while value != 0:
            str_value = str(value % base) + str_value
            value //= base
        return str_value
    elif convert_type == 'int':
        int_value = 0
        for bit_pos, bit_vlaue in enumerate(value):
            bit_pos = len(value) - bit_pos - 1
            int_value += int(bit_vlaue) * (base ** bit_pos)
        return int_value

    logging.exception('unkown convert_type', convert_type)

@lru_cache
def all_bitstrings(n_qubits, base = 2):
    all_bitstings = np.zeros((base**n_qubits, n_qubits), dtype=np.int8)
    for value in range(base**n_qubits):
        bitstring = decimal(value, 'str', base = base)
        bitstring = '0' * (n_qubits - len(bitstring)) + bitstring
        bitstring = np.array(list(bitstring)).astype(np.int8)
        all_bitstings[value] = bitstring
        
    return all_bitstings


def statuscnt_to_npformat(state_cnt):
    meas_list, cnt_list = [], []
    for i, (meas, cnt) in enumerate(state_cnt.items()):
        meas = np.array(list(meas)).astype(np.int8)
        meas_list.append(meas)
        cnt_list.append(cnt)

    meas_np = np.array(meas_list, dtype=np.int8)
    cnt_np = np.array(cnt_list, dtype=np.double)
    return [meas_np, cnt_np]

def npformat_to_statuscnt(np_format):
    bstrs, counts = np_format
    status_count = {}
    for bstr, count in zip(bstrs, counts):
        bstr = ''.join([str(elm) for elm in bstr])
        status_count[bstr] = count
    return status_count


@ray.remote
def status_count_to_np_format_remote(state_cnt):
    return statuscnt_to_npformat(state_cnt)

def benchmarking_result_to_np_format(protocol_results: dict, multi_process = False):
    # 转换成
    # real, [mea, count]
    #  的格式
    ideals = [
        np.array(list(ideal)).astype(np.int8)
        for ideal in protocol_results.keys()
    ]
    meas_cnts = map(statuscnt_to_npformat, list(protocol_results.values()), show_progress=True)
    
    return ideals, meas_cnts


def expand(bitstring: str|np.ndarray, measured_qubits, n_qubits):
    if isinstance(bitstring, str):
        new_bitstring = '2'*n_qubits
    else:
        new_bitstring = np.ones(n_qubits, dtype=np.int8) * 2
        
    for bit, qubit in zip(bitstring, measured_qubits):
        new_bitstring[qubit] = bit
        
    return new_bitstring

def to_int(bstr: np.ndarray, base = 2): 
    n_bstr = len(bstr)
    return int(np.sum(bstr * base**(n_bstr - np.arange(n_bstr)-1)))


def downsample_bitstring(bitstring, qubits):
    new_bitstring = ''.join([bitstring[qubit] for qubit in qubits])
    return new_bitstring

def downsample_status_count(stats_count: dict, qubits: list):
    new_stats_count = defaultdict(int)
    
    for bitstring, count in stats_count.items():
        new_bitstring = downsample_bitstring(bitstring, qubits)
        new_stats_count[new_bitstring] += count
    
    return dict(new_stats_count)

def downsample(protocol_result: dict[dict], qubits:list):
    new_protocol_result = {}
    
    for real_bitsting, status_count in protocol_result.items():
        new_real_bitsring = downsample_bitstring(real_bitsting,qubits)
        if new_real_bitsring == "2"*len(qubits):
            continue
        new_status_count = downsample_status_count(status_count,qubits)
        
        if new_real_bitsring not in new_protocol_result:
            new_protocol_result[new_real_bitsring] = new_status_count
        else:
            for measure_bitstring, count in new_status_count.items():
                if measure_bitstring in new_protocol_result[new_real_bitsring]:
                    new_protocol_result[new_real_bitsring][measure_bitstring] += count
                else:
                    new_protocol_result[new_real_bitsring][measure_bitstring] = count
    
    for real_bitsting, status_count in new_protocol_result.items():
        total_count = sum(status_count.values())
        for measure_bitstring, count in status_count.items():
            status_count[measure_bitstring] = count / total_count
        
    return new_protocol_result