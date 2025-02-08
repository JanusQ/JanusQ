import logging
import pickle 
import os
import json
import traceback
import numpy as np


dirname = os.path.dirname(__file__)



try:
    with open(os.path.join(dirname, 'fidelity_dataset_5q.pkl'), "rb") as f:
        real_qc_5bit = pickle.load(f)
    real_qc_5bit = (real_qc_5bit[0][:500], real_qc_5bit[1][:500])
except:
    traceback.print_exc()

try:
    with open(os.path.join(dirname, 'benchmark_circuits.json'), "r") as f:
        benchmark_circuits_and_results = json.load(f)  
except:
    traceback.print_exc()

try:
    with open(os.path.join(dirname, 'ghz_8qubit.json'), "r") as f:
        ghz_8qubit = json.load(f) 
except:
    traceback.print_exc()

try:
    with open(os.path.join(dirname, 'ghz_error.json'), "r") as f:
        ghz_error = json.load(f)  
except:
    traceback.print_exc()

try:
    with open(os.path.join(dirname, 'protocol_8.json'), "r") as f:
        protocol_8 = json.load(f)   
except:
    traceback.print_exc()

try:
    with open(os.path.join(dirname, 'ghz_8qubit.json'), "r") as f:
        ghz_8qubit = json.load(f) 
except:
    traceback.print_exc()

try:
    with open(os.path.join(dirname, 'ghz_error.json'), "r") as f:
        ghz_error = json.load(f)  
except:
    traceback.print_exc()

try:
    with open(os.path.join(dirname, 'protocol_8.json'), "r") as f:
        protocol_8 = json.load(f) 
        protocol_8 = (np.array(protocol_8[0]),protocol_8[1])
except:
    traceback.print_exc()
    
try:
    with open(os.path.join(dirname, 'matrices_ibu.json'), "r") as f:
        matrices_ibu = json.load(f)   
except:
    traceback.print_exc()
    
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):

        renamed_module = module
        if module.startswith("data_objects"):
            renamed_module = "janusq.objects" + module.split("data_objects")[1]

        return super(RenameUnpickler, self).find_class(renamed_module, name)

def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()

with open(os.path.join(dirname, "fidelity_dataset_18q.pkl"), "rb") as f:
    real_qc_18bit = renamed_load(f)
