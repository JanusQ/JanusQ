

import logging
import pickle 
import os



dirname = os.path.dirname(__file__)





with open(os.path.join(dirname, 'fidelity_dataset_5q.pkl'), "rb") as f:
    real_qc_5bit = pickle.load(f)   # TODO: 整理成circuits一个数组，fidelity一个数组的形式


with open(os.path.join(dirname, 'benchmark_circuits.pickle'), "rb") as f:
    benchmark_circuits = pickle.load(f)  

with open(os.path.join(dirname, 'ghz_8qubit.pickle'), "rb") as f:
    ghz_8qubit = pickle.load(f) 

with open(os.path.join(dirname, 'ghz_error.pickle'), "rb") as f:
    ghz_error = pickle.load(f)  

with open(os.path.join(dirname, 'protocol_8.pickle'), "rb") as f:
    protocol_8 = pickle.load(f)   

with open(os.path.join(dirname, 'matrices_ibu.pickle'), "rb") as f:
    matrices_ibu = pickle.load(f)   

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):

        renamed_module = module
        if module.startswith("data_objects"):# or module.startwith("analysis") or module.startwith("data_objects")
            renamed_module = "janusq." + module

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


with open(os.path.join(dirname, "fidelity_dataset_18q.pkl"), "rb") as f:
    real_qc_18bit = renamed_load(f)

