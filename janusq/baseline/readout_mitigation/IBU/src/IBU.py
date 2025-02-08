from .IBUFull import IBUFull
from .IBUReduced import IBUReduced
import numpy as np
from typing import List, Union


def IBU(mats_true: List[np.ndarray], params: dict,
        mem_constrained: bool = False) -> Union[IBUFull, IBUReduced]:
    """
    :param mats_true: a list of 2x2 conditional probability tables representing
                      error probabilities for each qubit
    :param params: a dict specifying the following:
                - exp_name: str, name of experiment
                - num_qubits: int, number of qubits
                - method: str, "full" or "reduced"
                - library: str, "jax" or "tensorflow" or "numpy"
                - use_log: bool, whether to use log-space (numerical precision)
                - verbose: bool, verbosity of status updates
    :param mem_constrained: bool: True/False, for IBU Reduced ONLY; uses a
                            memory efficient implementation
    :return: object of IBUFull, or IBUReduced, depending on method specified in
             params dict.
    """
    if params['method'] == 'full':
        return IBUFull(mats_true, params)
    elif params['method'] == 'reduced':
        return IBUReduced(mats_true, params, mem_constrained)
    else:
        raise "Unsupported method!"
