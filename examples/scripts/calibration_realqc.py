"""
Perform readout calibration on the GHZ circuit


Parameters
----------
Mitigator: int,int
    qubits represents the number of qubit
    n_iters represents the number of iteration




Returns
----------

"""



"""Perform readout calibration on the GHZ circuit

Quick utility that wraps input validation,
``next(ShuffleSplit().split(X, y))``, and application to input data
into a single call for splitting (and optionally subsampling) data into a
one-liner.

Read more in the :ref:`User Guide <cross_validation>`.

Parameters
----------
*arrays : sequence of indexables with same length / shape[0]
    Allowed inputs are lists, numpy arrays, scipy-sparse
    matrices or pandas dataframes.

test_size : float or int, default=None
    If float, should be between 0.0 and 1.0 and represent the proportion
    of the dataset to include in the test split. If int, represents the
    absolute number of test samples. If None, the value is set to the
    complement of the train size. If ``train_size`` is also None, it will
    be set to 0.25.

train_size : float or int, default=None
    If float, should be between 0.0 and 1.0 and represent the
    proportion of the dataset to include in the train split. If
    int, represents the absolute number of train samples. If None,
    the value is automatically set to the complement of the test size.

random_state : int, RandomState instance or None, default=None
    Controls the shuffling applied to the data before applying the split.
    Pass an int for reproducible output across multiple function calls.
    See :term:`Glossary <random_state>`.

shuffle : bool, default=True
    Whether or not to shuffle the data before splitting. If shuffle=False
    then stratify must be None.

stratify : array-like, default=None
    If not None, data is split in a stratified fashion, using this as
    the class labels.
    Read more in the :ref:`User Guide <stratification>`.

Returns
-------
splitting : list, length=2 * len(arrays)
    List containing train-test split of inputs.

    .. versionadded:: 0.16
        If the input is sparse, the output will be a
        ``scipy.sparse.csr_matrix``. Else, output type is the same as the
        input type.

Examples
--------
>>> import numpy as np
>>> from sklearn.model_selection import train_test_split
>>> X, y = np.arange(10).reshape((5, 2)), range(5)
>>> X
array([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]])
>>> list(y)
[0, 1, 2, 3, 4]

>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.33, random_state=42)
...
>>> X_train
array([[4, 5],
        [0, 1],
        [6, 7]])
>>> y_train
[2, 0, 3]
>>> X_test
array([[2, 3],
        [8, 9]])
>>> y_test
[1, 4]

>>> train_test_split(y, shuffle=False)
[[0, 1, 2], [3, 4]]
"""



import sys, os
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
sys.path.append('..')

import logging
logging.basicConfig(level=logging.ERROR)
from janusq.optimizations.readout_mitigation.fem import Mitigator
from qiskit.quantum_info.analysis import hellinger_fidelity
from janusq.optimizations.readout_mitigation.fem.tools import npformat_to_statuscnt
from janusq.dataset import protocol_8,ghz_8qubit


benchmark_circuits_and_results = protocol_8
ghz_output = ghz_8qubit     


qubits = 8


mitigator = Mitigator(qubits, n_iters = 2)
scores = mitigator.init(benchmark_circuits_and_results, group_size = 2, partation_methods=[
                         'max-cut'],multi_process=False, draw_grouping = True)


n_qubits = 4
outout_ideal = {'1'*n_qubits:0.5,'0'*n_qubits:0.5}
output_fem = mitigator.mitigate(ghz_output[0],[i for i in range(n_qubits)], cho = 1 )
output_fem = npformat_to_statuscnt(output_fem)

print("Janus-FEM fidelity: ",hellinger_fidelity(outout_ideal,output_fem))

