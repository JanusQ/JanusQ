
## MorphQPV
language: English | [中文](doc/README.zh-CN.md)

MorphQPV is a tool to facilitate confident assertion-based verification in quantum computing. It provides a framework for analyzing and verifying quantum circuits using a new type of formalism. It defines an assertion statement that consists of assume-guarantee primitives and tracepoint pragma to label the target quantum state. Then, we can characterize the ground-truth relation between states using isomorphism-based approximation, which can effectively get the program states under various inputs while avoiding repeated executions. Finally, the verification is formulated as a constraint optimization problem with a confidence estimation model to enable rigorous analysis. 
## Getting Started
### Installation
```bash
conda create -n morphenv python=3.9
conda activate morphenv
python -m pip install -r requirements.txt
```
If the installation fails, please try to install the required packages with no version limit.
```bash
python -m pip install -r requirementswithnoversion.txt
```
### Example
Here is an example of using MorphQPV to verify a quantum circuit:

```python
from morphqpv import MorphQC
from morphqpv import IsPure,Equal,NotEqual
import numpy as np
with MorphQC() as morphQC:
    ### morphQC is a quantum circuit, the gate is applyed to the qubits in the order of the list
    ## we can add tracepoint to label the quantum state
    morphQC.add_tracepoint(0,1,2) ## the state after the first 3 qubits is labeled as tracepoint 0
    morphQC.assume(0,IsPure()) ## the state in tracepoint 0 is assumed to be pure
    morphQC.x([1,3]) ## apply x gate to  qubit 1 and 3
    morphQC.y([0,1,2])  ## apply y gate to qubit 0,1,2
    for i in range(4):
        morphQC.cnot([i, i+1]) ## apply cnot gate to qubit i and i+1
    morphQC.s([0,2,4]) ## apply s gate to qubit 0,2,4
    morphQC.add_tracepoint(2,4) ## the state after qubit 2 and 4 is labeled as tracepoint 1
    morphQC.assume(1,IsPure())  ## the state in tracepoint 1 is assumed to be pure
    morphQC.rz([0,1,2,3,4],np.pi/3) ## apply rz gate to qubit 0,1,2,3,4
    morphQC.h([0,1,2,3,4]) ## apply h gate to qubit 0,1,2,3,4
    morphQC.rx([0,1,2,3,4],np.pi/3) ## apply rx(pi/3) gate to qubit 0,1,2,3,4
    morphQC.ry([0,1,2,3,4],np.pi/3) ## apply ry(pi/3) gate to qubit 0,1,2,3,4
    morphQC.add_tracepoint(0,3) ## the state after qubit 0 and 3 is labeled as tracepoint 2
    morphQC.assume(2,IsPure()) ## the state in tracepoint 2 is assumed to be pure
    morphQC.guarantee([1,2],Equal()) ## the state in tracepoint 1 and 2 are guaranteed to be equal
print(morphQC.assertion) ## print the assertion statement and verify result
```
The script file `main_exp.py` provides a more complex example. There are hyper-parameters in each stage of the verification. Users can check the details in [config](doc/morphconfig.md).
## Artifact Evaluation
The detailed evaluation can be found in the [evaluation](doc/evaluation.md) file.
## License
This project is licensed under the GNU GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details
