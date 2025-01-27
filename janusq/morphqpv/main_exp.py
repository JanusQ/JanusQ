from morphQPV import MorphQC,Config
from morphQPV import IsPure,Equal,NotEqual
from morphQPV import StateVector,Expectation
from morphQPV import pauliX,pauliY,pauliZ,hadamard
import numpy as np
myconfig = Config()
myconfig.solver = 'sgd' ## set the stochastic gradient descent method to solve the assertion
with MorphQC(config=myconfig) as morphQC:
    ### morphQC is a quantum circuit, the gate is applyed to the qubits in the order of the list
    ## we can add tracepoint to label the quantum state
    morphQC.add_tracepoint(0,1) ## the state after the first 3 qubits is labeled as tracepoint 0
    morphQC.assume(0,IsPure()) ## the state in tracepoint 0 is assumed to be pure
    morphQC.assume(0,Equal(Expectation(pauliX@pauliY)),0.4)
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
    morphQC.assume([0,2],Equal(Expectation(pauliX@pauliY)),)
    morphQC.guarantee([1,2],Equal()) ## the state in tracepoint 1 and 2 are guaranteed to be equal
    morphQC.guarantee([0,1],NotEqual()) ## the state in tracepoint 0,1 and 2 are guaranteed to be different
print(morphQC.assertion) ## print the assertion statement and verify result
