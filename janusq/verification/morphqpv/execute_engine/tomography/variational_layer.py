import pennylane as qml
import numpy as np
import itertools
def single_layer_MPS( parameters, wires ):
    qml.RX( parameters[0], wires = wires[0] )
    qml.CNOT( wires = wires)
    qml.RX( parameters[1], wires = wires[0] )
    qml.RX( parameters[2], wires = wires[1] )

def Variational_circuit_MPS(params, wires):
    n_wires = len(wires)
    qml.RX( params[0], wires = n_wires-1 )
    params = params[:-1].reshape(n_wires-2,3)
    for k in reversed(range(1,n_wires-1)) :
        single_layer_MPS( params[k-1] , [k,k+1] )

def Variational_state(params, wires):
    assert len(params) == 2**(len(wires)+1)-2
    qml.ArbitraryStatePreparation( params, wires = wires )

def RealAmplitude_Ry_layer(parms,wires):
    assert len(parms) == len(wires)
    for i in range(len(wires)):
        qml.RY(parms[i],wires=wires[i])

def RealAmplitude_cx_layer(wires,entanglement):
    if entanglement == 'full':
        entanglement = list(itertools.combinations(wires,2))
    elif entanglement == 'linear':
        entanglement = [(i,i+1) for i in range(len(wires)-1)]
    elif entanglement == 'circular':
        entanglement = [(i,(i+1)%len(wires)) for i in range(len(wires))]
    for i,j in entanglement:
        qml.CNOT(wires=[i,j])

def RealAmplitudes(parms,wires,entanglement='full',reps=None):
    if reps is None:
        reps = len(parms)//len(wires) - 1
    parms = np.array(parms).reshape(reps+1,len(wires))
    for i in range(reps):
        RealAmplitude_Ry_layer(parms[i],wires)
        RealAmplitude_cx_layer(wires,entanglement)
    RealAmplitude_Ry_layer(parms[-1],wires)
    


