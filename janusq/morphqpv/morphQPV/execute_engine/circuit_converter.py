import pennylane as qml

from qiskit import QuantumCircuit
import copy

from qiskit.circuit import ClassicalRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGNode,DAGOpNode
from math import pi
from collections.abc import Iterable
import numpy as np
def layer_circuit_to_qml_circuit(layer_cirucit: Iterable[Iterable[dict]]) -> qml.QNode:
    """ convert the layer_circuit to qml circuit

    Args:
        layer_cirucit (Iterable[Iterable[dict]]):  the layer_circuit to be converted
        Define the structure of the layer_circuit:
            1. layer_circuit is a list of layers
            2. each layer is a list of gates
            3. each gate is a dict with the following keys:
                'name': the name of the gate
                'qubits': the qubits the gate acts on
                'params': the parameters of the gate (if any)
                there are some notes for the gates:
                (1) If the gate is single qubits gate without params,like H,X,Y,Z, Measure, Tracepoint.
                the qubits is a list of qubit index, e.g. [0,1],when the gate acts on qubit 0 and 1.
                (2) If the gate is single qubits gate with params, the qubits is a list of qubit index, e.g. [0,1],when the gate acts on qubit 0 and 1. and if the params is a float, the qubits will be apply the gate with same param. if the params is a list, the qubits will be apply the gate with respective param.
                (3) If the gate is multi qubits gate, the qubits is a list of qubit index, e.g. [0,1], which means the gate acts on qubit 0 and 1, and the control qubit is the last qubit in the list.

    Returns:
        qml.QNode: the qml circuit
    Raises:
        Exception: Unkown gate type
    """
    gate_map = {
        'h': qml.Hadamard,
        'x': qml.PauliX,
        'y': qml.PauliY,
        'z': qml.PauliZ,
        's': qml.S,
        't': qml.T,
        'u3': qml.U3,
        'u2': qml.U2,
        'u1': qml.U1,
        'rx': qml.RX,
        'ry': qml.RY,
        'rz': qml.RZ,
        'cx': qml.CNOT,
        'cz': qml.CZ
    }
    for layer in layer_cirucit:
        for gate in layer:
            gate_qubits = gate['qubits']
            if gate['name'] in ['h', 'x', 'y', 'z', 's', 't']:
                if isinstance(gate_qubits, Iterable):
                    for q in gate_qubits:
                        gate_map[gate['name'].lower()](wires=q)
                else:
                    gate_map[gate['name'].lower()](wires=gate_qubits)
            elif gate['name'] == 'measure':
                qml.sample(qml.PauliZ(wires=gate_qubits))
            elif gate['name'] == 'sdg':
                if isinstance(gate_qubits, Iterable):
                    for q in gate_qubits:
                        qml.PhaseShift(-pi/2, wires=q)
                else:
                    qml.PhaseShift(-pi/2, wires=gate_qubits)
            
            elif gate['name'] == 'u3' or gate['name'] == 'u':
                theta, phi, lam = gate['params']  # TODO: 这个地方的参数位置需要检查一下
                qml.U3(theta, phi, lam, wires=gate_qubits)
            elif gate['name'] == 'u2':
                phi, lam = gate['params']
                qml.U2(phi, lam, wires=gate_qubits)
            elif gate['name'] == 'u1':
                phi = gate['params'][0]
                qml.U1(phi, wires=gate_qubits)
            elif gate['name']== 'flipkey':
                key = np.array([int(i) for i in list(gate['params'])])
                qml.ctrl(qml.FlipSign(key, wires=gate['ctrled_qubits']),control = gate['ctrl_qubits'][0])
            elif gate['name'] in ['rx', 'ry', 'rz']:
                if isinstance(gate['params'],Iterable):
                    if isinstance(gate_qubits, Iterable):
                        for i,q in enumerate(gate_qubits):
                            gate_map[gate['name'].lower()](gate['params'][i],wires=q)
                    else:
                        gate_map[gate['name'].lower()](gate['params'][0],wires=gate_qubits)
                else:
                    if isinstance(gate_qubits, Iterable):
                        for q in gate_qubits:
                            gate_map[gate['name'].lower()](gate['params'],wires=q)
                    else:
                        gate_map[gate['name'].lower()](gate['params'],wires=gate_qubits)

            elif gate['name'] == 'cx':
                qml.CNOT(wires=gate_qubits)
            elif gate['name'] == 'cz':
                qml.CZ(wires=gate_qubits)
            elif gate['name'] == 'cond_x':
                m_0 = qml.measure(gate_qubits[0])
                qml.cond(m_0, qml.PauliX)(wires=gate_qubits[1])
            elif gate['name'] == 'cond_z':
                m_0 = qml.measure(gate_qubits[0])
                qml.cond(m_0, qml.PauliZ)(wires=gate_qubits[1])
            elif gate['name'] == 'mcz':
                qml.MultiControlledZ(wires=gate_qubits[:-1], control_wires=gate_qubits[-1])
            elif gate['name'] == 'initialize':
                qml.QubitStateVector(gate['params'], wires=gate_qubits)
            elif gate['name'] == 'unitary':
                unitary = gate['params']
                qml.QubitUnitary(unitary, wires=gate_qubits)
            elif gate['name'] == 'wirecut' or gate['name'] == 'tracepoint':
                pass
            elif gate['name'] == 'ctrl':
                operation = gate['params']
                operation = qml.QubitUnitary(operation, wires=gate['ctrled_qubits'])
                qml.ctrl(operation, control=gate['ctrl_qubits'])
            elif gate['name'] == 'swap':
                qml.SWAP(wires=gate_qubits)
            elif gate['name'] == 'channel':
                channel = gate['params']
                qml.QubitChannel(channel, wires=gate_qubits)
            else:
                raise Exception('Unkown gate type', gate)
            
def layer_circuit_to_qiskit_circuit(layer_cirucit,N_qubit):
    qiskit_circuit = QuantumCircuit(N_qubit)
    for layer in layer_cirucit:
        for gate in layer:
            n_qubits = gate['qubits']
            if gate['name'] == 'u3' or gate['name'] == 'u':
                theta, phi, lam = gate['params']  # TODO: 这个地方的参数位置需要检查一下
                qiskit_circuit.u(theta, phi, lam, qubit=n_qubits)
            elif gate['name'] == 'u2':
                phi, lam = gate['params']
                qiskit_circuit.u(pi/2,phi, lam, qubit=n_qubits)
            elif gate['name'] == 'u1':
                lam = gate['params'][0]
                qiskit_circuit.p(lam, qubit=n_qubits)
            elif gate['name'] == 'h':
                qiskit_circuit.h(qubit=n_qubits)
            elif gate['name'] == 'x':
                qiskit_circuit.x(qubit=n_qubits)
            elif gate['name'] == 'y':
                qiskit_circuit.y(qubit=n_qubits)
            elif gate['name'] == 'z':
                qiskit_circuit.z(qubit=n_qubits)
            elif gate['name'] == 'sdg':
                qiskit_circuit.sdg(qubit=n_qubits)
            elif gate['name'] == 's':
                qiskit_circuit.s(qubit=n_qubits)
            elif gate['name'] == 'swap':
                qiskit_circuit.swap(qubit1=n_qubits[0], qubit2=n_qubits[1])
            elif gate['name'] == 'rz':
                qiskit_circuit.rz(gate['params'][0], qubit=n_qubits)
            elif gate['name'] == 'cond_x':
                classcial_register = ClassicalRegister(1)
                qiskit_circuit.add_register(classcial_register)
                qiskit_circuit.measure(n_qubits[0], classcial_register)
                qiskit_circuit.x(qubit=n_qubits[1]).c_if(classcial_register, 1)
            elif gate['name'] == 'cond_z':
                classcial_register = ClassicalRegister(1)
                qiskit_circuit.add_register(classcial_register)
                qiskit_circuit.measure(n_qubits[0], classcial_register)
                qiskit_circuit.z(qubit=n_qubits[1]).c_if(classcial_register, 1)
            elif gate['name'] == 'cx':
                qiskit_circuit.cx(control_qubit=n_qubits[0], target_qubit=n_qubits[1])
            elif gate['name'] == 'cz':
                qiskit_circuit.cz(control_qubit=n_qubits[0], target_qubit=n_qubits[1])
            elif gate['name'] == 'mcz':
                qiskit_circuit.mcp(gate['params'][0], control_qubits=n_qubits[:-1], target_qubit=n_qubits[-1])
            elif gate['name'] == 'ctrl':
                unitary = gate['params']
                operation_circuit = QuantumCircuit(len(gate['ctrled_qubits']))
                operation_circuit.unitary(unitary, qubits=list(range(len(gate['ctrled_qubits']))))
                custom_control = operation_circuit.to_gate().control(len(gate['ctrl_qubits']))
                qiskit_circuit.append(custom_control, gate['ctrl_qubits'] + gate['ctrled_qubits'])
            elif gate['name'] == 'initialize':
                qiskit_circuit.initialize(gate['params'],qubits= n_qubits)
            elif gate['name'] == 'unitary':
                unitary = gate['params']
                qiskit_circuit.unitary(unitary,qubits= n_qubits)
            elif gate['name'] == 'channel':
                channel = gate['params']
                qiskit_circuit.channel(channel, qubits= n_qubits)
            elif gate['name'] == 'measure':
                classcial_register = ClassicalRegister(len(n_qubits))
                qiskit_circuit.add_register(classcial_register)
                qiskit_circuit.measure(n_qubits, classcial_register)
            elif gate['name'] == 'cswap':
                qiskit_circuit.cswap(n_qubits[0], n_qubits[1], n_qubits[2])
            elif gate['name'] == 'tracepoint':
                pass
            else:
                raise Exception('Unkown gate type', gate)
    return qiskit_circuit

def qiskit_circuit_to_layer_cirucit(qiskit_circuit: QuantumCircuit) -> list:
    layer2qiskit_instructions = get_layered_instructions(qiskit_circuit)
    layer_circuit = []
    for layer_index, layer_instructions in enumerate(layer2qiskit_instructions):
        layer_instructions = [qiskit_to_layer_gate(
            instruction) for instruction in layer_instructions]
        layer_circuit.append(layer_instructions)

    return layer_circuit


def format_circuit(layer_cirucit):
    new_circuit = []

    id = 0
    for layer in layer_cirucit:
        layer = copy.deepcopy(layer)

        new_layer = []
        for gate in layer:
            gate['id'] = id
            gate['layer'] = len(new_circuit)
            new_layer.append(gate)
            id += 1
        new_circuit.append(new_layer)

        if layer[0]['name'] == 'breakpoint':
            assert len(layer) == 1  # 如果存在一个assertion, 就不能存在别的操作
            # new_circuit.append([])  # breakpoint 占两层，下一层用unitary gate来制备

    return new_circuit


def get_layered_instructions(circuit):
    dagcircuit, instructions, nodes = my_circuit_to_dag(circuit)
    # dagcircuit.draw(filename = 'dag.svg')
    graph_layers = dagcircuit.multigraph_layers()

    layer2operations = []  # Remove input and output nodes
    for layer in graph_layers:
        layer = [node for node in layer if isinstance(
            node, DAGOpNode) and node.op.name not in ('barrier', 'measure')]
        if len(layer) != 0:
            layer2operations.append(layer)

    layer2instructions = []
    instruction2layer = [None] * len(nodes)
    for layer, operations in enumerate(layer2operations):  # 层号，该层操作
        layer_instructions = []
        for node in operations:  # 该层的一个操作
            assert node.op.name != 'barrier'
            # print(node.op.name)
            index = nodes.index(node)
            layer_instructions.append(instructions[index])
            instruction2layer[index] = layer  # instruction在第几层
        layer2instructions.append(layer_instructions)

    return layer2instructions  #, instruction2layer, instructions, dagcircuit, nodes


def qiskit_to_layer_gate(instruction):
    name = instruction.operation.name
    parms = list(instruction.operation.params)
    return {
        'name': name,
        'qubits': [qubit.index for qubit in instruction.qubits],
        'params': [ele if isinstance(ele, float) else float(ele) for ele in parms],
    }


def my_circuit_to_dag(circuit: QuantumCircuit):
    instructions = []
    dagnodes = []

    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name
    dagcircuit.global_phase = circuit.global_phase
    dagcircuit.calibrations = circuit.calibrations
    dagcircuit.metadata = circuit.metadata

    dagcircuit.add_qubits(circuit.qubits)
    dagcircuit.add_clbits(circuit.clbits)

    for register in circuit.qregs:
        dagcircuit.add_qreg(register)

    for register in circuit.cregs:
        dagcircuit.add_creg(register)

    # for instruction, qargs, cargs in circuit.data:
    for instruction in circuit.data:
        operation = instruction.operation

        dag_node = dagcircuit.apply_operation_back(
            operation, instruction.qubits, instruction.clbits
        )
        if operation.name == 'barrier':
            continue
        instructions.append(instruction)  # TODO: 这个里面会不会有barrier
        dagnodes.append(dag_node)
        operation._index = len(dagnodes) - 1
        assert instruction.operation.name == dag_node.op.name

    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    return dagcircuit, instructions, dagnodes
