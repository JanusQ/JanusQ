
from time import perf_counter
from scipy.linalg import expm
import numpy as np
import pennylane as qml
from chocoq.utils.gadget import iprint
def apply_convert(list_qubits, bit_string):
    num_qubits = len(bit_string)
    for i in range(0, num_qubits - 1):
        qml.CNOT(wires=[list_qubits[i + 1], list_qubits[i]])
        if bit_string[i] == bit_string[i + 1]:
            qml.PauliX(list_qubits[i])
    qml.Hadamard(list_qubits[-1])
    qml.PauliX(list_qubits[-1])

def apply_reverse(list_qubits, bit_string):
    num_qubits = len(bit_string)
    qml.PauliX(list_qubits[-1])
    qml.Hadamard(list_qubits[-1])
    for i in range(num_qubits - 2, -1, -1):
        if bit_string[i] == bit_string[i + 1]:
            qml.PauliX(list_qubits[i])
        qml.CNOT(wires=[list_qubits[i + 1], list_qubits[i]])

def decompose_phase_gate(list_qubits, ancilla, phase):
    num_qubits = len(list_qubits)
    if num_qubits == 1:
        qml.PhaseShift(phase, wires=list_qubits[0])
    elif num_qubits == 2:
        qml.ControlledPhaseShift(phase, wires=list_qubits)
    else:
        half_num_qubit = num_qubits // 2
        qr1 = list_qubits[:half_num_qubit]
        qr2 = list_qubits[half_num_qubit:]
        qml.RZ(-phase/2, wires=ancilla[0])
        qml.MultiControlledX(wires=list(qr1) + [ancilla[0]])
        qml.RZ(phase/2, wires=ancilla[0])
        qml.MultiControlledX(wires=list(qr2) + [ancilla[0]])
        qml.RZ(-phase/2, wires=ancilla[0])
        qml.MultiControlledX(wires=list(qr1) + [ancilla[0]])
        qml.RZ(phase/2, wires=ancilla[0])
        qml.MultiControlledX(wires=list(qr2) + [ancilla[0]])

# separate functions facilitate library calls
def driver_component(list_qubits, list_ancilla, bit_string, t):
    apply_convert(list_qubits, bit_string)
    # # qml.Barrier
    decompose_phase_gate(list_qubits, list_ancilla, -t)
    # # qml.Barrier
    qml.PauliX(list_qubits[-1])
    decompose_phase_gate(list_qubits, list_ancilla, t)
    qml.PauliX(list_qubits[-1])
    # # qml.Barrier
    apply_reverse(list_qubits, bit_string)
    pass
    
def get_driver_component(num_qubits, t, bit_string):
    num_wires = num_qubits + 2
    dev = qml.device('default.qubit', wires=num_wires)

    @qml.qnode(dev)
    def circuit():
        driver_component(list(range(num_qubits)), [num_qubits], bit_string,  2 * np.pi * t)
        return qml.state()
    # iprint(qml.matrix(circuit)())
    return circuit

def tensor_product(matrices):
    sigma_plus = np.array([[0, 1], [0, 0]])
    sigma_minus = np.array([[0, 0], [1, 0]])
    identity = np.array([[1, 0], [0, 1]])
    mlist = [sigma_plus, sigma_minus, identity]
    result = mlist[matrices[0]]
    for matrix in matrices[1:]:
        result = np.kron(result, mlist[matrix])
    return result

def get_simulate_unitary(t, bit_string):    
    # for unitary build contrary_v
    U = expm(-1j * 2 * np.pi * t * (tensor_product(bit_string)+tensor_product([not(i) for i in bit_string])))
    return U

def decompose_unitary(t, bit_string):
    unitary_gate = get_simulate_unitary(t, bit_string)
    # iprint(unitary_gate)
    dev = qml.device('default.qubit', wires=len(bit_string))
    
    @qml.qnode(dev)
    def circuit():
        qml.QubitUnitary(unitary_gate, wires=range(len(bit_string)))
        return qml.state()
    return circuit

if __name__ == '__main__':
    # 不要截断 是否使用科学计数法 输出浮点数位数 宽度
    def set_print_form(suppress=True, precision=4, linewidth=300):
        np.set_iprintoptions(threshold=np.inf, suppress=suppress, precision=precision,  linewidth=linewidth)
    set_print_form()
    import numpy as np
    def get_decompose_time(bit_string=[0, 0, 1, 1], f=None):
        t = 0.9
        time_dict = {}
        write_string = ''
        num_qubits = len(bit_string)
        time_start = perf_counter()
        qc = get_driver_component(num_qubits, t, bit_string)
        # iprint(qml.draw(qc)())
        # iprint(qml.matrix(qc)().round(2))
        # exit()
        time_end = perf_counter()
        time_init = time_end - time_start
        time_dict['ours'] = time_init
        iprint(f'decompose time: {time_end - time_start}')
        depth = qml.specs(qc)()['resources'].depth
        num_gates = qml.specs(qc)()['resources'].num_gates
        iprint(f'depth: {depth}')
        iprint(f'num_gates: {num_gates}')
        write_string += f'{str(num_qubits)}, {str(num_qubits)}, {str(time_init)}, {str(depth)}, {str(num_gates)},\n'

        # decompose into multi-qubit gates
        for i in range(6, 1, -1):
            time_start = perf_counter()
            # transpile_circuit = decompose_circuit(qc, i)
            transpile_circuit = qc # Tried but failed
            time_end = perf_counter()
            time_decompose = time_end - time_start
            depth = qml.specs(transpile_circuit)()['resources'].depth
            # num_nonlocal_gates = transpile_circuit.num_nonlocal_gates()
            time_decompose += time_init
            iprint(f'max_control_qubits: {i}')
            iprint(f'depth: {depth}')
            # iprint(f'num_nonlocal_gates: {num_nonlocal_gates}')
            iprint(f'time_decompose: {time_decompose}')
            write_string += f'{str(num_qubits)}, {str(i)}, {str(time_decompose)}, {str(depth)}, {str(num_gates)},\n'
        
        time_start = perf_counter()
        uqc = decompose_unitary(t, bit_string)
        # iprint(qml.matrix(uqc)())
        # exit()
        time_end = perf_counter()
        iprint(f'unitary_time: {time_end - time_start}')
        udepth = qml.specs(uqc)()['resources'].depth
        unum_gates = qml.specs(uqc)()['resources'].num_gates
        iprint(f'udepth: {udepth}')
        iprint(f'unum_gates: {unum_gates}') #?? it's wrong
        time_dict['unitary'] = time_end - time_start
        write_string += f'{str(num_qubits)}, {str(2)}, {str(time_dict["unitary"])}, {str(udepth)}, {str(unum_gates)},\n'
        if f:
            f.write(write_string)
        return time_dict
    time_list = []
    with open('../../../implementations/data/decompose_result.csv','w+') as f:
        f.write('num_qubits, ours, unitary\n')
        for i in range(3, 9):
            bit_string = np.random.randint(2, size=i)
            # bit_string = [0, 1, 0, 1, 1, 1, 1]
            iprint(f'num_qubits: {i}')
            time = get_decompose_time(bit_string, f)
            time_list.append(time)
            iprint('--------------------')
    import pandas as pd
    df = pd.DataFrame(time_list)
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    scale = 0.009
    fig = plt.figure(figsize=(2500 * scale , 800 * scale))
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 62
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
    mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    # set axes linewidth
    mpl.rcParams['axes.linewidth'] = 5
    ## set ticks linewidth
    mpl.rcParams['xtick.major.size'] = 20
    mpl.rcParams['xtick.major.width'] = 5
    mpl.rcParams['ytick.major.size'] = 20
    mpl.rcParams['ytick.major.width'] = 5
    axes = plt.axes([0,0,1,1])
    for i, col in enumerate(df.columns):
        axes.bar(df.index + 0.3 * i, df[col], width=0.3, label=col)
    axes.legend(frameon=False, bbox_to_anchor=(1, 1), ncol=4)
    axes.tick_params(axis='x', which='major', length=20)
    axes.set_xlabel('# qubits')
    axes.set_yscale('log')
    axes.set_xticks(df.index + 0.15)
    axes.set_xticklabels(df.index + 3)
    axes.grid(axis='y', linestyle='--', linewidth=3, color='#B0B0B0')
    axes.set_ylabel('Time (s)')
    fig.savefig('../../../implementations/figures/decompose_optimize.pdf', dpi=600, format='pdf', bbox_inches='tight')