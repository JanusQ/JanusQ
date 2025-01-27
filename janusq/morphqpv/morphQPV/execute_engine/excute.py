from .circuit_converter import *
from qiskit.quantum_info import partial_trace,DensityMatrix,random_clifford
from qiskit import Aer, execute,transpile
import pennylane as qml
import numpy as np
from functools import partial
from tqdm import tqdm
from .variational_tomography import estimate_input,estimate_output
from .metric import fidelity
from .evaluateCloud import IBMCloudRun
from qiskit import Aer
from qiskit.providers.fake_provider import FakeLagos
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import StateTomography
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator, Options
def convert_counts_to_distribution(counts):
    distribution = np.zeros(2**len(list(counts.keys())[0]))
    for key in counts:
        distribution[int(key,2)] = counts[key]
    return distribution

def convert_density_to_state(density_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
    # idx = np.argmax(eigenvalues)
    statevector = eigenvectors[-1]
    statevector= statevector.astype(np.complex128)
    statevector = statevector/np.linalg.norm(statevector,ord=2)
    return statevector.reshape(-1)

def convert_state_to_density(state_vector):
    state_vector = state_vector.reshape(-1,1)
    density_matrix = state_vector @ state_vector.conj().T
    return density_matrix/density_matrix.trace()

def get_sample_densitymatrix(data):
    sample,clliford = data
    dev = qml.device('default.qubit', wires=sample.size)
    @qml.qnode(dev)
    def circuit(sample):
        for i in sample:
            if sample[i] == 1:
                qml.PauliX(i)
        layer_circuit_to_qml_circuit(clliford[::-1])
        return qml.density_matrix(wires=range(sample.size))
    return np.array(circuit(sample))

def get_random_clliford(append_qubits):
    n_qubits = len(append_qubits)
    circuit = random_clifford(n_qubits).to_circuit()
    # circuit = transpile(circuit,basis_gates=['u1','u2','u3','cx'],optimization_level=3)
    return qiskit_circuit_to_layer_cirucit(circuit)

class ExcuteEngine:

    def __init__(self,layer_circuit) -> None:
        self.circuit = layer_circuit
        self.qubits = ExcuteEngine.get_qubits(layer_circuit)
        self.n_qubits = len(self.qubits)
        pass
    @property
    def gates_num(self):
        return sum([len(layer) for layer in self.circuit])
    @staticmethod
    def get_qubits(layer_circuit):
        qubits = []
        for layer in layer_circuit:
            for gate in layer:
                qubits += gate['qubits']
        return list(set(qubits))
    
    @staticmethod
    def output_state_tomography(layer_circuit,output_qubits,device='ibmq'):
        if device == 'simulate':
            return ExcuteEngine.excute_on_pennylane(layer_circuit,'density',output_qubits=output_qubits)
        if device == 'clliford':
            return ExcuteEngine.excute_on_qiskit(layer_circuit,'state_tomography_clliford',output_qubits=output_qubits)
        if device == 'mps':
            return ExcuteEngine.excute_on_qiskit(layer_circuit,'state_tomography_mps',output_qubits=output_qubits)
        if device == 'qasm':
            return ExcuteEngine.excute_on_qiskit(layer_circuit,'state_tomography_qasm')
        if device == 'qasm_noise':
            return ExcuteEngine.excute_on_qiskit(layer_circuit,'state_tomography_qasm_noise')
        elif device == 'ibmq':
            state,fid,n = estimate_output(layer_circuit,len(ExcuteEngine.get_qubits(layer_circuit)),output_qubits,dev_type='ibmnoiseqasm')
            print(f'output tomography, shots: {n}, fidelity: {fid}')
            return state
    
    @staticmethod
    def output_state_statistic(layer_circuit,output_qubits,device='ibmq'):
        if device == 'simulate':
            return ExcuteEngine.excute_on_pennylane(layer_circuit,'expectation',output_qubits=output_qubits)
        elif device == 'ibmq':
            return ExcuteEngine.excute_on_qiskit(layer_circuit,'distribution_ibmfakeq')
        elif device == 'qasm':
            return ExcuteEngine.excute_on_qiskit(layer_circuit,'state_tomography_qasm')

    @staticmethod
    def qiskit_state_tomography(backend,layer_circuit,output_qubits):
        qstexp = StateTomography(layer_circuit,measurement_indices=output_qubits)
        qstdata = qstexp.run(backend,seed_simulation=100).block_for_results()
        state_result = qstdata.analysis_results("state")
        return state_result.value.data

    @staticmethod
    def excute_on_qiskit(layer_circuit,type='sample',shots=1000,output_qubits=None):
        N_qubits = len(ExcuteEngine.get_qubits(layer_circuit))
        qiskit_circuit = layer_circuit_to_qiskit_circuit(layer_circuit,N_qubits)
        if type == 'state_tomography_qasm':
            # backend = AerSimulator.from_backend(FakeLagos())
            backend = Aer.get_backend('qasm_simulator')
            return ExcuteEngine.qiskit_state_tomography(backend,qiskit_circuit,output_qubits)
        if type == 'state_tomography_clliford':
            backend = Aer.get_backend('extended_stabilizer')
            return ExcuteEngine.qiskit_state_tomography(backend,qiskit_circuit,output_qubits)
        if type == 'state_tomography_mps':
            backend = Aer.get_backend('aer_simulator_stabilizer')
            return ExcuteEngine.qiskit_state_tomography(backend,qiskit_circuit,output_qubits)
        
        if type == 'state_tomography_qasm_noise':
            backend = AerSimulator.from_backend(FakeLagos())
            qstexp = StateTomography(qiskit_circuit)
            options = Options(shots=10000,resilience_level=3,optimization_level=3)
            qstdata = qstexp.run(backend, seed_simulation=100,run_options=options).block_for_results()
            state_result = qstdata.analysis_results("state")
            # fid_result = qstdata.analysis_results("state_fidelity")
            return state_result.value.data
        if type == 'clliford':
            sim_stabilizer = Aer.get_backend('aer_simulator_stabilizer')
            job_stabilizer_sim = sim_stabilizer.run(qiskit_circuit).result()
            return total_density
        if type == 'mps':
            qiskit_circuit.save_statevector(label='end_sv')
            simulator = AerSimulator(method='matrix_product_state')
            tcirc = transpile(qiskit_circuit, simulator)
            result = simulator.run(tcirc).result()
            data = result.data(0)
            return data['end_sv']
        if type == 'prob':
            backend = Aer.get_backend('qasm_simulator')
            qiskit_circuit.measure_all()
            distribution = execute(qiskit_circuit, backend,shots=shots).result().get_counts(qiskit_circuit)
            return convert_counts_to_distribution(distribution)/shots

        if type == 'statevector':
            qiskit_circuit.remove_final_measurements()
            backend = Aer.get_backend('statevector_simulator')
            result = execute(qiskit_circuit, backend).result()
            return result.get_statevector(qiskit_circuit)
        if type == 'sample':
            backend = Aer.get_backend('qasm_simulator')
            result = execute(qiskit_circuit, backend,shots=shots).result()
            return result.get_counts(qiskit_circuit)
        if type == 'density':
            qiskit_circuit.remove_final_measurements()
            total_density = DensityMatrix.from_instruction(qiskit_circuit).data
            if output_qubits is None:
                return total_density
            else:
                return partial_trace(total_density,qargs=[i for i in range(N_qubits) if i not in output_qubits])
        if type == 'distribution_ibmq':
            dev = IBMCloudRun(shots=10000)
            qiskit_circuit.measure_all()
            distribution,job_id = dev.systemRun(qiskit_circuit)
            return distribution
        if type == 'distribution_ibmfakeq':
            dev = IBMCloudRun(shots=10000)
            qiskit_circuit.measure_all()
            distribution,job_id = dev.simulate(qiskit_circuit,noise_device='ibm_lagos')
            return distribution
        if type == 'distribution_qasm':
            
            backend = Aer.get_backend('qasm_simulator')
            noise_model = NoiseModel.from_backend(FakeLagos())
            backend.set_options(noise_model=noise_model,coupling_map=FakeLagos().configuration().coupling_map,basis_gates=FakeLagos().configuration().basis_gates)
            qiskit_circuit.measure_all()
            distribution = execute(qiskit_circuit, backend,shots=10000).result().get_counts(qiskit_circuit)

            return convert_counts_to_distribution(distribution)/10000
        if type == 'unitary':
            import qiskit.quantum_info as qi
            op = qi.Operator(qiskit_circuit)
            return op._data
    

    @staticmethod
    def excute_on_pennylane(layer_circuit,type='sample',shots=1000,output_qubits=None,input_state=None,input_qubits=None,N_qubits=None):
        if output_qubits is None:
            output_qubits = ExcuteEngine.get_qubits(layer_circuit)
        if N_qubits is None:
            N_qubits = len(ExcuteEngine.get_qubits(layer_circuit))
        dev = qml.device('default.qubit', wires=N_qubits,shots=shots)
        ## noise simulation
        if type == 'sample':
            @qml.qnode(dev)
            def circuit():
                layer_circuit_to_qml_circuit(layer_circuit)
                return qml.sample()
            return circuit()
        if type == 'distribution':
            @qml.qnode(dev)
            def circuit():
                layer_circuit_to_qml_circuit(layer_circuit)
                return qml.probs(wires=output_qubits)
            return circuit()
        if type == 'expectation':
            @qml.qnode(dev)
            def circuit():
                layer_circuit_to_qml_circuit(layer_circuit)
                return qml.expval(qml.PauliZ(wires=0)@qml.PauliZ(wires=1)@qml.PauliZ(wires=2))
            return circuit()
        if type == 'statevector':
            @qml.qnode(dev)
            def circuit():
                layer_circuit_to_qml_circuit(layer_circuit)
                return qml.state()
            return circuit()
        if type == 'density':
            @qml.qnode(dev)
            def circuit():
                layer_circuit_to_qml_circuit(layer_circuit)
                return qml.density_matrix(output_qubits)
            return circuit()
        
        if type == 'unitary':
            @qml.qnode(dev)
            def circuit():
                layer_circuit_to_qml_circuit(layer_circuit)
                return qml.unitary(wires=output_qubits)
            return circuit()
        if type == 'definedinput':
            @qml.qnode(dev)
            def circuit(input_state,input_qubits):
                qml.QubitStateVector(input_state, wires=input_qubits)
                layer_circuit_to_qml_circuit(layer_circuit)
                return qml.density_matrix(wires=output_qubits)
            return circuit(input_state,input_qubits)
        if type == 'definedinputstatevector':
            @qml.qnode(dev)
            def circuit(input_state,input_qubits):
                qml.QubitStateVector(input_state, wires=input_qubits)
                layer_circuit_to_qml_circuit(layer_circuit)
                return qml.state()
            return circuit(input_state,input_qubits)
    
    def partial_output_tomography(self,qubits,minimal_shots=100):

        clifford_circuits = [get_random_clliford(qubits) for _ in range(minimal_shots)]
        excuting_circuits = [self.circuit + clifford_circuit for clifford_circuit in clifford_circuits]
        excute_function = partial(ExcuteEngine.excute_on_pennylane,type='sample',shots=1,output_qubits=qubits)
        results = list(map(excute_function,tqdm(excuting_circuits,desc='output tomographing--clliford')))
        Phis = list(map(get_sample_densitymatrix,zip(results,clifford_circuits)))
        return Phis
    
    def state_tomography_by_clliford(self,output_qubits=None,max_shots = 10000,unit_shots = 1000,real=None):
        shots = 0
        fid = 0
        Phis = []
        if output_qubits is None:
            output_qubits = self.qubits
        n_qubits = len(output_qubits)
        current_density = np.eye(2**n_qubits)/2**n_qubits
        while abs(fid-1) > 1e-2:
            Phis += self.partial_output_tomography(output_qubits,minimal_shots=unit_shots)
            shots += unit_shots
            build_density = (2**n_qubits+1)*np.sum(Phis,axis=0)/shots - np.eye(2**n_qubits)
            # build_density = build_density/np.trace(build_density)
            fid = fidelity(real,build_density)
            current_density =build_density
            print(f'output tomography, shots: {shots}, fidelity: {fid}')
            if shots > max_shots:
                break
        return build_density

    
    def input_tomography(self,output_qubits,real_state):
        
        return estimate_input(self.circuit,self.n_qubits,self.qubits,real_state,output_qubits=output_qubits)
    
    def output_tomography(self,output_qubits,real_state=None):
        
        return estimate_output(self.circuit,self.n_qubits,output_qubits)

    
            
    def get_density_matrix(self,qubits,platform='pennylane'):
        if platform == 'qiskit':
            return ExcuteEngine.excute_on_qiskit(self.circuit,'density',qubits=qubits)
        elif platform == 'pennylane':
            return ExcuteEngine.excute_on_pennylane(self.circuit,'density',output_qubits=qubits)
    
    def get_statevector(self,platform='pennylane'):
        if platform == 'qiskit':
            return ExcuteEngine.excute_on_qiskit(self.circuit,'statevector')
        elif platform == 'pennylane':
            return ExcuteEngine.excute_on_pennylane(self.circuit,'statevector')
    

def unit_test():
    from scipy.stats import unitary_group
    all_qubits = list(range(2))
    N_qubit = len(all_qubits)
    U = unitary_group.rvs(2**N_qubit)
    V = unitary_group.rvs(2**N_qubit)
    layer_circuit = [
        [{'name':'unitary','params': U,'qubits':all_qubits}],
        [{'name':'unitary','params': V,'qubits':all_qubits}],
    ]
    engine = ExcuteEngine(layer_circuit)
    print(engine.get_density_matrix())
    print(engine.get_statevector())
    print(engine.excute_on_qiskit(layer_circuit,'density'))
    print(engine.excute_on_qiskit(layer_circuit,'statevector'))
    print(engine.excute_on_pennylane(layer_circuit,'statevector'))
    print(engine.excute_on_pennylane(layer_circuit,'density'))
    print(engine.state_tomography_by_clliford(layer_circuit,[0,1],shots=1000))

if __name__ == '__main__':
    unit_test()