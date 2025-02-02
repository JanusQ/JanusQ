import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


from morphQPV.execute_engine.metric   import fidelity
from morphQPV.execute_engine.excute import ExcuteEngine,convert_state_to_density
from morphQPV.approximate_complex.orthgonal_app import approximate
from scipy.stats import unitary_group
from data.Qbenchmark import layer_circuit_generator
from scipy.linalg import qr
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def generate_base_states(n_qubits,n_base_states):
    for i in range(n_base_states):
        base_state = np.zeros((2**n_qubits),dtype=np.complex128)
        base_state[i] = 1
        yield base_state


def get_proj_unitary(target_state):
    """Get the projection matrix of the target state."""
    # orthgonal_basis = get_orthgonal_basis(target_state)
    N = target_state.shape[0]
    unitary = np.zeros((N,N),dtype=np.complex128)
    Q, R = qr(np.column_stack((target_state, np.eye(N))), mode='economic')
    unitary[:, 0] = target_state
    # Ensure that the first column of Q is |phi> (up to a global phase, which doesn't matter)
    if np.allclose(Q[:, 0],target_state, atol=1e-10):
        unitary[:, 1:] = Q[:, 1:]
    elif np.allclose(Q[:, 0], - target_state, atol=1e-10):  # The phase might be flipped
        unitary[:, 1:] = -Q[:, 1:]
  
def generate_orthgonal_states(n_qubits):
    target_state = np.random.randn((2**n_qubits))
    target_state[np.abs(target_state) < 1e-3] = 0
    target_state = target_state/np.linalg.norm(target_state,ord=2)
    N = target_state.shape[0]
    ## get the orthgonal basis
    orthgonal_basis = []
    orthgonal_basis.append(target_state)
    for i in range(1,N):
        base_state = np.random.randn((2**n_qubits))
        ## clip the target state
        ## if some elements are too small, we set them to zero
        base_state[np.abs(base_state) < 1e-3] = 0
        base_state = base_state/np.linalg.norm(base_state,ord=2)
        for j in range(i):
            base_state -= np.dot(base_state,orthgonal_basis[j])*orthgonal_basis[j]
        base_state = base_state/np.linalg.norm(base_state,ord=2)
        orthgonal_basis.append(base_state)
    for i in range(N):
        orthgonal_basis[i] = orthgonal_basis[i].astype(np.complex128)
    return orthgonal_basis
def get_fidelity(circuit,base_states,acc_num):
    for _ in range(acc_num):
        input_state = generate_input_states(circuit)
        approximate_state = approximate(input_state,base_states)
        yield  np.abs(approximate_state.conj().dot(input_state)) ** 2

def generate_input_states(circuit,n_qubits=6):
    random_idx = np.random.randint(1,len(circuit))
    input_state = ExcuteEngine.excute_on_pennylane(circuit[:random_idx],type='statevector',output_qubits=list(range(n_qubits)),N_qubits=n_qubits)
    return input_state

def get_results(algos, n_qubits,filename = 'distribution_random',resdir = 'examples/fig10-confidence/'):
    with open(f'{resdir}{filename}.csv','w') as f:
        f.write('algo,samples,mean,std\n')
    with open(f'{resdir}{filename}_detail.csv','w') as f:
        f.write('algo,samples,accuracy\n')
        
    all_base_states = generate_orthgonal_states(n_qubits)
    for samples in range(2,2**n_qubits+1,2):
        base_states = all_base_states[:samples]
        for algo in algos:
            circuit = layer_circuit_generator(algo,n_qubits,m=2,k=2)
            fidelity = list(get_fidelity(circuit,base_states,200))
            print(algo,samples,np.mean(fidelity),np.std(fidelity))
            with open(f'{resdir}{filename}.csv','a') as f:
                f.write(f'{algo},{samples},{np.mean(fidelity)},{np.std(fidelity)}\n')
            with open(f'{resdir}{filename}_detail.csv','a') as f:
                for fidelity in fidelity:
                    f.write(f'{algo},{samples},{fidelity}\n')

def plot_results(algos,filename,resdir = 'examples/fig10-confidence/'):
    accs = pd.read_csv(f'{resdir}{filename}_detail.csv')
    accsdata = {}
    for group, acc in accs.groupby(['algo','samples']):
        accsdata[group] = acc
    accsdata.keys()
    def get_confidence(acc,threldhold):
        confidence = np.sum(acc['accuracy']>threldhold)/len(acc)
        return confidence
    def beta_distribution(x,alpha):
        beta = 64 - alpha
        y = (x**(alpha-1))*((1-x)**(beta-1))/((alpha-1)*(beta-1)+1e-6)
        return y
    def theo_confidence(x,threldhold):
        alpha = int(np.mean(x)*64)
        beta = 64 - alpha
        y = beta_distribution(np.arange(0,1,0.01),alpha)
        y = y/np.sum(y)
        ## integrate the distribution for x>threldhold
        confidence_t = 1-np.sum(y[0:int(threldhold*100)])
        return confidence_t
    scale = 0.01
    fig = plt.figure(figsize=(2200*scale , 2200*scale))
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
    mpl.rcParams['xtick.minor.size'] = 10
    mpl.rcParams['xtick.minor.width'] = 3
    mpl.rcParams['ytick.major.size'] = 20
    mpl.rcParams['ytick.major.width'] = 5
    mpl.rcParams['ytick.minor.size'] = 10
    mpl.rcParams['ytick.minor.width'] = 3
    markersize = 20
    linewidth = 5
    threldhold = 0.55
    # axes = plt.axes([0,0,1,0.45])
    colors = ['#1F77B4','#FF8663','#008000','#FF0000','#E80101']
    ## draw the distribution of the real data
    for i,algo in enumerate(algos):
        axes = plt.axes([0,1-0.3*i,1,0.3])
        m = algos.index(algo)%(len(colors)-1)+1
        accuracy = accs[accs.algo==algo]
        confidences = np.array([get_confidence(accsdata[(algo,sample)],threldhold) for sample in accuracy.samples.unique()])
        theoral_confidences = []
        for sample in accuracy.samples.unique():
            acc = accsdata[(algo,sample)]
            confidence = theo_confidence(acc['accuracy'],threldhold)
            theoral_confidences.append(confidence)
        theoral_confidences = np.array(theoral_confidences)
        axes.plot(accuracy.samples.unique(),theoral_confidences*100, color=colors[m],linewidth=linewidth, label='estimation',linestyle='--')
        axes.plot(accuracy.samples.unique(),confidences*100, color=colors[m],linewidth=linewidth, label=algo,linestyle='-',marker='.',markersize=markersize)
        axes.set_yticks(np.arange(0,110,5), minor=True)
        axes.set_yticks(np.arange(0,110,20), minor=False)
        axes.grid(which='major', axis='y', linewidth=2, linestyle='-', color='0.75')
        axes.set_ylim([0,100])
        axes.set_ylabel('confidence (%)',fontsize=62)
        axes.legend(loc='upper left',fontsize=62)
    max_sample = max(accuracy.samples.unique())
    axes.set_xticks(np.arange(0,max_sample+1,4), minor=False)
    axes.set_xticklabels(np.arange(0,max_sample+1,4), fontsize=62)
    axes.set_xticks(np.arange(0,max_sample+1,1), minor=True)
    axes.set_xlabel('Number of samples',fontsize=62)
    axes.set_xlim([0,max_sample])
    plt.savefig(f'{resdir}confidence.svg', bbox_inches='tight')

if __name__ == '__main__':
    directory = os.path.abspath(__file__).split('/')[-1].split('.')[0]
    respath = os.path.join(os.path.dirname(__file__), f'{directory}/')
    if not os.path.exists(respath):
        os.makedirs(respath)
    algos = ['xeb','qnn','shor','qec']
    get_results(algos,6,filename = 'distribution_samples64',resdir = respath)
    plot_results(algos,filename = 'distribution_samples64',resdir = respath)



