import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pandas as pd
def get_theory_data_once(qubit, sample):
    return {
        'qubits': qubit,
        'samples': sample,
        'probability': np.clip(sample / (2 ** (qubit+1)) *5*(1+np.random.normal(0, 0.01)), 0, 1)
    }
def get_theory_data(qubits):
    for qubit in qubits:
        for sample in np.linspace(0, 2 ** (qubit+1), 100):
            yield get_theory_data_once(qubit, sample)

def get_surface_data(qubits):
    samples = []
    for qubit in qubits:
        samples.extend([int(i) for i in np.linspace(0, 2 ** (qubit+1)/5, min(200,int(2 ** (qubit+1)/5)))])
    samples = np.array(samples)
    samples.sort(axis=0)
    for qubit in qubits:
        probability = []
        new_qubits = [qubit-0.03,qubit+0.03]
        data = [get_theory_data_once(qubit, sample)['probability'] for sample in samples]
        probability.append(data)
        probability.append(data)

        yield new_qubits,np.array(samples),np.array(probability)
            

def plot_results():
    scale = 0.01
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(2500*scale , 2000*scale))
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 60/0.01*scale
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
    mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    # set axes linewidth
    mpl.rcParams['axes.linewidth'] = 5
    ## set ticks linewidth
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 5
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 5
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 5
    data_all = pd.DataFrame(get_theory_data(range(5, 17)))
    ax = plt.axes(projection='3d')
    start = 3
    end = 19
    xs = np.arange(start,end)
    for x,y,z in get_surface_data(xs):
        y = np.log2(y)
        X, Y = np.meshgrid(x, y)
        z = z.T
        cmap = plt.cm.OrRd
        ax.plot_surface(X,Y,z,linewidth=0,color='black',alpha=0.8)
        X = np.vstack((X[:,-1],X[:,-1]))
        Y = np.vstack((Y[:,-1],np.ones_like(Y[:,-1])* np.max(y)))
        z = np.vstack((z[:,-1],z[:,-1]))
        ax.plot_surface(X,Y,z,linewidth=0,color='#99C8EA',alpha=0.7)

    diff=0
    ax.set_xticks(np.arange(start,end,3)-diff)
    ax.set_xticks(np.arange(start,end)-diff, minor=True)
    # ax.set_yticks(np.arange(9,16,2)-0.25, minor=True)
    ax.set_xticklabels(np.arange(start,end,3))
    ax.set_yticks(np.arange(0,16,4))
    ax.set_yticklabels([r'$2^{'+str(i)+r'}$' for i in np.arange(0,16,4)])
    ax.set_zticks(np.arange(0,1.1,0.2))
    ax.set_zticklabels(np.arange(0,1.1,0.2)*100)

    ax.set_zlim([0,1])
    ax.set_ylim([0,max(y)])
    ax.view_init(elev=17, azim=-36);
    ## background color
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ## grid
    ax.grid(xdata=x, ydata=y, zdata=z, linestyle='--', linewidth=0.3, color='gray')
    ## save
    directory = os.path.abspath(__file__).split('/')[-1].split('.')[0]
    resultspath = os.path.join(os.path.dirname(__file__), f'{directory}/')
    if not os.path.exists(resultspath):
        os.mkdir(resultspath)
    fig.savefig(f'{resultspath}fig9(b)-theorem2.svg', dpi=600, format='svg', bbox_inches='tight')

if __name__ == '__main__':
    plot_results()

