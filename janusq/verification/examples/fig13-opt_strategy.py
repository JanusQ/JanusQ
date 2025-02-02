import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import cv2
def get_data_mnist(mnist,n_qubits):
    data, labels = mnist.data.numpy(), mnist.targets.numpy()
    # idx = [i for i in range(labels.shape[0]) if int(labels[i]) in [3, 6]]
    data = data / 255.0  # 归一化像素值
    data = data.astype('float64')
    data_ = []
    labels_ = []
    for i in range(data.shape[0]):
        t = data[i].reshape((28, 28))
        t = cv2.resize(t, (2**(n_qubits//2), 2**(n_qubits//2))).reshape((1, 2**(n_qubits)))
        norm = np.linalg.norm(t)
        if np.sum(t) > 0:
            t = t / norm
            data_.append(t)
            labels_.append(labels[i])
    data_ = np.concatenate(data_, axis=0)
    labels_ = np.array(labels_)
    ##save the data
    np.save(f'data/minist_data_{n_qubits}.npy',data_)
    np.save(f'data/minist_labels_{n_qubits}.npy',labels_)
    return data_, labels_

def PCA_anlysis(data):
    # 计算协方差矩阵
    cov = np.cov(data, rowvar=0)
    # 计算特征值和特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(cov))
    # 将特征值从大到小排序
    eigValInd = np.argsort(eigVals)
    # 取前topNfeat个特征向量 使得贡献率达到98%
    topNfeat = 0
    for i in range(len(eigVals)):
        if np.sum(eigVals[:i]) / np.sum(eigVals) > 0.98:
            topNfeat = i
            break
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVects[:, eigValInd]
    # 将数据转换到新空间
    lowDDataMat = data * redEigVects
    return i, lowDDataMat, redEigVects
def strategy1minist(qubit_range=[4,6,8,10]):
    # 获取mnist数据集
    from torchvision import datasets, transforms
    Mnist = datasets.MNIST(root='data/minist', train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))
    for n_qubits in qubit_range:
        X_test, y_test = get_data_mnist(Mnist,n_qubits)
        max_sample,lowDDataMat, redEigVects = PCA_anlysis(X_test)
        print(max_sample)
        print(2**n_qubits*2)
        print(max_sample/2**(n_qubits+1))
        yield max_sample


def tomography_shots(n_qubits):
    ## this data is estimated from the variatial quantum tomography results for density matrix, unit: shots
    return {
        4: 300,
        6: 1000,
        8: 5000,
        10: 40000
    }[n_qubits]

def strategy3_shots(n_qubits):
    ## this formula is estimated from the variatial quantum tomography for distribution
    return 2**(n_qubits-1)

def strategy2_samples(n_qubits):
    ## |x+y> = |x>|0> + |0>|y> so the samples is 2*2^(n/2)
    return 2*2**(n_qubits//2)

def original_samples(n_qubits):
    ## the original samples is 2^(n+1)
    return 2**(n_qubits+1)

def plot_results():
    #convert to pandas dataframe
    import pandas as pd
    datas = []
    qubits_range = [4,6,8,10]
    strategy1ministsamples = strategy1minist(qubits_range)
    strategy1ministsamples = {qubits:sample for qubits,sample in zip(qubits_range,strategy1ministsamples)}

    for qubits in qubits_range:
        datas.append({
            'qubits': qubits,
            'strategy-adapt': strategy1ministsamples[qubits]* tomography_shots(qubits),
            'strategy-const': strategy2_samples(qubits)* tomography_shots(qubits),
            'strategy-prop': original_samples(qubits)* strategy3_shots(qubits),
            'non-optimize': original_samples(qubits)* tomography_shots(qubits)
        })
    dfshots = pd.DataFrame(datas)

    datasamples = []
    for qubits in [4,6,8,10]:
        datasamples.append({
            'qubits':qubits,
            'strategy-adapt': strategy1ministsamples[qubits],
            'strategy-const': strategy2_samples(qubits),
            'strategy-prop': original_samples(qubits),
            'non-optimize': original_samples(qubits)
        })
    dfsamples = pd.DataFrame(datasamples)
        
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    scale = 0.009
    fig = plt.figure(figsize=(2500*scale , 800*scale))
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
    markersize = 20
    linewidth = 5

    colors = ['#3274A1','#E80101','#7EB57E','#96CBF0']

    axes = plt.axes([0,0,0.42,1])
    for i,col in enumerate(['non-optimize','strategy-adapt','strategy-const','strategy-prop']):
        axes.bar(dfsamples.index+0.15*i,dfsamples[col],width=0.15,color=colors[i],label=col)
    axes.legend(frameon=False,bbox_to_anchor=(2.5,1.3),ncol=4)
    axes.tick_params(axis='x',which='major',length=20)
    # axes.set_xticklabels(allnames,rotation=90)
    axes.set_xlabel('# qubits')
    axes.set_yscale('log')
    axes.set_xticks(dfshots.index+0.15*2)
    axes.set_xticklabels(dfshots.qubits)
    axes.grid(axis='y',linestyle='--',linewidth=3,color='#B0B0B0')
    axes.set_ylabel('max input samples')

    axes = plt.axes([0.58,0,0.42,1])
    axes.grid(axis='y',linestyle='--',linewidth=3,color='#B0B0B0')
    for i,col in enumerate(['non-optimize','strategy-adapt','strategy-const','strategy-prop']):
        axes.bar(dfshots.index+0.15*i,dfshots[col],width=0.15,color=colors[i],label=col)
    axes.bgcolor = '#F3F3F3'
    # axes.set_yscale('log')
    axes.tick_params(axis='x',which='major',width=5,length=20)
    # axes.set_xticklabels(allnames,rotation=90)
    axes.set_xlabel('# qubits')
    axes.set_yscale('log')
    axes.set_yticks([1e1,1e3,1e5,1e7])
    axes.set_ylim([1,1e8])
    axes.set_xticks(dfshots.index+0.15*2)
    axes.set_xticklabels(dfshots.qubits)
    axes.set_ylabel('output shots per input')
    axes.set_ylabel('total shots')
    if os.path.exists('examples/fig11-opt_strategy/') == False:
        os.makedirs('examples/fig11-opt_strategy/')
    fig.savefig('examples/fig11-opt_strategy/optimize.pdf',dpi=600,format='pdf',bbox_inches='tight')

if __name__ == "__main__":
    plot_results()