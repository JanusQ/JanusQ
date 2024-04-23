# JanusQ
A Software Framework for Analyzing, Optimizing, and Implementing Quantum Circuit.

## Installation
For Linux:  
- Python: version >= 3.10
  - In shell, 'python --version' views the python version.
```shell
pip install janusq.whl
# or you can pull source code directly from github.
git clone git@github.com:JanusQ/JanusQ.git
cd JanusQ
pip install -r requirements.txt




# or you can directly pull docker image from docker hub.
docker pull janusq:janusq:latest
```

<!-- File Download:[Linux Janusq.](https://github.com/JanusQ/JanusQ/blob/main/dist/janusq-0.1.0-py3-none-any.whl) -->

## Structure of the Framework
- JanusCT
  - vectorization
    - janusq/analysis/vectorization.py: This python script holds code related to the generation of path table and the vecterization of circuits
  - fidelity preidiction
    - janusq/analysis/fidelity_prediction.py: This python script holds code related to training a model and using it to predict the fidelity of a circuit.
  - fidelity optimization
    -  janusq/optimizations/mapping/mapping_ct.py：This python script holds code related to a typical compilation flow includes routing and scheduling. The routing pass transforms the circuit to satisfy the processor topology.
  - unitary decomposition
    - janusq/analysis/unitary_decompostion.py: This python script holds code related to takes a unitary as input and decomposes it into matrices of basic gates, resulting in an equivalent circuit.
  - bug identification
    - identify the potential bugs in the quantum algorithm implementation.
- JanusFEM
  - benchmarking.py. Generate a circuit for measuring calibration matrices.
  - mitigation.py. Implement iterative calibration, which includes quantifying interactions between qubits and constructing Bayesian networks.
  - tools.py. Implement data format conversion.
- HyQSAT
  - hyqsat/common: This dir stores common functions about cnf files, such as readCNF.
  - hyqsat/solveSatBy**.py: The main python function; use to solve sat problem.
- time crystal


## Related papers
**[ASPLOS 2024] MorphQPV: Exploiting Isomorphism in Quantum Programs to Facilitate Confident Verification**  
Siwei Tan, Debin Xiang, Liqiang Lu, Junlin Lu (Peking University), Qiuping Jiang (Ningbo University), Mingshuai Chen, and Jianwei Yin  

**[ASPLOS 2024] QuFEM: Fast and Accurate Quantum Readout Calibration Using the Finite Element Method**  
Siwei Tan, Hanyu Zhang, Jia Yu, Congliang Lang, Xinkui Zhao, Mingshuai Chen, Yun Liang, Liqiang Lu, and Jianwei Yin (Zhejiang University)  

**[HPCA 2023] HyQSAT: A Hybrid Approach for 3-SAT Problems by Integrating Quantum Annealer with CDCL**  
Siwei Tan, Mingqian Yu, Andre Python, Yongheng Shang, Tingting Li, Liqiang Lu, Jianwei Yin (Zhejiang University)  

**[MICRO 2023] QuCT: A Framework for Analyzing Quantum Circuit by Extracting Contextual and Topological Features**  
Siwei Tan, Congliang Lang, Liang Xiang; Shudi Wang, Xinghui Jia, Ziqi Tan, Tingting Li (Zhejiang University), Jieming Yin (Nanjing University of Posts and Telecommunications); Yongheng Shang, Andre Python, Liqiang Lu, Jianwei Yin (Zhejiang University)

**[Nature 2022] Digital Quantum Simulation of Floquet Symmetry Protected Topological Phases**  
Xu Zhang (Zhejiang University), Wenjie Jiang (Tsinghua University), Jinfeng Deng, Ke Wang, Jiachen Chen, Pengfei Zhang, Wenhui Ren, Hang Dong, Shibo Xu, Yu Gao, Feitong Jin, Xuhao Zhu, Qiujiang Guo, Hekang Li, Chao Song, Alexey V. Gorshkov, Thomas Iadecola, Fangli Liu, Zhe-Xuan Gong, Zhen Wang* (Zhejiang University), Dong-Ling Deng* (Tsinghua University) & Haohua Wang (Zhejiang University)

# Note
1. Permission denied 'minisat_core'  
chmod +x janusq/hyqsat/minisat_core