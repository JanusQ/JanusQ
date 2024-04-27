# JanusQ
A Software Framework for Analyzing, Optimizing, and Implementing Quantum Circuit.

## Installation

JanusQ can be installed by docker, wheel, and soruce code. Docker is recommended, as all functions have been tested on it.
Linux (Ubuntu 22.04 latest) platform and Python (3.10) is prefered when instaling by wheel or soruce code.

### From docker (Recommended)
Pull docker using docker 
```shell
docker pull janusq/janusq:latest
```
Run docker image
```shell
docker run -itd -p 8888:22 -p 9999:23 --name tutorial janusq/janusq
```
The jupyter notebook can be visited in "http://localhost:9999/lab?". The docker can be accessed via
```shell
ssh root@localhost -p 8888

or

 docker exec -it tutorial  bash
```
The code is in "/JanusQ". The examples that can be directly run is in "/JanusQ/examples"

### From wheel
Download janusq.whl from "JanusQ-main/dist".
```shell
pip install janusq.whl
```  
        
### From source code
Run following commends.
```shell
git clone git@github.com:JanusQ/JanusQ.git
cd JanusQ
pip install -r requirements.txt

```  
Then run commends to install Janus-SAT.
```shell
cd ./hyqsat
cmake .
make install
cp libm* ../janusq/hyqsat
cp minisat_core ../janusq/hyqsat
```  

## Structure of the Framework
- Janus-CT
  - vectorization
    - janusq/analysis/vectorization.py: This python script holds code related to the generation of path table and the vecterization of circuits
  - fidelity preidiction
    - janusq/analysis/fidelity_prediction.py: This python script holds code related to training a model and using it to predict the fidelity of a circuit.
  - unitary decomposition
    - janusq/analysis/unitary_decompostion.py: This python script holds code related to takes a unitary as input and decomposes it into matrices of basic gates, resulting in an equivalent circuit.
- Janus-FEM
  - benchmarking.py. Generate a circuit for measuring calibration matrices.
  - mitigation.py. Implement iterative calibration, which includes quantifying interactions between qubits and constructing Bayesian networks.
  - tools.py. Implement data format conversion.
- Janus-SAT
  - hyqsat/common: This dir stores common functions about cnf files, such as readCNF.
  - hyqsat/solveSatBy**.py: The main python function; use to solve sat problem.
- Janus-TC: Simulation Of Time crystal


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
1. When error "Permission denied 'minisat_core'" occurs, run following commends:
```shell
chmod +x janusq/hyqsat/minisat_core
```
