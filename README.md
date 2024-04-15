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
```
<!-- File Download:[Linux Janusq.](https://github.com/JanusQ/JanusQ/blob/main/dist/janusq-0.1.0-py3-none-any.whl) -->

## Structure of the Framework
- JanusQ
  - analysis: Circuit analysis.
  - optimizations: Circuit optimizations and readout optimizations.
  - data_objects: Generate random or algorithm circuit.
  - hyqsat: Solve 3-SAT problems.


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
