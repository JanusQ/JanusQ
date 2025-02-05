# JanusQ: A Software Framework for Analyzing, Optimizing, Verifying, Implementing, and Calibrating Quantum Circuit

This is an open-source framework for quantum computing developed by Zhejiang University. 

**There will be a tutorial on Janus 3.0 in HPCA 2025 in Las Vagas, USA！**

## Install JanusQ

JanusQ can be installed by docker, wheel, and soruce code. Docker is recommended, as all functions have been tested on it. Linux (Ubuntu 22.04 latest) platform and Python (3.10) is prefered when instaling by wheel or soruce code.

### From docker (Recommended)

Pull docker image

```bash
docker pull janusq/janusq:latest
```

Run docker image

```bash
docker run -itd -p 8888:22 -p 9999:23 --name tutorial janusq/janusq
```

The docker can be accessed via

```bash
ssh root@localhost -p 8888
```

or

```bash
docker exec -it tutorial bash
```

The source code is in "/JanusQ/janusq" and examples can be directly run in "/JanusQ/examples/ipynb". The jupyter notebook can be visited in http://localhost:9999/lab.

### From wheel

Download [janusq.whl](https://github.com/JanusQ/JanusQ/blob/main/dist/janusq-0.1.0-py3-none-any.whl) and install with `pip`

```bash
pip install janusq.whl
```  

### From source code

Pull the source code from github and install the dependencies

```bash
git clone git@github.com:JanusQ/JanusQ.git
cd JanusQ
pip install -r requirements.txt
```  

Set up for HyQSAT

```bash
cd ./janusq/application/hyqsat/solver
cmake .
make install
```  

Set up for Choco-Q

- For Linux with CPU

```bash
  cd ./janusq/application/chocoq
  conda env create -f environment_linux_cpu.yml
  conda activate chocoq_cpu
  pip install .
```

- For Linux with GPU

```bash
  cd ./janusq/application/chocoq
  conda env create -f environment_linux_gpu.yml
  conda activate chocoq_qpu
  pip install .
```

- For MasOS

```bash
  cd ./janusq/application/chocoq
  conda env create -f environment_macos.yml
  conda activate chocoq
  pip install .
```

## Structure of JanusQ

- QuCT
  - Vectorization
    - "janusq/analysis/vectorization.py": Python script to generate the path table and vecterization of circuits, which serves as a upstream model for various analysis and optimization tasks.
  - Fidelity preidiction and optimization
    - "janusq/analysis/fidelity_prediction.py": Python script to train a downstream model for circuit fidelity prediction.
    - "janusq/optimizations/mapping/mapping_ct.py"：Python script of a typical compilation flow, including routing and scheduling. The compilation flow transforms the circuit to satisfy the processor topology so as to optimize the circuit fidelity based on the prediction model.
  - Unitary decomposition
    - "janusq/analysis/unitary_decompostion.py": Python script that takes a unitary as input and decomposes it into matrices of basic gates, resulting in an equivalent circuit.
- MorphQPV
- QuFEM
  - "janusq/optimizations/readout_mitigation/fem/benchmarking.py": Python script to generate a circuit for measuring calibration matrices.
  - "janusq/optimizations/readout_mitigation/fem/mitigation.py": Python script to implement iterative calibration, which includes quantifying interactions between qubits and constructing Bayesian networks.
  - "janusq/optimizations/readout_mitigation/fem/tools.py": Python script to implement data format conversion.
- HyQSAT
  - "janusq/hyqsat/common": Directory holding common functions about cnf files, such as "read_cnf.py".
  - "janusq/hyqsat/solver.py": Python script holding different APIs used to solve sat problem.
- Choco-Q


## Related paper

**[MICRO 2023] QuCT: A Framework for Analyzing Quantum Circuit by Extracting Contextual and Topological Features**  
Siwei Tan, Congliang Lang, Liang Xiang, Shudi Wang, Xinghui Jia, Ziqi Tan, Tingting Li (Zhejiang University), Jieming Yin (Nanjing University of Posts and Telecommunications), Yongheng Shang, Andre Python, Liqiang Lu, and Jianwei Yin (Zhejiang University)

**[ASPLOS 2024] MorphQPV: Exploiting Isomorphism in Quantum Programs to Facilitate Confident Verification**  
Siwei Tan, Debin Xiang, Liqiang Lu (Zhejiang University), Junlin Lu (Peking University), Qiuping Jiang (Ningbo University), Mingshuai Chen, and Jianwei Yin (Zhejiang University)

**[ASPLOS 2024] QuFEM: Fast and Accurate Quantum Readout Calibration Using the Finite Element Method**  
Siwei Tan, Hanyu Zhang, Jia Yu, Congliang Lang, Xinkui Zhao, Mingshuai Chen (Zhejiang University), Yun Liang (Peking University), Liqiang Lu, and Jianwei Yin (Zhejiang University)

**[HPCA 2023] HyQSAT: A Hybrid Approach for 3-SAT Problems by Integrating Quantum Annealer with CDCL**  
Siwei Tan, Mingqian Yu, Andre Python, Yongheng Shang, Tingting Li, Liqiang Lu, and Jianwei Yin (Zhejiang University)

**[HPCA 2025] Choco-Q: Commute Hamiltonian-based QAOA for Constrained Binary Optimization**  
Debin Xiang, Qifan Jiang, Liqiang Lu, Siwei Tan, and Jianwei Yin (Zhejiang University)

## Q&A

1. Error "Permission denied 'minisat_core'"

    Run following command

    ```bash
    chmod +x janusq/hyqsat/minisat_core
    ```

2. Error "Library not loaded: @rpath/libpython3.10.dylib" 

    Run following command

    ```bash
    install_name_tool -add_rpath /path/to/python3.10/lib janusq/hyqsat/minisat_core
    ```
