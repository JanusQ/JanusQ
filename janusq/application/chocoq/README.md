# Choco

## Installation

You can install the virtual environment in three versions:

- **CPU version**: Simulates quantum circuits using the CPU.
- **GPU version**: Simulates quantum circuits using the GPU (requires CUDA 12 support).
- **macOS version**: For users running macOS.

### CPU version (Ubuntu)
1. Install the envirnoment by conda
```bash
conda env create -f environment_linux_cpu.yml
```
2. activate the installed virtual environment.
```bash
conda activate chocoq_cpu
```
If the installation is interrupted or fails, you need to delete the environment and reinstall: 
```bash
conda remove -n chocoq_cpu --all
```


### GPU version （Ubuntu, need the support of CUDA12）
1. Install the envirnoment by conda
```bash
conda env create -f environment_linux_gpu.yml
```
2. activate the installed virtual environment.
```bash
conda activate choco_gpu
```
If the nextwork is poor, the installation time for GPU will be longer, please be patient.
When the installation is interrupted or fails, you need to delete the environment and reinstall: 
```bash
conda remove -n choco_gpu --all
```

### macOS version
1. Install the environment using conda:
```bash
conda env create -f environment_macos.yml
```
2. Activate the installed virtual environment:
```bash
conda activate chocoq
```
If the installation is interrupted or fails, you need to delete the environment and reinstall: 
```bash
conda remove -n chocoq --all
```

## Test for installation
Run corresponding test files according to the installation version:

```bash
python testbed_cpu.py
```
for CPU version or 
```bash
python testbed_gpu.py
```
for GPU version.

If you see "Environment configuration is successful!" it means the installation is successful.

If the installation fails, you may consider:

1. Ensure that the correct conda environment is activated in the terminal.
2. After switching environments, execute `pip install -e .` under `Choco-Q/` to install the chocoq package.
2. Make sure the Python execution environment is set to the corresponding conda environment. you may disable the user site by 
```bash
 export PYTHONNOUSERSITE=1
```

## Reproduce the experiments by Notebooks
1. [implementations/0_test.ipynb](implementations/0_test.ipynb) - Custom optimization problem and solve it.

2. [implementations/1_table.ipynb](implementations/1_table.ipynb) - Table 1 in the paper.