# MorphQC Configuration
The configuration of morphQC is implemented by a `Config` class in [morphQPV/assume_guarantee/config.py](../morphQPV/assume_guarantee/config.py), It can be used like this:
```python
from morphQPV import MorphQC,Config
myconfig = Config()
myconfig.solver = 'lgd'
myconfig.steps = 1000
myconfig.step_size = 0.01
myconfig.min_weight = 0.01
with MorphQC(config= myconfig) as qc:
    ...
```
The `Config` class includes the following parts:
- [General](#general)
  - solver: str = 'sgd' 
    > the optimization solver, can be 'sgd' or 'lgd'
  - base_num: int = 100 
    > the number of samples to approximate the quantum state
  - device: str = 'simulate' 
    > the device to run the input sampling, can be 'simulate' for simulation or 'ibmq' for real quantum device

- [configuration for solver](#tracepoint)
  - max_weight: int = 100 
    > the weight for larger penalty
  - high_weight: int = 10 
    > the weight for high penalty
  - min_weight: float = 0.01 
    > the minimum weight for the penalty
  - step_size: float = 0.01 
    > the step size for optimization solver, i.e., learning rate
  - steps: int = 1000 
    > the number of steps for optimization solver
  - early_stopping_iter: int = 100 
    > the number of steps for early stop
- [configuration for sampling](#tracepoint)
  - base_num: int = 100 
    > the number of samples to approximate the quantum state
  - device: str = 'simulate'
    > the device to run the input sampling, can be 'simulate' for simulation or 'ibmq' for real quantum device

