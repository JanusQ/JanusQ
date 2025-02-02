## MorphQPV
language: [English](../README.md) | [中文](README.zh-CN.md)

MorphQPV 是一个用于量子计算的自信断言验证工具。它提供了一个分析和验证量子电路的框架，使用了一种新的形式主义。它定义了一个断言语句，其中包含假设保证原语和跟踪点编译指示来标记目标量子状态。然后，我们可以使用基于同构的近似来表征状态之间的真实关系，这可以在避免重复执行的同时有效地获得各种输入下的程序状态。最后，验证被制定为一个约束优化问题，其中包含置信度估计模型，以实现严格的分析。
## Getting Started
### Installation
```bash
conda create -n morphenv python=3.9
conda activate morphenv
python -m pip install -r requirements.txt
```
## Example

```python
from morphQPV import MorphQC
from morphQPV import IsPure,Equal,NotEqual
import numpy as np
with MorphQC() as morphQC:
    ### morphQC is a quantum circuit, the gate is applyed to the qubits in the order of the list
    ## we can add tracepoint to label the quantum state
    morphQC.add_tracepoint(0,1,2) ## the state after the first 3 qubits is labeled as tracepoint 0
    morphQC.assume(0,IsPure()) ## the state in tracepoint 0 is assumed to be pure
    morphQC.x([1,3]) ## apply x gate to  qubit 1 and 3
    morphQC.y([0,1,2])  ## apply y gate to qubit 0,1,2
    for i in range(4):
        morphQC.cnot([i, i+1]) ## apply cnot gate to qubit i and i+1
    morphQC.s([0,2,4]) ## apply s gate to qubit 0,2,4
    morphQC.add_tracepoint(2,4) ## the state after qubit 2 and 4 is labeled as tracepoint 1
    morphQC.assume(1,IsPure())  ## the state in tracepoint 1 is assumed to be pure
    morphQC.rz([0,1,2,3,4],np.pi/3) ## apply rz gate to qubit 0,1,2,3,4
    morphQC.h([0,1,2,3,4]) ## apply h gate to qubit 0,1,2,3,4
    morphQC.rx([0,1,2,3,4],np.pi/3) ## apply rx(pi/3) gate to qubit 0,1,2,3,4
    morphQC.ry([0,1,2,3,4],np.pi/3) ## apply ry(pi/3) gate to qubit 0,1,2,3,4
    morphQC.add_tracepoint(0,3) ## the state after qubit 0 and 3 is labeled as tracepoint 2
    morphQC.assume(2,IsPure()) ## the state in tracepoint 2 is assumed to be pure
    morphQC.guarantee([1,2],Equal()) ## the state in tracepoint 1 and 2 are guaranteed to be equal
print(morphQC.assertion) ## print the assertion statement and verify result
```
程序文件 `main_exp.py` 提供了一个更复杂的示例。在验证的每个阶段都有超参数。用户可以在[config](doc/morphconfig.md)中查看详细信息。
## License
项目使用 MIT License 授权 - 详情请参阅 [LICENSE.md](LICENSE.md) 文件
```
