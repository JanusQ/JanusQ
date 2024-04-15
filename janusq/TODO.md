# 需要做的事情 (3/15前)

## 代码相关的东西
    1. 代码
        pip install
    2. 文档
    3. tutorial （jupyter notebook）
    4. docker

## 总体 (贾星辉)
1. 把所有的print都换成logger, 分成info, warning, error
2. 统一配成一个docker
3. 给一个例子
4. 补充requirements，封装成可以pip install的包

## 注意 
1. jupyter 最后整理成一个网站类似 https://pennylane.ai/qml/demonstrations/
    1. 类似
        1. https://qiskit-extensions.github.io/qiskit-experiments/manuals/verification/randomized_benchmarking.html
        1. https://pennylane.ai/qml/demos/tutorial_odegen/
    2. 结构
        1. 哪篇paper，文章介绍
        2. 一步一步的执行，每一步包含
            1. 文章的section
            2. 背后的物理含义
            3. 有意思的事情
        3. Reference
        4. Version Information
2. logger 可以配置并且有专门保存的地方
    

## QuCT (郎聪亮)
### 保真度预测
相关的文件:
1. tests/test_fidelity_prediction_5q.py
2. tests/test_fidelity_prediction_18q.py
3. tests/test_optimization.py

TODO:
1. 加入RB，放到 baselines/fidelity_prediction
2. jupyter, 18比特真机 （对比QuCT， RB）, 复现论文 Figure 9 (a)
3. jupyter, 100比特模拟器, 复现论文 Figure 12 (a)
    1. 介绍weight的意思，通过fidelity_model.plot_path_error()，加入导出成excel，对比模拟器配置的error_paths，复现Figure 13 (b)
5. jupyter, 优化 复现 Figure 13 (b)

### 酉矩阵分解
TODO:
1. 解决下decomposition没法指定backend的问题
1. 加入ccd和qfast，放到 baselines/unitary_decompostion
2. jupyter, 分解随机 + 算法，对比ccd，qfast

### 整合
1. jupyter, 保真度优化+酉矩阵分解统一加入qiskit的transpile

## QuFEM （张涵禹）
相关的文件:
1. tests/test_readout_mitigation.py
2. optimizations/readout_mitigation/fem

TODO:
1. 加入IBU，放到baselines/readout_calibration
2. 加入IterativeSamplingProtocol (optimizations/readout_mitigation/fem/benchmarking.py)
3. jupyter，模拟器，介绍benchmarking、mitigation
4. jupyter，真机 136比特，分析

## HyQSAT (谭思危，周凯文)


## 拓扑时间晶体 （陶辰宁）