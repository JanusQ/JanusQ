
import numpy as np

class InnerProductOptimizer:
    def __init__(self,name='exact_solver'):
        super().__init__()
        pass
    
    def solve(self,inputs,outputs,real_target,target='input'):
        """ 对于正交的基态，可以通过计算内积来求解 
        定理： 如果输入态正交，那么输出态也是正交的
        Args:
            inputs: 输入的量子态
            outputs: 输出的量子态
            real_target: 真实的量子态或者密度矩阵
            target: 优化的目标，'input'或者'output'
        """
        if len(inputs[0])==1:
            ## vector
            if target == 'input':
                return list(map(lambda x: np.dot(x,real_target),inputs))
            elif target == 'output':
                return list(map(lambda x: np.dot(x,real_target),outputs))
        else:
            ## matrix
            if target == 'input':
                return list(map(lambda x: np.trace(x@real_target),inputs))
            elif target == 'output':
                return list(map(lambda x: np.trace(x@real_target),outputs))

    