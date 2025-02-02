import gurobipy as gp
from gurobipy import GRB
import numpy as np

def build_RI_state(parms,states):
    return sum(parm * state.real for parm,state in zip(parms,states)),\
        sum(parm * state.imag for parm,state in zip(parms,states))

class LinearSolver(gp.Model):
    def __init__(self,name='linear_program'):
        super().__init__()
        pass

    
    def linear_cost_function_state(self,parms,inputs,outputs,real_state,target='output'):
        """ 损失函数： 量子态的重建误差
        Args:
            parms: 参数
            inputs: 输入的量子态
            outputs: 输出的量子态
            real_state: 真实的量子态
            target: 优化的目标，'input'或者'output'
        """
        buildinputR, buildinputI = build_RI_state(parms,inputs)
        buildoutputR, buildoutputI = build_RI_state(parms,outputs)
        if target == 'input':
            self.setObjective(sum(np.square(buildinputR-real_state.real))\
                            + sum(np.square(buildinputI-real_state.imag)))
        elif target == 'output':
            ## 保证build_input 归一化
            self.addConstr(sum(np.square(buildinputR[i]) for i in range(buildinputR.shape[0])) == 1)
            self.addConstr(sum(np.square(buildoutputR[i]) for i in range(buildoutputR.shape[0])) == 1)
            self.setObjective(np.square(buildoutputR[0]))
    
    def solve(self,inputs,outputs,real_target,target='input'):
        """ 求解线性规划
        Args:
            inputs: 输入的量子态
            outputs: 输出的量子态
            real_target: 真实的量子态或者密度矩阵
            target: 优化的目标，'input'或者'output'
        """
        parms = self.addMVar(shape=len(inputs),lb=-1,ub=1,name='parms',vtype=GRB.CONTINUOUS)
        ## add constraints
        self.Params.NonConvex = 2
        self.Params.timeLimit = 6000
        self.Params.BarConvTol = 1e-1
        if len(inputs[0].shape) == 1:
            self.linear_cost_function_state(parms,inputs,outputs,real_target,target)
        else:
            self.linear_cost_function_density(parms,inputs,outputs,real_target,target)
        self.optimize()
        
        if self.status == GRB.OPTIMAL:
            return [parm.x for parm in parms]
        else:
            print('No solution')
            return None,None