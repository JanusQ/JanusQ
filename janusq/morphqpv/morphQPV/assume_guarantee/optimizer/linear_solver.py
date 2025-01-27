import gurobipy as gp
from gurobipy import GRB
import numpy as np

def build_RI_state(parms,samples):
    return sum(parm * output.real for parm,output in zip(parms,samples)),\
        sum(parm * output.imag for parm,output in zip(parms,samples))
class linearOptimizer(gp.Model):
    def __init__(self,pramsize,name=None):
        super().__init__(name=name)
        parms = self.addMVar(shape=pramsize,lb=-1,ub=1,name='parms',vtype=GRB.CONTINUOUS)
    def add_constraint(self,predicate):
        pass
    def add_objective(self,predicate):
        pass

class LinearSolver(gp.Model):
    def __init__(self,name='linear_program'):
        super().__init__(name=name)
        pass

    def linear_cost_function_density(self,parms,inputs,outputs,real_density,target):
        """ 损失函数： 密度矩阵的重建误差
        Args:
            parms: 参数
            inputs: 输入的量子态
            outputs: 输出的量子态
            real_density: 真实的密度矩阵
            target: 优化的目标，'input'或者'output'
        """
        buildinputR,buildinputI = build_RI_state(parms,inputs)
        buildinputSqureR = buildinputR@buildinputR - buildinputI@buildinputI
        buildinputSqureI = buildinputR@buildinputI + buildinputI@buildinputR
        if target == 'input':
            self.addConstr(sum(buildinputR[i][i] for i in range(buildinputR.shape[0])) == 1)
            self.addConstr(sum(buildinputI[i][i] for i in range(buildinputI.shape[0])) == 0)
            self.setObjective(sum(sum(np.square(buildinputR-real_density.real)))\
                            + sum(sum(np.square(buildinputI-real_density.imag)))
            )
        elif target == 'output':
            self.addConstr(buildinputSqureI == buildinputI)
            self.addConstr(buildinputSqureR == buildinputR)
            self.addConstr(sum(buildinputR[i][i] for i in range(buildinputR.shape[0])) == 1)
            self.addConstr(sum(buildinputI[i][i] for i in range(buildinputI.shape[0])) == 0)
            buildoutputR, buildoutputI = build_RI_state(parms,outputs)
            self.setObjective(sum(sum(np.square(buildoutputR-real_density.real)))\
                            + sum(sum(np.square(buildoutputI-real_density.imag)))
            )
    def linear_cost_function_state(self,parms,inputs,outputs,real_state,target):
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
            self.addConstr(sum(np.square(buildoutputR[i])+ np.square(buildoutputI[i]) for i in range(buildoutputR.shape[0])) == 1)
            self.setObjective(sum(np.square(buildinputR-real_state.real))\
                            + sum(np.square(buildinputI-real_state.imag)))
        elif target == 'output':
            ## 保证build_input 归一化
            self.addConstr(sum(np.square(buildinputR[i])+ np.square(buildinputI[i]) for i in range(buildinputR.shape[0])) == 1)
            self.addConstr(sum(np.square(buildoutputR[i])+ np.square(buildoutputI[i]) for i in range(buildoutputR.shape[0])) == 1)
            self.setObjective(sum(sum(np.square(buildoutputR-real_state.real)))\
                            + sum(sum(np.square(buildoutputI-real_state.imag)))
            )
    
    def solve(self,inputs,outputs,real_target,target,jobname=None):
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
        self.Params.timeLimit = 500
        self.Params.LogToConsole = 0
        self.Params.LogFile = f'{jobname}.log'
        if len(inputs[0].shape) == 1:
            self.linear_cost_function_state(parms,inputs,outputs,real_target,target)
        else:
            self.linear_cost_function_density(parms,inputs,outputs,real_target,target)
        self.optimize()
        
        print(f'parmlength:{len(inputs)},iterations:{self.IterCount},obj:{self.objVal}')
        if self.status == GRB.OPTIMAL:
            return [parm.x for parm in parms]
        else:
            print('No solution')
            return None,None