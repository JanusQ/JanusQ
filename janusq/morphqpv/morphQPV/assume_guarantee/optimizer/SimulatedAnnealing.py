""" 
    Using Simulated Annealing for continuous variable problem
    
"""
import numpy as np
from  tqdm  import tqdm
from numpy.random import rand, uniform

def corana_update_v(vi, ai, ns,ci =2):
    if ai > 0.6 * ns:
        vi *= (1 + ci * (ai/ns - 0.6) / 0.4)
    elif ai < 0.4 * ns:
        vi /= (1 + ci * (0.4 - ai/ns) / 0.4)
    return vi

class SimulatedAnnealing:
    def __init__(self, ns=20, nε=4, γ=0.85, ) -> None:
        
        self.t = 10
        self.ε = 0.001
        self.ns = ns
        self.nε = nε
        self.γ = γ
        pass
    def corana_update(self,a):
        return np.array(list(map(lambda x:corana_update_v(x[0],x[1],self.ns),zip(self.v,a))))
    
    def basis_update(self,a,i,cost_function,parms,y,*args):
        parms_new = parms.copy()
        parms_new[i] = parms[i] + self.v[i] * (2*rand()-1)
        y_new = cost_function(parms_new,*args)
        Δy = y_new - y 
        if Δy < 0 or rand() < np.exp(-Δy/self.t):
            parms = parms_new
            y = y_new
            a[i] += 1
            if y_new < self.y_best:
                self.parms_best = parms_new
                self.y_best = y_new
        
        return parms,y,a
    def random_basis_update(self,cost_function,a,parms,y,*args):
        i = np.random.randint(0,len(a))
        return self.basis_update(a,i,cost_function,parms,y,*args)
    
    def linear_update(self,cost_function,a,parms,y,*args):
        update_diff = self.v * (2*rand()-1)
        parms_new = parms + update_diff
        y_new = cost_function(parms_new,*args)
        Δy = y_new - y 
        if Δy < 0 or rand() < np.exp(-Δy/self.t):
            parms = parms_new
            y = y_new
            a += np.abs(update_diff)/np.max(np.abs(update_diff))
            if y_new < self.y_best:
                self.parms_best = parms_new
                self.y_best = y_new
        return parms,y,a
    
    def optimize_params(self,param_size,cost_function,*args):
        """ 优化器
        Args:
            inputs: 输入的量子态
            outputs: 重建的量子态
            real_target: 真实的量子态
            cost_function: 损失函数
        """
        self.nt=min(100, 5*param_size)

        self.c =np.full(param_size, 2)
        self.v =np.full(param_size, 0.5)
        # 采用模拟退火算法
        # 1. 初始化参数
        parms = np.random.rand(param_size)
        y = cost_function(parms,*args)
        self.parms_best = parms.copy()
        self.y_best = y
        
        y_arr = []
        n = param_size
        a = np.zeros(n)
        counts_cycles = 0
        counts_resets = 0
        with tqdm(total=100000) as pbar:
            while True:
                # for i in range(n):
                #     self.parms_best,self.y_best,y,a = self.basis_update(a,i,cost_function,parms,y,*args)
                for _ in range(min(n,100)):
                    parms,y,a = self.random_basis_update(cost_function,a,parms,y,*args)
                # for _ in range(min(n,100)):
                #     parms,y,a = self.linear_update(cost_function,a,parms,y,*args)

                counts_cycles += 1
                if counts_cycles < self.ns:
                    continue
                
                counts_cycles = 0
                self.v = self.corana_update(a) 
                a[:] = 0
                counts_resets += 1
                if counts_resets < self.nt:
                    continue

                self.t *= self.γ
                counts_resets = 0
                y_arr.append(y)
                pbar.update(1)
                pbar.set_postfix_str(f'loss = {y}, best_loss = {self.y_best}')
                if len(y_arr) > self.nε and y_arr[-1] - self.y_best <= self.ε and np.all(np.abs(y_arr[-1] - y_arr[-u]) <= self.ε for u in range(1, self.nε+1)):
                    y = self.y_best
                    parms = self.parms_best
                    break
                
        return self.parms_best