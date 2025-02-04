import numpy as np
from typing import List

from chocoq.utils import iprint

class DataAnalyzer():
    def __init__(self, *, collapse_state_lst: List, probs_lst: List, obj_func, best_cost, lin_constr_mtx):
        self.states_probs_zip = zip(collapse_state_lst, probs_lst)
        self.obj_func = obj_func
        self.best_cost = best_cost
        self.lin_constr_mtx = lin_constr_mtx
    
    def summary(self):
        best_cost = self.best_cost
        mean_cost = 0
        best_solution_probs = 0
        in_constraints_probs = 0

        iprint()
        for cs, pr in self.states_probs_zip:
            pcost = self.obj_func(cs)
            if pr >= 1e-3:
                iprint(f'{cs}: {pcost} ~ {pr}')
            if all([np.dot(cs,constr[:-1]) == constr[-1] for constr in self.lin_constr_mtx]):
                in_constraints_probs += pr
                if pcost == best_cost:
                    best_solution_probs += pr
            mean_cost += pcost * pr
        best_solution_probs *= 100
        in_constraints_probs *= 100
        # maxprobidex = np.argmax(probs)
        # max_prob_solution = collapse_state[maxprobidex]
        # cost = self.obj_dir * self.obj_func(max_prob_solution)
        ARG = abs((mean_cost - best_cost) / (best_cost + 1e-8))
        # iprint(f"max_prob_solution: {max_prob_solution}, cost: {cost}, max_prob: {probs[maxprobidex]:.2%}") #-
        iprint(f'\nbest_solution_probs: {best_solution_probs:.1f}')
        iprint(f'in_constraint_probs: {in_constraints_probs:.1f}')
        iprint(f'ARG: {ARG:.10f}')
        iprint(f"mean_cost: {mean_cost:.1f}\n")
        
        return [best_solution_probs, in_constraints_probs, ARG]