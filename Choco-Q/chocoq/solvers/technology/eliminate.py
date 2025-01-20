from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
from chocoq.solvers.abstract_solver import Solver
from chocoq.solvers.optimizers import Optimizer
from chocoq.solvers.qiskit.provider import Provider
from chocoq.solvers.options import ModelOption
import numpy as np
import gurobipy as gp
from typing import Type, Dict, List, Iterable

class Eliminate_variables():
    def __init__(
        self,
        *,
        prb_model: LcboModel,
        solver: Type[Solver],
        optimizer: Optimizer,
        provider: Provider,
        num_layers: int,
        shots: int = 1024,
        num_frozen_qubit: int,
    ):
        self.prb_model = prb_model  
        self.solver = solver
        self.optimizer = optimizer
        self.provider = provider
        self.num_layers = num_layers
        self.shots = shots
        self.num_frozen_qubit = num_frozen_qubit
        
        pass


    def solve(self):
        self.circuit_analyze = []
        best_cost = self.prb_model.best_cost
        num_frozen_qubit = self.num_frozen_qubit

        non_zero_counts = np.count_nonzero(self.prb_model.driver_bitstr, axis=0)
        sorted_indices = np.argsort(non_zero_counts)[::-1][:num_frozen_qubit]
        frozen_idx_list = sorted(sorted_indices)

        def find_feasible_solution_with_gurobi(A, fixed_values=None):
            num_vars = A.shape[1] - 1  # Number of variables
            # Create a new model
            model = gp.Model("feasible_solution")
            model.setParam('OutputFlag', 0)
            # Create variables
            variables = []
            for i in range(num_vars):
                variables.append(model.addVar(vtype=gp.GRB.BINARY, name=f"x{i}"))
            # Set objective (minimization problem, but no objective here since we just need a feasible solution)
            # Add constraints
            for i in range(A.shape[0]):
                lhs = gp.quicksum(A[i, j] * variables[j] for j in range(num_vars))
                rhs = A[i, -1]
                model.addConstr(lhs == rhs)
                
            # Add fixed values constraints
            if fixed_values:
                idx, fix = fixed_values
                for i, f in zip(idx, fix):
                    model.addConstr(variables[i] == f)
            
            model.optimize()
            
            if model.status == gp.GRB.Status.OPTIMAL:
                # Retrieve solution
                solution = [int(variables[i].x) for i in range(num_vars)]
                return solution
            else:
                return None
        
        ARG_list = []
        in_constraints_probs_list = []
        best_solution_probs_list = []
        iteration_count_list = []

        for i in range(2**num_frozen_qubit):
            frozen_state_list = [int(j) for j in list(bin(i)[2:].zfill(num_frozen_qubit))]
            elimi_linear_constraints = self.prb_model.lin_constr_mtx.copy()
            for idx, state in zip(frozen_idx_list, frozen_state_list):
                elimi_linear_constraints[:,-1] -= elimi_linear_constraints[:, idx] * state
            elimi_linear_constraints = np.delete(elimi_linear_constraints, frozen_idx_list, axis=1)
            elimi_feasible_solution = find_feasible_solution_with_gurobi(elimi_linear_constraints)
            
            if elimi_feasible_solution is None:
                continue
        

            def process_obj_dct(obj_dct: Dict, frozen_idx_list, frozen_state_list):
                obj_tmp_dct: Dict[int, List] = {}
                zero_indices = [frozen_idx_list[i] for i, x in enumerate(frozen_state_list) if x == 0]
                nonzero_indices = [i for i in frozen_idx_list if i not in zero_indices]
                frozen_idx_list = np.array(frozen_idx_list)
                for dimension in obj_dct:
                    for objective_term in obj_dct[dimension]:
                        if any(idx in objective_term[0] for idx in zero_indices):
                            # 如果 frozen_state == 0 且 x 在内层列表中，移除整个iterm
                            continue
                        else:
                            # 如果 frozen_state == 1 且 x 在内层列表中，移除iterm中的 x
                            iterm = [varbs for varbs in objective_term[0] if varbs not in nonzero_indices]
                            iterm = [x - np.sum(frozen_idx_list < x) for x in iterm]
                            if iterm:
                                tuple_len = len(iterm)
                                if tuple_len == 0:
                                    continue
                                if tuple_len not in obj_tmp_dct:
                                    obj_tmp_dct[tuple_len] = []
                                obj_tmp_dct[tuple_len].append((iterm, objective_term[1]))
                return obj_tmp_dct
            
            elimi_obj_dct = process_obj_dct(self.prb_model.obj_dct, frozen_idx_list, frozen_state_list)
            num_qubits = len(self.prb_model.variables) - len(frozen_idx_list)
            obj_dir = self.prb_model.obj_dir
            from chocoq.utils.linear_system import to_row_echelon_form

            from chocoq.utils.linear_system import find_basic_solution
            elimi_Hd_bitstr_list = find_basic_solution(elimi_linear_constraints[:,:-1]) if len(elimi_linear_constraints) > 0 else []
            
            def elimi_objective_func_map(origin_func):
                def elimi_objective_func(variables: Iterable):
                    def insert_states(filtered_list, idx_list, state_list):
                        result = []
                        state_index = 0
                        filtered_index = 0

                        for i in range(len(filtered_list) + len(state_list)):
                            if i in idx_list:
                                result.append(state_list[state_index])
                                state_index += 1
                            else:
                                result.append(filtered_list[filtered_index])
                                filtered_index += 1
                        return result
                    return origin_func(insert_states(variables, frozen_idx_list, frozen_state_list))
                return elimi_objective_func
            
            elimi_obj_func = elimi_objective_func_map(self.prb_model.obj_func)
            elimi_Hd_bitstr_list = to_row_echelon_form(elimi_Hd_bitstr_list)
            
            if len(elimi_Hd_bitstr_list) == 0:
                cost = obj_dir * elimi_obj_func(elimi_feasible_solution)
                ARG = abs((cost - best_cost) / best_cost)
                best_solution_probs = 100 if cost == best_cost else 0
                in_constraints_probs = 100 if all([np.dot(elimi_feasible_solution, constr[:-1]) == constr[-1] for constr in elimi_linear_constraints]) else 0
                ARG_list.append(ARG)
                best_solution_probs_list.append(best_solution_probs)
                in_constraints_probs_list.append(in_constraints_probs)
                iteration_count_list.append(0)
                continue

            model_option = ModelOption(
                num_qubits=num_qubits,
                penalty_lambda=self.prb_model.penalty_lambda,
                feasible_state = elimi_feasible_solution,
                obj_dct=elimi_obj_dct,
                lin_constr_mtx=elimi_linear_constraints,
                Hd_bitstr_list=elimi_Hd_bitstr_list,
                obj_dir=obj_dir,
                obj_func=elimi_obj_func,
                best_cost=best_cost
            )
            elimi_solver = self.solver(
                prb_model=model_option,
                optimizer=self.optimizer,
                provider=self.provider,
                num_layers=self.num_layers,
                shots=self.shots
            )
            self.circuit_analyze.extend(elimi_solver.circuit_analyze(['culled_depth']))
            elimi_solver.solve()
            u, v, w, x = elimi_solver.evaluation()
            best_solution_probs_list.append(u)
            in_constraints_probs_list.append(v)
            ARG_list.append(w)
            iteration_count_list.append(x)

        self.best_solution_probs_list = best_solution_probs_list
        self.in_constraints_probs_list = in_constraints_probs_list
        self.ARG_list = ARG_list
        self.iteration_count_list = iteration_count_list
        return best_solution_probs_list, in_constraints_probs_list, ARG_list, iteration_count_list
    
    def evaluation(self):
        return max(self.best_solution_probs_list)
    
    def depth(self):
        return sum(self.circuit_analyze) / len(self.circuit_analyze)
    