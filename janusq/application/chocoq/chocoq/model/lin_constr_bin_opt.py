import numpy as np
from typing import Dict, Tuple, Set, List, Union

from chocoq.model.model import Variable, Expression, Constraint, Model
from chocoq.utils import set_print_form
from chocoq.utils import iprint
from chocoq.utils.errors import QuickFeedbackException
from chocoq.utils.linear_system import find_basic_solution
from chocoq.solvers.options import CircuitOption, OptimizerOption, ModelOption

class LinearConstrainedBinaryOptimization(Model):
    def __init__(self):
        """ a linear constrainted binary optimization problem. """
        set_print_form()
        super().__init__()
        self.slack_groups = 0
        self.obj_dir = None
        self.penalty_lambda = 0x7FFF # 大数
        self.additional_slack_constraints: List[Tuple[Constraint, Dict[Variable]]] = [] # 约束和松弛变量
        

        self._variables_idx = None
        self._driver_bitstr = None
        self._obj_func = None
        self._lin_constr_mtx: np.ndarray = None  # yeld by @property 是 for_cyclic 和 for_others 的并集
        self._obj_dct: Dict[int, List] = None
        self._best_cost = None


        # self._constraints_classify_cyclic_others  = None # cyclic 用于存∑=x
        # self.objective_func_term_list = [[], []] # 暂设目标函数最高次为2, 任意次的子类自行重载, 解释见 statement

        # self.collapse_state = None
        # self.probs = None
    def __repr__(self):
        repr = super().__repr__()
        return (
            f"{repr}"
            f"penalty_lambda:\n{self.penalty_lambda}\n\n"
        )

    
    def update(self):
        """设定内部数据结构为None, 实现调用 @property 时重新生成"""
        self._variables_idx = None
        self._driver_bitstr = None
        self._obj_func = None
        self._lin_constr_mtx = None
        self._obj_dct = None
        self._best_cost = None
    
    @property
    def variables_idx(self) -> Dict[str, int]:
        if self._variables_idx is None:
            idx = 0
            idx_dict = {}
            for var in self.variables:
                idx_dict[var.name] = idx
                idx += 1

            self._variables_idx = idx_dict

        return self._variables_idx

    @property
    def lin_constr_mtx(self):
        # 子类自建linear_constraint 再分类到 for_cyclic & for_others
        if self._lin_constr_mtx is None:
            vct_len = len(self.variables) + 1
            lin_constr_lst = []
            for constr in self.constraints:
                constr_vct = [0] * vct_len
                for var_tuple, coeff in constr.expr.terms.items():
                    if len(var_tuple) != 1:
                        continue
                    # 线性约束，只有一个var在tuple中
                    index = self.var_idx(var_tuple[0])
                    constr_vct[index] = coeff
                constr_vct[-1] = constr.rhs
                lin_constr_lst.append(constr_vct)
                
            self._lin_constr_mtx = np.array(lin_constr_lst)
        
        return self._lin_constr_mtx
    
    @property
    def obj_dct(self):
        if self._obj_dct is None:
            obj_tmp_dct: Dict[int, List] = {}
            for vars_tuple, coeff in self.objective.terms.items():
                tuple_len = len(vars_tuple)
                # 常数项无需编码
                if tuple_len == 0:
                    continue
                if tuple_len not in obj_tmp_dct:
                    obj_tmp_dct[tuple_len] = []

                indexs = []
                for var in vars_tuple:
                    index = self.var_idx(var)
                    indexs.append(index)

                obj_tmp_dct[tuple_len].append((indexs, coeff))

            self._obj_dct = obj_tmp_dct
        
        return self._obj_dct

    def get_driver_bitstr(self):
        # 如果子类有快速求解约束方程方案，override 是推荐的
        return find_basic_solution(self.lin_constr_mtx[:,:-1]) if len(self.lin_constr_mtx) > 0 else []

    @property
    def driver_bitstr(self):
        if self._driver_bitstr is None:
            self._driver_bitstr = self.get_driver_bitstr()

        return self._driver_bitstr

    def generate_obj_function(self) -> callable:
        def obj_function(values: List[Union[int, float]]) -> Union[int, float]:
            obj_pnt_expr = self.objective
            for constr in self.constraints:
                obj_pnt_expr += self.obj_dir * self.penalty_lambda * (constr.expr - constr.rhs) ** 2
                # obj_pnt_expr += self.obj_dir * self.penalty_lambda / len(self.constraints) ** 2 * (constr.expr - constr.rhs) ** 2

            result = 0
            for vars_tuple, coeff in obj_pnt_expr.terms.items():
                term_value = coeff
                for var in vars_tuple:
                    index = self.var_idx(var)
                    term_value *= values[index]
                result += term_value
                
            return result
        
        return obj_function
    
    @property
    def obj_func(self) -> callable:
        if self._obj_func is None:
            self._obj_func = self.generate_obj_function()

        return self._obj_func
    # def set_optimization_direction(self, dir):

    def setObjective(self, expression, sense):
        assert sense in ['min', 'max']
        super().setObjective(expression, sense)
        self.obj_dir = 1 if self.obj_sense == 'min' else -1

    # def find_state_probability(self, state):
    
    def set_penalty_lambda(self, penalty_lambda = None):
        self.penalty_lambda = penalty_lambda

    def var_idx(self, variable: Variable):
        """输入变量对象, 输出变量在问题对象变量列表中的下标, 算objective等用"""
        try:
            return self.variables_idx[variable.name]
        except ValueError:
            raise ValueError(f"Variable {variable} does not exist in the model.")
        

    def addConstr(self, constraint: Constraint):
        """自动为不等式约束添加松弛变量，转换为等式约束"""
        if constraint.sense in ['<=', '>=']:
            is_greater_equal = constraint.sense == '>='
            emin = constraint.expr.min_for_lin()
            emax = constraint.expr.max_for_lin()

            if is_greater_equal:
                constraint.rhs = max(emin, constraint.rhs)
                emin = constraint.rhs
            else:
                constraint.rhs = min(emax, constraint.rhs)
                emax = constraint.rhs

            diff = emax - emin
            slack_vars = self.addVars(diff, name=f'slk_{self.slack_groups}')
            self.slack_groups += 1

            if is_greater_equal:
                constraint.expr -= sum(slack_vars.values())
            else:
                constraint.expr += sum(slack_vars.values())

            constraint.sense = '=='
            self.additional_slack_constraints.append((constraint, slack_vars))

        self.constraints.append(constraint)

    def get_feasible_solution(self):
        # 如果子类有快速 generate 可行解方案，override 是推荐的
        for i in range(1 << len(self.variables)):
            bitstr = [int(j) for j in list(bin(i)[2:].zfill(len(self.variables)))]
            if all([np.dot(bitstr,constr[:-1]) == constr[-1] for constr in self.lin_constr_mtx]):
                return bitstr
        raise RuntimeError("找不到可行解") 

    def fill_feasible_solution(self, fsb_lst: List):
        """ 根据变量的选择，自动填补松弛变量 """
        for cnst, slack_vars in self.additional_slack_constraints:
            rhs = cnst.rhs
            expr = cnst.expr
            
            # 带入松弛后的等式约束，合并到右值
            for vars_tuple, coeff in expr.terms.items():
                # 用非松弛变量计算，含松弛变量的项跳过，这里默认松弛变量是单独的项
                if any(var.name in [slack_var.name for slack_var in slack_vars.values()] for var in vars_tuple):
                    continue
                temp = 1
                for var in vars_tuple:
                    temp *= fsb_lst[self.var_idx(var)]
                rhs -= temp * coeff
            # 将松弛变量根据右值赋 1 以满足等式，等量个数就行（绝对值整数）
            for key in range(np.abs(rhs).astype(int)):
                fsb_lst[self.var_idx(slack_vars[key])] = 1

    def calculate_feasible_solution(self):
        count = 0
        from tqdm import tqdm
        with tqdm(total=1 << len(self.variables)) as pbar:
            for i in range(1 << len(self.variables)):
                bitstr = [int(j) for j in list(bin(i)[2:].zfill(len(self.variables)))]
                if all([np.dot(bitstr,constr[:-1]) == constr[-1] for constr in self.lin_constr_mtx]):
                    count += 1
                pbar.update(1)
        return count

    
    @property
    def best_cost(self):
        if self._best_cost is None:
            best_cost, best_solution_case = self.optimize_with_gurobi()
            iprint(f'best_cost: {best_cost}')
            iprint(f'best_solution_case: {list(best_solution_case.values())}\n')
            self._best_cost = best_cost
        return self._best_cost
    
    def to_model_option(self) -> ModelOption:
        model_option = ModelOption(
            num_qubits = len(self.variables),
            penalty_lambda = self.penalty_lambda,
            feasible_state = self.get_feasible_solution(),
            obj_dct = self.obj_dct,
            lin_constr_mtx = self.lin_constr_mtx,
            Hd_bitstr_list = self.driver_bitstr,
            obj_dir=self.obj_dir,
            obj_func = self.obj_func,
            best_cost=self.best_cost
        )
        return model_option

    def optimize(self):
        """"""
        return self.optimize_with_gurobi()
        

        # collapse_state_str = [''.join([str(x) for x in state]) for state in collapse_state]
        # iprint(dict(zip(collapse_state_str, probs)))

    #     circuit_option.algorithm_optimization_method = self.algorithm_optimization_method

    #     iprint(f'fsb_state: {circuit_option.feasiable_state}') #-
    #     iprint(f'driver_bit_stirng:\n {self.driver_bitstr}') #-
    #     objective_func_map = {
    #         'penalty': self.objective_penalty,
    #         'cyclic': self.objective_cyclic,
    #         'commute': self.objective_commute,
    #         'HEA': self.objective_penalty
    #     }
    #     if self.algorithm_optimization_method in objective_func_map:
    #         circuit_option.objective_func = objective_func_map.get(self.algorithm_optimization_method)
    #     try:
    #         collapse_state, probs, iteration_count = solve(optimizer_option, circuit_option)
    #     except QuickFeedbackException as qfe:
    #         return qfe.data
    # ----------------------------



    # @property
    # def dctm_driver_bitstr(self):
    #     return find_basic_solution(self.dctm_linear_constraints[:,:-1]) if len(self.dctm_linear_constraints) > 0 else []

    # def dichotomy_optimize(self, optimizer_option: OptimizerOption, circuit_option: CircuitOption, num_frozen_qubit: int = 1) -> None: 
    #     self.opt_mtd = 'dichotomy'
    #     self.num_frozen_qubit = num_frozen_qubit
    #     iprint(self.driver_bitstr)
    #     iprint()
    #     # 最多非零元素的列索引, 对该比特冻结 | 注意, 不是约束最多的列，是driver_bitstr最多的列
    #     iprint(self.driver_bitstr)
    #     non_zero_counts = np.count_nonzero(self.driver_bitstr, axis=0)
    #     sorted_indices = np.argsort(non_zero_counts)[::-1][:num_frozen_qubit]
    #     self.frozen_idx_list = sorted(sorted_indices)
    #     # self.frozen_idx_list = [1, 4]

    #     def find_feasible_solution_with_gurobi(A, fixed_values=None):
    #         num_vars = A.shape[1] - 1  # Number of variables
    #         # Create a new model
    #         model = gp.Model("feasible_solution")
    #         model.setParam('OutputFlag', 0)
    #         # Create variables
    #         variables = []
    #         for i in range(num_vars):
    #             variables.append(model.addVar(vtype=gp.GRB.BINARY, name=f"x{i}"))
    #         # Set objective (minimization problem, but no objective here since we just need a feasible solution)
    #         # Add constraints
    #         for i in range(A.shape[0]):
    #             lhs = gp.quicksum(A[i, j] * variables[j] for j in range(num_vars))
    #             rhs = A[i, -1]
    #             model.addConstr(lhs == rhs)
                
    #         # Add fixed values constraints
    #         if fixed_values:
    #             idx, fix = fixed_values
    #             for i, f in zip(idx, fix):
    #                 model.addConstr(variables[i] == f)
            
    #         model.optimize()
            
    #         if model.status == gp.GRB.Status.OPTIMAL:
    #             # Retrieve solution
    #             solution = [int(variables[i].x) for i in range(num_vars)]
    #             return solution
    #         else:
    #             return None
    #     ARG_list = []
    #     in_constraints_probs_list = []
    #     best_solution_probs_list = []
    #     iteration_count_list = []
    #     objective_func_map = {
    #         'penalty': self.objective_penalty,
    #         'cyclic': self.objective_cyclic,
    #         'commute': self.objective_commute,
    #         'HEA': self.objective_penalty
    #     }
    #     # 找到最优解的cost / by groubi
    #     best_cost = self.get_best_cost()
    #     for i in range(2**num_frozen_qubit):
    #         self.frozen_state_list = [int(j) for j in list(bin(i)[2:].zfill(num_frozen_qubit))]
    #         # 调整约束矩阵 1 修改常数列(c - frozen_state * frozen_idx), 2 剔除 frozen 列
    #         dctm_linear_constraints = self.linear_constraints.copy()
    #         # iprint(f'self.linear_constraints:\n {self.linear_constraints}')
    #         for idx, state in zip(self.frozen_idx_list, self.frozen_state_list):
    #             dctm_linear_constraints[:,-1] -= dctm_linear_constraints[:, idx] * state
    #         # print("old\n", self.linear_constraints)
    #         self.dctm_linear_constraints = np.delete(dctm_linear_constraints, self.frozen_idx_list, axis=1)
    #         # iprint(f'self.dichotomy_linear_constraints:\n {self.dctm_linear_constraints}')
    #         dctm_feasible_solution = find_feasible_solution_with_gurobi(self.dctm_linear_constraints)
    #         if dctm_feasible_solution is None:
    #             continue
    #         # circuit_option.feasiable_state = np.delete(self.get_feasible_solution(), self.frozen_idx, axis=0)
    #         circuit_option.feasiable_state = dctm_feasible_solution

    #         circuit_option.num_qubits = len(self.variables) - len(self.frozen_idx_list)
    #         circuit_option.algorithm_optimization_method = self.algorithm_optimization_method
    #         circuit_option.penalty_lambda = self.penalty_lambda
    #         #+++ 这样只冻结了一种形态, 另一种形态待补
    #         iprint(f"frzoen_idx_list{self.frozen_idx_list}")
    #         iprint(f"frzoen_state_list{self.frozen_state_list}")
    #         iprint("feasible:", circuit_option.feasiable_state)
    #         # 处理剔除 frozen_qubit 后的目标函数    
    #         def process_objective_term_list(objective_iterm_list, frozen_idx_list, frozen_state_list):
    #             process_list = []
    #             zero_indices = [frozen_idx_list[i] for i, x in enumerate(frozen_state_list) if x == 0]
    #             nonzero_indices = [i for i in frozen_idx_list if i not in zero_indices]
    #             frozen_idx_list = np.array(frozen_idx_list)
    #             for dimension in objective_iterm_list:
    #                 dimension_list = []
    #                 for objective_term in dimension:
    #                     if any(idx in objective_term[0] for idx in zero_indices):
    #                         # 如果 frozen_state == 0 且 x 在内层列表中，移除整个iterm
    #                         continue
    #                     else:
    #                         # 如果 frozen_state == 1 且 x 在内层列表中，移除iterm中的 x
    #                         iterm = [varbs for varbs in objective_term[0] if varbs not in nonzero_indices]
    #                         iterm = [x - np.sum(frozen_idx_list < x) for x in iterm]
    #                         if iterm:
    #                             dimension_list.append((iterm, objective_term[1]))
    #                 # 空维度也要占位
    #                 process_list.append(dimension_list)
    #             return process_list
    #         circuit_option.objective_func_term_list = process_objective_term_list(self.objective_func_term_list, self.frozen_idx_list, self.frozen_state_list)
    #         # iprint('term_list', circuit_option.objective_func_term_list)
    #         circuit_option.constraints_for_cyclic = self.constraints_classify_cyclic_others[0]
    #         circuit_option.constraints_for_others = self.constraints_classify_cyclic_others[1]
    #         circuit_option.Hd_bits_list = to_row_echelon_form(self.dctm_driver_bitstr)
    #         # iprint(f'dctm_driver_bitstr:\n{self.dctm_driver_bitstr}') #-
    #         # iprint(f'Hd_bits_list:\n{circuit_option.Hd_bits_list}') #-

    #         def dctm_objective_func_map(method: str):
    #             def dctm_objective_func(variables: Iterable):
    #                 def insert_states(filtered_list, idx_list, state_list):
    #                     result = []
    #                     state_index = 0
    #                     filtered_index = 0

    #                     for i in range(len(filtered_list) + len(state_list)):
    #                         if i in idx_list:
    #                             result.append(state_list[state_index])
    #                             state_index += 1
    #                         else:
    #                             result.append(filtered_list[filtered_index])
    #                             filtered_index += 1
    #                     return result
    #                 return objective_func_map.get(method)(insert_states(variables, self.frozen_idx_list, self.frozen_state_list))
    #             return dctm_objective_func
    #         circuit_option.objective_func = dctm_objective_func_map(self.algorithm_optimization_method)
            
    #         if len(circuit_option.Hd_bits_list) == 0:
    #             cost = self.cost_dir * circuit_option.objective_func(circuit_option.feasiable_state)
    #             ARG = abs((cost - best_cost) / best_cost)
    #             best_solution_probs = 100 if cost == best_cost else 0
    #             in_constraints_probs = 100 if all([np.dot(circuit_option.feasiable_state, constr[:-1]) == constr[-1] for constr in self.dctm_linear_constraints]) else 0
    #             ARG_list.append(ARG)
    #             best_solution_probs_list.append(best_solution_probs)
    #             in_constraints_probs_list.append(100)
    #             iteration_count_list.append(0)
    #             continue
    #         ###################################

                    
    #         try:
    #             collapse_state, probs, iteration_count = solve(optimizer_option, circuit_option)
    #         except QuickFeedbackException as qfe:
    #             return qfe.data
    #         self.collapse_state=collapse_state
    #         self.probs=probs
    #         # 最优解的cost / by groubi
    #         iprint(f'best_cost: {best_cost}')
    #         mean_cost = 0
    #         best_solution_probs = 0
    #         in_constraints_probs = 0
    #         for cs, pr in zip(self.collapse_state, self.probs):
    #             pcost = self.cost_dir * dctm_objective_func_map('penalty')(cs)
    #             if pr >= 1e-3:
    #                 iprint(f'{cs}: {pcost} - {pr}')
    #             if all([np.dot(cs, constr[:-1]) == constr[-1] for constr in self.dctm_linear_constraints]):
    #                 in_constraints_probs += pr
    #                 if pcost == best_cost:
    #                     best_solution_probs += pr
    #             mean_cost += pcost * pr
    #         best_solution_probs *= 100
    #         in_constraints_probs *= 100
    #         maxprobidex = np.argmax(probs)
    #         max_prob_solution = collapse_state[maxprobidex]
    #         cost = self.cost_dir * circuit_option.objective_func(max_prob_solution)
    #         iprint(f"max_prob_solution: {max_prob_solution}, cost: {cost}, max_prob: {probs[maxprobidex]:.2%}") #-
    #         iprint(f'best_solution_probs: {best_solution_probs}')
    #         iprint(f"mean_cost: {mean_cost}")
    #         iprint(f'in_constraint_probs: {in_constraints_probs}')
    #         ARG = abs((mean_cost - best_cost) / best_cost)
    #         iprint(f'ARG: {ARG}')
    #         ARG_list.append(ARG)
    #         in_constraints_probs_list.append(in_constraints_probs)
    #         best_solution_probs_list.append(best_solution_probs)
    #         iteration_count_list.append(iteration_count)
    #     return ARG_list, in_constraints_probs_list, best_solution_probs_list, iteration_count_list  



