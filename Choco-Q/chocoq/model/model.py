from __future__ import annotations
from collections import defaultdict
from typing import Dict, Tuple, Union, List, Set, Iterable
from itertools import product
from gurobipy import Model as GurobiModel, GRB, LinExpr
from functools import reduce
import operator

coeff_type = int
to_lin_constr: bool = True

def set_coeff_type(type_):
    global coeff_type
    coeff_type = type_

def set_to_lin_constr(flag: bool):
    # 暂时默认只有线性约束       
    global to_lin_constr
    to_lin_constr = flag

def fast_mul(terms: Iterable):
    return reduce(operator.mul, terms)

class Variable:
    # 推荐通过 Model 使用 Variable
    existing_variable_names_in_class: Set[str] = set()  # 类变量，存储已经存在类空间中的变量名
    unnamed_index = 1
    
    def __init__(self, vtype='binary', name='unnamed', enforce_unique_name_in_class: bool = True):
        if name == 'unnamed':
            name = f'unnamed_{Variable.unnamed_index}'
            Variable.unnamed_index += 1
        
        if enforce_unique_name_in_class:
            if name in Variable.existing_variable_names_in_class:
                raise ValueError(f"Variable with name '{name}' already exists in class.")
            
            Variable.existing_variable_names_in_class.add(name)
        self.vtype = vtype
        self.name = name
        # self.x = None

    def to_expression(self) -> Expression:
        """将 Variable 转为 Expression, 处理加减乘除/生成约束"""
        return Expression({tuple([self]): coeff_type(1)})

    def __neg__(self) -> Expression:
        return -1 * self.to_expression()

    def __add__(self, other) -> Expression:
        return self.to_expression() + other

    def __radd__(self, other) -> Expression:
        return self + other

    def __sub__(self, other) -> Expression:
        return self.to_expression() - other
    
    def __rsub__(self, other) -> Expression:
        return other - self.to_expression()

    def __mul__(self, other) -> Expression:
        return self.to_expression() * other

    def __rmul__(self, other) -> Expression:
        return self * other

    # def __truediv__(self, other):
    #     return self.to_expression() / other
    def __le__(self, other) -> Constraint:
        return Constraint(self.to_expression(), '<=', other)
    
    def __ge__(self, other) -> Constraint:
        return Constraint(self.to_expression(), '>=', other)

    def __eq__(self, other) -> Constraint:
        return Constraint(self.to_expression(), '==', other)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"{self.name}"

class Expression:
    def __init__(self, terms: Dict[Tuple[Variable, ...], Union[int, float]] = None):
        self.terms = {tuple(sorted(term, key=lambda var: var.name)): coeff for term, coeff in (terms or {}).items()}
    
    def extract_constant(self):
        """提取常数项，并从表达式中移除"""
        if () in self.terms:
            constant = self.terms.pop(())
            return constant
        return coeff_type(0)
    
    def max_for_lin(self):
        """得到线性表达式的最大值，用来等式约束转换添加辅助变量"""
        assert all(len(term) <= 1 for term in self.terms.keys()), "Expression is not linear."
        
        max_value = 0
        for vars, coeff in self.terms.items():
            if vars:  # 如果不是常数项
                max_value += max(coeff, 0)
            else:
                max_value += coeff  # 常数项直接加到结果中
        
        return max_value

    def min_for_lin(self):
        """得到线性表达式的最小值，用来等式约束转换添加辅助变量"""
        assert all(len(term) <= 1 for term in self.terms.keys()), "Expression is not linear."
        
        min_value = 0
        for vars, coeff in self.terms.items():
            if vars:  # 如果不是常数项
                min_value += min(coeff, 0)
            else:
                min_value += coeff  # 常数项直接加到结果中
        
        return min_value
    
    def to_gurobi_expr(self, gurobi_vars):
        expr = LinExpr()
        for vars_tuple, coeff in self.terms.items():
            term_expr = coeff
            for var in vars_tuple:
                term_expr *= gurobi_vars[var.name]
            expr += term_expr
        return expr
    
    def __add__(self, other):
        result = defaultdict(coeff_type, self.terms)
        if isinstance(other, (int, float)):
            other = Expression({(): coeff_type(other)})
        elif isinstance(other, Variable):
            other = other.to_expression()
        for var, coeff in other.terms.items():
            result[var] += coeff
        return Expression(result)
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return -1 * self + other

    def __mul__(self, other):
        result = defaultdict(coeff_type)
        if isinstance(other, (int, float)):
            other = Expression({(): coeff_type(other)})
        elif isinstance(other, Variable):
            other = other.to_expression()
        for var_1, coeff_1 in self.terms.items():
            for var_2, coeff_2 in other.terms.items():
                combined_var = tuple(sorted(var_1 + var_2, key=lambda v: v.name))
                result[combined_var] += coeff_1 * coeff_2
        return Expression(result)

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return -1 * self
    
    def __pow__(self, power: int):
        """处理幂运算，快速幂"""
        if power < 0:
            raise ValueError("Negative powers are not supported.")
        
        result = Expression({(): 1})  # 初始为常数1
        base = self

        while power:
            if power % 2:
                result *= base
            base *= base
            power //= 2
        
        return result
    
    def __le__(self, other):
        return Constraint(self, '<=', other)
    
    def __ge__(self, other):
        return Constraint(self, '>=', other)
    
    def __eq__(self, other):
        return Constraint(self, '==', other)

    def __repr__(self):
        terms_repr = " + ".join(
            f"{coeff} * {' * '.join(str(var) for var in term)}" 
            for term, coeff in self.terms.items() 
            if term  # 特判，跳过键为空的情况
        )
        # 如果有常数项（空元组），并且系数不为0，添加到结果中
        if () in self.terms and self.terms[()] != 0:
            if terms_repr:
                terms_repr = f"{terms_repr} + {self.terms[()]}"
            else:
                terms_repr = f"{self.terms[()]}"
        return terms_repr

class Constraint:
    def __init__(self, expr: Expression, sense, rhs):
        if isinstance(rhs, Variable):
            rhs = rhs.to_expression()
        # 提取 rhs 中的常数项
        if isinstance(rhs, Expression):
            rhs_const = rhs.extract_constant()
            rhs_expr = rhs
        else:
            rhs_const = rhs
            rhs_expr = Expression()

        # 提取 expr 中的常数项
        expr_const = expr.extract_constant()

        # 将 rhs 的非常数部分移到左边的表达式中
        self.expr: Expression = expr - rhs_expr
        self.sense = sense
        self.rhs = rhs_const - expr_const

    def __repr__(self):
        return f"{self.expr} {self.sense} {self.rhs}"

class Model:
    def __init__(self):
        self.variables: List[Variable] = []
        self.existing_var_names: Set[str] = set()
        self.constraints: List[Constraint] = []
        self.objective = None   
        self.obj_sense = None

    def addVar(self, vtype='binary', *, name) -> Variable:
        if name in self.existing_var_names:
            print(f"Variable with name '{name}' already exists in model.")
            return None
        
        self.existing_var_names.add(name)
        var = Variable(vtype, name, False)
        self.variables.append(var)
        return var
    
    def addVars(self, *dimensions, vtype='binary', name) -> Dict[Tuple[int, ...], Variable]:
        if name in self.existing_var_names:
            print(f"Variable with name '{name}' already exists in model.")
            return None

        self.existing_var_names.add(name)
        vars = {}
        # 为了同步gurobi的索引方式，要特殊处理一维情况
        is_single_dim = len(dimensions) == 1
        # 生成所有可能的索引组合
        dimension_ranges = [range(d) for d in dimensions]
        for index_tuple in product(*dimension_ranges):
            var_name = f"{name}_{'_'.join(map(str, index_tuple))}"
            var = Variable(vtype, var_name, False)
            self.variables.append(var)
            if is_single_dim:
                vars[index_tuple[0]] = var
            else:
                vars[index_tuple] = var
        return vars
    
    def setObjective(self, expression: Expression, sense):
        self.objective = expression
        self.obj_sense = sense

    def addConstr(self, constraint: Constraint):
        self.constraints.append(constraint)

    def addConstrs(self, constraints: List[Constraint]):
        for constr in constraints:
            self.addConstr(constr)
    
    def update(self):
        """将模型的更改同步到模型内部的数据结构"""
        pass
    
    def to_gurobi_model(self, vtype=GRB.BINARY) -> GurobiModel:
        # 创建一个 Gurobi 模型实例
        gurobi_model = GurobiModel()

        # 添加变量
        gurobi_vars = {}
        for var in self.variables:
            gurobi_vars[var.name] = gurobi_model.addVar(vtype=vtype, name=var.name)
        
        # 添加目标函数
        obj_expr = 0
        for vars_tuple, coeff in self.objective.terms.items():
            term_value = coeff
            for var in vars_tuple:
                term_value *= gurobi_vars[var.name]
            obj_expr += term_value
        
        gurobi_model.setObjective(obj_expr, GRB.MINIMIZE if self.obj_sense == 'min' else GRB.MAXIMIZE)

        # 添加约束
        for constr in self.constraints:
            if constr.sense == '==':
                gurobi_model.addConstr(constr.expr.to_gurobi_expr(gurobi_vars) == constr.rhs)
            elif constr.sense == '<=':
                gurobi_model.addConstr(constr.expr.to_gurobi_expr(gurobi_vars) <= constr.rhs)
            elif constr.sense == '>=':
                gurobi_model.addConstr(constr.expr.to_gurobi_expr(gurobi_vars) >= constr.rhs)

        return gurobi_model
    
    def optimize_with_gurobi(self):
        gurobi_model = self.to_gurobi_model()
        gurobi_model.setParam('OutputFlag', 0)
        gurobi_model.optimize()

        if gurobi_model.status == GRB.OPTIMAL:
            optimal_value = gurobi_model.objVal
            solution = {v.varName: v.x for v in gurobi_model.getVars()}
            return optimal_value, solution
        else:
            raise Exception("No optimal solution found.")

    def optimize(self):
        # Here we just assign some dummy values for the sake of demonstration.
        for var in self.variables:
            var.x = 0x7f
        # Let's assume the optimal objective value is 0 for this dummy solution
        self.objVal = 0xdb

    def __repr__(self):
        var_str = "   ".join([repr(var)+f" (type: {var.vtype})" for var in self.variables])
        constr_str = "\n".join([repr(constr) for constr in self.constraints])
        return (
            f"m:\n"
            f"variables:\n{var_str}\n\n"
            f"obj:\n{self.obj_sense} {self.objective}\n\n"
            f"s.t.:\n{constr_str}\n\n"
            )

if __name__ == '__main__':
    m = Model()
    num_facilities = 1
    num_demands = 1
    x = m.addVars(num_facilities, name="x")
    y = m.addVars(num_demands, num_facilities, name="y")
    m.setObjective(sum(3 * y[i, j] for i in range(num_demands) for j in range(num_facilities)) + sum(4 * x[j] for j in range(num_facilities)), 'min')

    m.addConstrs((x[j] <= 2 for i in range(num_demands) for j in range(num_facilities)))
    m.addConstrs((-2 <= -x[j] for i in range(num_demands) for j in range(num_facilities)))
    m.addConstrs((x[j] >= -1 for i in range(num_demands) for j in range(num_facilities)))
    m.addConstrs((x[j] >= 1 for i in range(num_demands) for j in range(num_facilities)))
    m.addConstrs((y[i, j] + x[j] <=  2 for i in range(num_demands) for j in range(num_facilities)))
    m.addConstrs((y[i, j] + x[j] + 10 >=  10 for i in range(num_demands) for j in range(num_facilities)))
    m.addConstrs((y[i, j] + x[j] + 10 >=  -10 for i in range(num_demands) for j in range(num_facilities)))
    m.addConstrs((y[i, j] + x[j] + 10 >=  11 for i in range(num_demands) for j in range(num_facilities)))
    m.addConstrs((y[i, j] + x[j] + 10 >=  13 for i in range(num_demands) for j in range(num_facilities)))
    m.addConstrs((10 >= 13 - y[i, j] - x[j] for i in range(num_demands) for j in range(num_facilities)))


    # m.optimize()
    print(m)
    # print(f"optimal objective value: {m.objVal}")
    # print(f"solution values: x={x.x}, y={y.x}, z={z.x}")

