should_print = True

from chocoq.problems.facility_location_problem import generate_flp
from chocoq.problems.set_cover_problem import generate_scp
from chocoq.problems.graph_coloring_problem import generate_gcp
from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
from chocoq.solvers.optimizers import CobylaOptimizer, AdamOptimizer
from chocoq.solvers.qiskit import (
    ChocoSolver, CyclicSolver, HeaSolver, PenaltySolver, NewSolver, NewXSolver, ChocoSolverSearch, 
    QtoSearchSolver, QtoSolver, QtoSimplifySolver, QtoSimplifyDiscardSolver, 
    AerGpuProvider, AerProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, DdsimProvider,
)

num_case = 1
# a, b = generate_scp(num_case,[(5, 5)])
a, b = generate_flp(num_case, [(4, 3)], 1, 20)
# a, b = generate_gcp(num_case, [(3, 2)])
# print(a[0][0])
# (1, [(2, 1), (3, 2), (3, 3), (4, 3), (4, 4)], 1, 20)

print(b)

best_lst = []
arg_lst = []

for i in range(num_case):
    opt = CobylaOptimizer(max_iter=200)
    aer = DdsimProvider()
    a[0][i].set_penalty_lambda(200)
    solver = QtoSimplifyDiscardSolver(
        prb_model=a[0][i],  # 问题模型
        optimizer=opt,  # 优化器
        provider=aer,  # 提供器（backend + 配对 pass_mannager ）
        num_layers=3,
        shots=1024
        # mcx_mode="linear",
    )
    result = solver.solve()
    u, v, w, x = solver.evaluation()
    print(f"{i}: {u}, {v}, {w}, {x}")

    best_lst.append(u)
    arg_lst.append(w)

print(solver.circuit_analyze(['depth', 'culled_depth', 'num_params']))
print(list(solver.time_analyze()))
# print(sum(best_lst) / num_case, sum(arg_lst) / num_case)
# t1, t2 = solver.time_analyze()
# print(counter.total_run_time )
# print("classical", t1)
# print("quantum", t2)