from janusq.application.hyqsat import solve_by_hyqsat, solve_by_minisat
# input cnf flie
file_path = "examples/data/cnf_examples/test/uf50-01.cnf"
# if verbose
verbose = True
# cpuLim time (s). 0 means infinite
cpu_lim = 0
# memLim . 0 means infinite
mem_lim = 0
result_janus = solve_by_hyqsat(file_path, verb=verbose, cpu_lim=cpu_lim, mem_lim=mem_lim)
result_minisat = solve_by_minisat(file_path, verb=verbose, cpu_lim=cpu_lim, mem_lim=mem_lim)