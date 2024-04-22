import logging
import re
import os
import subprocess
import inspect
import logging
import json

current_dir = os.getcwd()

def recongnize():
    pass

def match_result(res, cur):
    if cur.startswith("restarts"):
        res["restarts"] = int(re.findall(r'[0-9]+', cur)[0])
    elif cur.startswith("conflicts"):
        res["conflicts"] = int(re.findall(r'[0-9]+', cur)[0])
    elif cur.startswith("conflict cost"):
        res["conflict cost"] = float(re.findall(r'[0-9]+\.*[0-9]*', cur)[0])
    elif cur.startswith("decisions") and not cur.startswith("decisions cost"):
        res["decisions"] = int(re.findall(r'[0-9]+', cur)[0])
    elif cur.startswith("decisions cost"):
        res["decisions cost"] = float(re.findall(r'[0-9]+\.*[0-9]*', cur)[0])
    elif cur.startswith("propagations") and not cur.startswith("propagations cost"):
        res["propagations"] = int(re.findall(r'[0-9]+', cur)[0])
    elif cur.startswith("propagations cost"):
        res["propagations cost"] = float(re.findall(r'[0-9]+\.*[0-9]*', cur)[0])
    elif cur.startswith("conflict literals"):
        res["conflict literals"] = int(re.findall(r'[0-9]+', cur)[0])
    elif cur.startswith("actual CPU time"):
        res["actual CPU time"] = float(re.findall(r'[0-9]+\.*[0-9]*', cur)[0])
    elif cur.startswith("solving time"):
        res["solving time"] = float(re.findall(r'[0-9]+\.*[0-9]*', cur)[0])   
    elif cur.startswith("annealing time"):
        res["annealing time"] = float(re.findall(r'[0-9]+\.*[0-9]*', cur)[0])     
    elif cur.startswith("quantum count"):
        res["quantum count"] = int(re.findall(r'[0-9]+', cur)[0])  
    elif cur.startswith("simulation time"):
        res["simulation time"] = float(re.findall(r'[0-9]+\.*[0-9]*', cur)[0])   
    elif cur.startswith("quantum success number"):
        res["quantum success number"] = int(re.findall(r'[0-9]+', cur)[0])
    elif cur.startswith("quantum conflict number"):
        res["quantum conflict number"] = int(re.findall(r'[0-9]+', cur)[0])    
    elif cur.startswith("quantum one time solve number"):
        res["quantum one time solve number"] = int(re.findall(r'[0-9]+', cur)[0])
    elif cur.startswith("SATISFIABLE"):
        res["is satisfiable"] = True
    elif cur.startswith("UNSATISFIABLE"):
        res["is satisfiable"] = False
    elif cur.startswith("SAT"):
        res["is sat"] = True
    elif cur.startswith("UNSAT"):
        res["is sat"] = False

def solve_by_minisat(cnf_file, save=False, result_dir=".", verb=1, cpu_lim=0, mem_lim=0, strictp=False):
    '''
    description: using minisat method to solve sat domain problem.
    param {str} cnf_file: input a cnf file, which needs to be solve.
    param {bool} save: weather save result in result dir.
    param {str} result_dir: save result in result dir.
    param {bool} verb: weather print log.
    param {int} cpu_lim: cpu limit(core).
    param {int} mem_lim: memory limit(MB).
    param {bool} strictp: weather strict.
    '''
    if verb:
        verb = 1
    else:
        verb = 0
    res = {}
    process = subprocess.Popen([os.path.join(os.path.dirname(inspect.getfile(recongnize)), './minisat_core'), os.path.join(current_dir, cnf_file), result_dir, '1', str(cpu_lim),  str(mem_lim), str(strictp), "minisat", "null"], stdout=subprocess.PIPE, text=True)
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            cur = output.strip()
            match_result(res, cur)
            if verb:
                logging.info(cur)
    if save:
        with open(f'{result_dir}/cnf_file_result.txt', mode='w') as f:
            json.dump(res, f)
    return res
def solve_by_janusct(cnf_file, save=False, result_dir=".",verb=True, cpu_lim=0, mem_lim=0, strictp=False):
    '''
    description: using janusct method to solve sat domain problem.
    param {str} cnf_file: input a cnf file, which needs to be solve.
    param {bool} save: weather save result in result dir.
    param {str} result_dir: save result in result dir.
    param {bool} verb: weather print log.
    param {int} cpu_lim: cpu limit(core).
    param {int} mem_lim: memory limit(MB).
    param {bool} strictp: weather strict.
    
    Returns:
        dict with kes:
            'restarts': int,
            'conflicts': int,
            'conflict cost': in ms,
            'decisions': int,
            'decisions cost': 0.046,
            'propagations': int,
            'propagations cost': in ms,
            'conflict literals': int,
            'actual CPU time': in ms,
            'solving time': in ms,
            'annealing time': in ms,
            'simulation time': in ms,
            'quantum count': int,
            'quantum success number: int,,
            'quantum conflict number': int,
            'quantum one time solve number': int,
            'is satisfiable': True,

    '''
    if verb:
        verb = 1
    else:
        verb = 0
    process = subprocess.Popen([os.path.join(os.path.dirname(inspect.getfile(recongnize)), './minisat_core'), os.path.join(current_dir, cnf_file), result_dir, '1', str(cpu_lim),  str(mem_lim), str(strictp), "quantum", os.path.join(os.path.dirname(inspect.getfile(recongnize)), 'python/')], stdout=subprocess.PIPE, text=True)
    res = {}
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            cur = output.strip()
            match_result(res, cur)
            if verb:
                logging.info(cur)
    if save:
        with open(f'{result_dir}/cnf_file_result.txt', mode='w') as f:
            json.dump(res, f)
    return res
