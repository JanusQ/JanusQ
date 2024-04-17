import logging
import re
import os
import subprocess
import inspect
import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

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
    elif cur.startswith("decisions"):
        res["decisions"] = int(re.findall(r'[0-9]+', cur)[0])
    elif cur.startswith("decisions cost"):
        res["decisions cost"] = float(re.findall(r'[0-9]+\.*[0-9]*', cur)[0])
    elif cur.startswith("propagations"):
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
        res["isSatisfiable"] = True
    elif cur.startswith("UNSATISFIABLE"):
        res["isSatisfiable"] = False
    elif cur.startswith("SAT"):
        res["isSat"] = True
    elif cur.startswith("UNSAT"):
        res["isSat"] = False

def solve_by_minisat(cnf_file, result_dir="", verb=1, cpu_lim=0, mem_lim=0, strictp=False):
    if verb:
        verb = 1
    else:
        verb = 0
    res = {}
    process = subprocess.Popen([os.path.join(os.path.dirname(inspect.getfile(recongnize)), './minisat_core'), os.path.join(current_dir, cnf_file), result_dir, str(verb), str(cpu_lim),  str(mem_lim), str(strictp), "minisat", "null"], stdout=subprocess.PIPE, text=True)
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            cur = output.strip()
            match_result(res, cur)
            logging.info(cur)
    return res
def solve_by_janusct(cnf_file, result_dir="1",verb=True, cpu_lim=0, mem_lim=0, strictp=False):
    if verb:
        verb = 1
    else:
        verb = 0
    process = subprocess.Popen([os.path.join(os.path.dirname(inspect.getfile(recongnize)), './minisat_core'), os.path.join(current_dir, cnf_file), result_dir, str(verb), str(cpu_lim),  str(mem_lim), str(strictp), "quantum", os.path.join(os.path.dirname(inspect.getfile(recongnize)), 'python/')], stdout=subprocess.PIPE, text=True)
    res = {}
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            cur = output.strip()
            match_result(res, cur)
            logging.info(cur)
    return res

