import logging
import sys
import os
import subprocess
import inspect


def recongnize():
    pass


def solveByMinisat(cnf_file_dir, result_dir="", verb=1, cpuLim=0, memLim=0, strictp=False):
    # 异步执行CMD命令
    if verb:
        verb = 1
    else:
        verb = 0
    process = subprocess.Popen([os.path.join(os.path.dirname(inspect.getfile(recongnize)), './minisat_core'), os.path.join(current_dir, cnf_file_dir), result_dir, str(verb), str(cpuLim),  str(memLim), str(strictp), "minisat"], stdout=subprocess.PIPE, text=True)
    
    # 读取命令输出
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            logging.info(output.strip())

def solveByHyqsat(cnf_file_dir, result_dir="1",verb=True, cpuLim=0, memLim=0, strictp=False):
    # 异步执行CMD命令
    if verb:
        verb = 1
    else:
        verb = 0
    process = subprocess.Popen([os.path.join(os.path.dirname(inspect.getfile(recongnize)), './minisat_core'), os.path.join(current_dir, cnf_file_dir), result_dir, str(verb), str(cpuLim),  str(memLim), str(strictp), "quantum", os.path.join(os.path.dirname(inspect.getfile(recongnize)), 'python/')], stdout=subprocess.PIPE, text=True)
    
    # 读取命令输出
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            logging.info(output.strip())

logging.info(os.path.dirname(__file__))
logging.info(inspect.getfile(recongnize))
logging.info(os.path.join(os.path.dirname(inspect.getfile(recongnize)), './minisat_core'))
logging.info(os.path.join(os.path.dirname(inspect.getfile(recongnize)), 'python/'))
# solveByHyqsat('examples/UF50-218-1000/')