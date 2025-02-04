# Project: optical sat solver
# author: dcao, zhe

import logging
import numpy as np

def reduce_to_3sat(cnfs, n, m, output_file, verbose=0):
    new_cnfs = []
    new_n = n
    new_m = m
    logging.info("[REDUCE] Convert k-sat to 3-sat")
    for cnf in cnfs:
        if len(cnf) == 1:
            x = cnf[0]
            new_cnfs.append([x, x, x])
        elif len(cnf) == 2:
            x1, x2 = cnf
            new_cnfs.append([x1, x2, x1])
        elif len(cnf) == 3:
            new_cnfs.append(cnf)
        else:
            x1, x2 = cnf[:2]
            x_l2, x_l1 = cnf[-2:]
            new_n += 1
            new_cnfs.append([x1, x2, new_n])
            _n_dummy = len(cnf) - 4
            p = 1
            while _n_dummy > 0:
                p += 1
                x_t1 = -new_n
                new_n += 1
                x_t2 = new_n
                new_cnfs.append([x_t1, cnf[p], x_t2])
                new_m += 1
                _n_dummy -= 1
            new_cnfs.append([-new_n, x_l2, x_l1])
            new_m += 1
    if new_n > n and verbose>=1:
        logging.info(f"[REDUCE] {new_n-n} dummy varibles are added: {n+1},...,{new_n}")
        logging.info(f"[REDUCE] {new_m-m} new clauses are created")
        logging.info(f"[RECUDE] New header: {new_n} varibles, {new_m} clauses")
    
    if output_file:
        with open(output_file, 'w') as f:
            line = f"p cnf {new_n} {new_m}\n"
            f.write(line)
            for cnf in new_cnfs:
                cnf_str = [str(x) for x in cnf]
                line = ' '.join(cnf_str) + " 0\n"
                f.write(line)
            f.write('%')
        logging.info(f"[REDUCE] Reduced 3-sat file saved at {output_file}")

    return new_cnfs, new_n, new_m
