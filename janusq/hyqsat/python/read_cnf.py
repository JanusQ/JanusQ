import logging
from common_function import Var
  
def readCNF(filename):
    """
    Reads a DIMACS CNF format file, returns clauses (set of frozenset) and
    literals (set of int).
        :param filename: the file name
        :raises FileFormatError: when file format is wrong
        :returns: (clauses, literals)
    """
    with open(filename) as f:
        lines = [
            line.strip().split() for line in f.readlines()
            if (not (line.startswith('c') or
                        line.startswith('%') or
                        line.startswith('0'))
                and line != '\n')
        ]  #跳过前面的注释

    if lines[0][:2] == ['p', 'cnf']:
        count_literals, count_clauses = map(int, lines[0][-2:])
    else:
        logging.exception('Number of literals and clauses are not declared properly.')

    variables = set()
    clauses = list()

    for line in lines[1:]:
        if line[-1] != '0':
            logging.exception('file error: %s' % line)
        clause = list(map(int, line[:-1]))  #冻结后集合不能再添加或删除任何元素。 
        clause.sort(key = lambda x: abs(x))
        clause = tuple(clause)
        variables.update(map(abs, clause))
        clauses.append(clause)

    # if len(variables) != count_literals or len(lines) - 1 != count_clauses:
    #     raise Exception(
    #         'Unmatched literal count or clause count.'
    #         ' Literals expected: {}, actual: {}.'
    #         ' Clauses expected: {}, actual: {}.'
    #             .format(count_literals, len(variables), count_clauses, len(clauses)))

    return clauses, variables


def readCNF_intr(filename):
    with open(filename) as f:
        lines = [
            line.strip().strip('0').split() for line in f.readlines()
            if (not (line.startswith('c') or
                        line.startswith('%') or
                        line.startswith('0'))
                and line != '\n')
        ]  #跳过前面的注释

    old_var2var = {}
    var_count = 1
    clauses = []

    for line in lines:
        # print(line)
        for old_lit in line:
            old_var = Var(old_lit)
            if old_var not in old_var2var:
                old_var2var[old_var] = var_count
                var_count += 1
        line = [int(old_var2var[Var(lit)] * (Var(lit)/int(lit))) for lit in line]
        line.sort()
        line = tuple(line)
        clauses.append(line)

    return clauses, None

def dealCNF_intr(lines):
    split_line = lines.split(";")[:-1]
    var_count = 1
    clauses = []
    old_var2var = {}
    for line in split_line:
        # print(line)
        tmp = line.split(',')
        for old_lit in tmp:
            old_var = Var(old_lit)
            if old_var not in old_var2var:
                old_var2var[old_var] = var_count
                var_count += 1
        line = [int(old_var2var[Var(lit)] * (Var(lit)/int(lit))) for lit in tmp]
        line.sort()
        line = tuple(line)
        clauses.append(line)

    return clauses, None