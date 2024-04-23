/*****************************************************************************************[Main.cc]
Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**************************************************************************************************/

#include <errno.h>
#include <zlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <sys/stat.h>
#include <dirent.h>
#include <Python.h>

#include "minisat/utils/System.h"
#include "minisat/utils/ParseUtils.h"
#include "minisat/utils/Options.h"
#include "minisat/core/Dimacs.h"
#include "minisat/core/Solver.h"

#include <fstream>
// #include <minisat/tabu/solve_sat_2.h>

using namespace Minisat;

//=================================================================================================

static Solver *solver;
// Terminate by notifying the solver and back out gracefully. This is mainly to have a test-case
// for this feature of the Solver as it may take longer than an immediate call to '_exit()'.
static void SIGINT_interrupt(int) { solver->interrupt(); }

// Note that '_exit()' rather than 'exit()' has to be used. The reason is that 'exit()' calls
// destructors and may cause deadlocks if a malloc/free function happens to be running (these
// functions are guarded by locks for multithreaded use).
static void SIGINT_exit(int)
{
    printf("\n");
    printf("*** INTERRUPTED ***\n");
    if (solver->verbosity > 0)
    {
        solver->printStats();
        printf("\n");
        printf("*** INTERRUPTED ***\n");
    }
    _exit(1);
}

//=================================================================================================
// Main:
using namespace std;

int scan(Solver &S, string path, char **argv, vector<int>& opti);

void printAveResToFile(char const *path, int overall_decisions, int overall_conflict, int overall_propagations, double overall_timecost,
                       double overall_decision_timecost, double overall_conflict_timecost, double overall_propagations_timecost, double overall_delta_conflict_time, int solved_problem_num)
{
    FILE *fp;
    if ((fp = fopen(path, "wt+")) == NULL)
    {
        puts("Open file failure");
        exit(0);
    }
    fprintf(fp, "Final report:\n");
    fprintf(fp, "average timecost              : %f ms\n", double(overall_timecost / solved_problem_num));
    fprintf(fp, "average conflicts             : %f / problem\n", double(overall_conflict / solved_problem_num));
    fprintf(fp, "average conflict timecost     : %f ms\n", double(overall_conflict_timecost / solved_problem_num) / 1000);

    fprintf(fp, "average decisions             : %f / problem\n", double(overall_decisions / solved_problem_num));
    fprintf(fp, "average decision timecost     : %f ms\n", double(overall_decision_timecost / solved_problem_num) / 1000);

    fprintf(fp, "average propagations          : %f / problem\n", double(overall_propagations / solved_problem_num));
    fprintf(fp, "average propagation timecost  : %f ms\n", double(overall_propagations_timecost / solved_problem_num) / 1000);

    fprintf(fp, "average delta conflict time : %f ms", double(overall_delta_conflict_time) / solved_problem_num);
    fclose(fp);
}

class Hyqsat {
public:
    vector<bool> solveByMinisat(char **argv) {
        setUsageHelp("USAGE: %s [options] <input-file> <result-output-file>\n\n  where input may be either in plain or gzipped DIMACS.\n");
        setX86FPUPrecision();

        vector<int> opti(4);
        opti[0] = verb;
        opti[1] = cpuLim;
        opti[2] = memLim;
        opti[3] = strictp;

        printf("++++++++++++++++++++++++++++++************************************************************************\n");

        printf("Original method:\n");
        Solver correct_S;
        scan(correct_S, inputFile, argv, opti);
        vec<lbool> &correct_answer = correct_S.model;
        correct_S.printResults();
        return {};
    }
    vector<bool> solveByHyqsat(char** argv) {
        setUsageHelp("USAGE: %s [options] <input-file> <result-output-file>\n\n  where input may be either in plain or gzipped DIMACS.\n");
        setX86FPUPrecision();
        vector<int> opti(4);
        opti[0] = verb;
        opti[1] = cpuLim;
        opti[2] = memLim;
        opti[3] = strictp;

        Py_Initialize();

        printf("++++++++++++++++++++++++++++++************************************************************************\n");

        printf("Quantum method:\n");
        PyRun_SimpleString("import sys");
        string pythonPath = argv[8];
        string python_run_string = "sys.path.append('"+ pythonPath +"')";
        cout << "python sys: " << python_run_string << endl;
        PyRun_SimpleString(python_run_string.c_str());
        Solver S;
        S.enableQuantum();
        S.quantum_effect = true; // false; //

        scan(S, inputFile, argv, opti);
        S.printResults();
        Py_Finalize();
        return {};
    }
    Hyqsat(string filePath, int verb=1, int cpuLim=0, int memLim=0, bool strictp=false): inputFile(filePath), verb(verb), cpuLim(cpuLim), memLim(memLim), strictp(strictp) {}
private:
    string inputFile;
    int verb;
    int cpuLim;
    int memLim;
    bool strictp;
};

int main(int argc, char **argv)
{
    string dir_path_string, result_path_string;
    int verb, cpuLim, memLim;
    bool strictp;
    string methodType;
    if (argc == 1) {
        dir_path_string = "mul_cnf/";
        result_path_string = "result/";
        verb = 1;
        cpuLim = 0;
        memLim = 0;
        strictp = false;
        methodType = "q";
    } else {
        dir_path_string = argv[1];
        result_path_string = argv[2];
        verb = stoi(string(argv[3]));
        cpuLim = stoi(string(argv[4]));
        memLim = stoi(string(argv[5]));
        strictp = string(argv[6]) == "true";
        methodType = string(argv[7]);
    }
    // dir_path_string = "/home/jiaxinghui/hyqsat/examples/UF50-218-1000/uf50-01.cnf";
    // result_path_string = argv[2];
    // verb = 1;
    // cpuLim = 0;
    // memLim = 0;
    // strictp = false;
    // methodType = "quantum";
    Hyqsat hyqsat(dir_path_string, verb=verb, cpuLim=cpuLim, memLim=memLim, strictp=strictp);
    if (methodType == "quantum")
        hyqsat.solveByHyqsat(argv);
    else
        hyqsat.solveByMinisat(argv);
    return 0;
}

int scan(Solver &S, string path, char **argv, vector<int> &opti)
{
    try
    {
        // Extra options:
        IntOption verb("MAIN", "verb", "Verbosity level (0=silent, 1=some, 2=more).", opti[0], IntRange(0, 2));
        IntOption cpu_lim("MAIN", "cpu-lim", "Limit on CPU time allowed in seconds.\n", opti[1], IntRange(0, INT32_MAX));
        IntOption mem_lim("MAIN", "mem-lim", "Limit on memory usage in megabytes.\n", opti[2], IntRange(0, INT32_MAX));
        BoolOption strictp("MAIN", "strict", "Validate DIMACS header during parsing.", opti[3]);
        
        // parseOptions(argc, argv, true);
        S.verbosity = verb;

        solver = &S;
        // Use signal handlers that forcibly quit until the solver will be able to respond to
        // interrupts:
        sigTerm(SIGINT_exit);

        // Try to set resource limits:
        if (cpu_lim != 0)
            limitTime(cpu_lim);
        if (mem_lim != 0)
            limitMemory(mem_lim);
        std::ifstream in;
        in.open(path, ios::in);
        if (!in.is_open())
            printf("ERROR! Could not open file: %s\n", path), exit(1);

        if (S.verbosity > 0)
        {
            printf("============================[ Problem Statistics ]=============================\n");
            printf("|                                                                             |\n");
        }
        parse_DIMACS(in, S, (bool)strictp);
//        gzclose(in);
        in.close();
        if (S.verbosity > 0)
        {
            printf("|  Number of variables:  %12d                                         |\n", S.nVars());
            printf("|  Number of clauses:    %12d                                         |\n", S.nClauses());
        }
        //
        double parsed_time = cpuTime();

        // Change to signal-handlers that will only notify the solver and allow it to terminate
        // voluntarily:
        sigTerm(SIGINT_interrupt); //??

        vec<Lit> dummy;
        //        lbool ret = S.solveLimited(dummy);  //S.solveLimited(dummy);
        lbool ret = S.solve(dummy) ? l_True : l_False;
        if (S.verbosity > 0)
        {
            S.printStats();
            // S.printStatsToFile(resultFilePath);
            printf("\n");
        }
        printf(ret == l_True ? "SATISFIABLE\n" : ret == l_False ? "UNSATISFIABLE\n"
                                                                : "INDETERMINATE\n");
        if (ret == l_True)
        {
            printf("SAT\n");
            for (int i = 0; i < S.nVars(); i++)
                if (S.model[i] != l_Undef)
                    printf("%s%s%d", (i == 0) ? "" : " ", (S.model[i] == l_True) ? "" : "-", i + 1);
            printf(" 0\n");
        }
        else if (ret == l_False)
            printf("UNSAT\n");
        else
            printf("INDET\n");

        return (ret == l_True ? 10 : ret == l_False ? 20
                                                    : 0);
        // #endif
    }
    catch (OutOfMemoryException &)
    {
        printf("===============================================================================\n");
        printf("INDETERMINATE\n");
        exit(0);
    }
}
