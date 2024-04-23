/****************************************************************************************[Dimacs.h]
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
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**************************************************************************************************/

#ifndef Minisat_Dimacs_h
#define Minisat_Dimacs_h

#include <stdio.h>
//#include <minisat/tabu/solve_sat_2.h>

#include "minisat/utils/ParseUtils.h"
#include "minisat/core/SolverTypes.h"


#include <iostream>
#include <fstream>
#include <regex>
using namespace std;

namespace Minisat {

//=================================================================================================
// DIMACS Parser:

template<class B, class Solver>
static void readClause(B& in, Solver& S, vec<Lit>& lits) {
    int     parsed_lit, var;
    lits.clear();
    for (;;){
        parsed_lit = parseInt(in);
        if (parsed_lit == 0) break;
        var = abs(parsed_lit)-1;
        while (var >= S.nVars()) S.newVar();
        lits.push( (parsed_lit > 0) ? mkLit(var) : ~mkLit(var) );
    }
}

char* lbool2string(lbool b){
    if (b == l_Undef)
        return "u";
    if (b == l_True)
        return "t";
    if (b == l_False)
        return "f";
    assert(true);
}

template <class T>
void printClause_dimacs(T &lits){
    printf("Clause:");
    for(int j = 0; j < lits.size(); j++) {
        Lit literal = lits[j];

        char *var_satisfy_s = "";
        Var variable = var(literal);
        bool signal = sign(literal);
        printf("[%s%d %s]", signal? "^":"", variable, var_satisfy_s);
    }

    printf("\n");
}

//template<class B, class Solver>
//static void parse_DIMACS_main(B& in, Solver& S, bool strictp = false) {
//    vec<Lit> lits;
//    int vars    = 0;
//    int clauses = 0;
//    int cnt     = 0;
//    char str[1024];
//    while (in.getline(str)){
//        skipWhitespace(in);
//        if (*in == EOF) break;
//        else if(*in == '%') break;
//        else if (*in == 'p'){
//            if (eagerMatch(in, "p cnf")){
//                vars    = parseInt(in);
//                clauses = parseInt(in);
//                // SATRACE'06 hack
//                // if (clauses > 4000000)
//                //     S.eliminate(true);
//            }else{
//                printf("PARSE ERROR! Unexpected char: %c\n", *in), exit(3);
//            }
//        } else if (*in == 'c' || *in == 'p')
//            skipLine(in);
//        else if(*in == '%')
//            skipLine(in);
//        else{
//            cnt++;
//            readClause(in, S, lits);
////            printClause_dimacs(lits);
//            S.addClause_(lits);
//        }
//    }
//    if (strictp && cnt != clauses)
//        printf("PARSE ERROR! DIMACS header mismatch: wrong number of clauses\n");
//}

template<class B, class Solver>
static void parse_DIMACS_main(B& in, Solver& S, bool strictp = false) {
    vec<Lit> lits;
    int vars    = 0;
    int clauses = 0;
    int cnt     = 0;
    for (;;){
        skipWhitespace(in);
        if (*in == EOF) break;
        else if(*in == '%') break;
        else if (*in == 'p'){
            if (eagerMatch(in, "p cnf")){
                vars    = parseInt(in);
                clauses = parseInt(in);
                // SATRACE'06 hack
                // if (clauses > 4000000)
                //     S.eliminate(true);
            }else{
                printf("PARSE ERROR! Unexpected char: %c\n", *in), exit(3);
            }
        } else if (*in == 'c' || *in == 'p')
            skipLine(in);
        else if(*in == '%')
            skipLine(in);
        else{

            cnt++;
            readClause(in, S, lits);
//            printClause_dimacs(lits);
            S.addClause_(lits);
        }
    }
    if (strictp && cnt != clauses)
        printf("PARSE ERROR! DIMACS header mismatch: wrong number of clauses\n");
}

// Inserts problem into solver.

//template<class Solver>
//static void parse_DIMACS(gzFile input_stream, Solver& S, bool strictp = false) {
//    StreamBuffer in(input_stream);
//    parse_DIMACS_main(in, S, strictp); }

////=================================================================================================
//}
    std::vector<int> extractNumbers(const std::string& str)
    {
        std::regex pattern("-*\\d+\\.?\\d*"); // 正则表达式模式匹配整数和浮点数

        std::sregex_iterator it(str.begin(), str.end(), pattern);
        std::sregex_iterator end;
        vector<int> res;
        while (it != end)
        {
            res.emplace_back(stoi(it -> str()));
            ++it;
        }
        return res;
    }
    template<class Solver>
    static void parse_DIMACS(ifstream &in, Solver& S, bool strictp = false) {
    //    StreamBuffer in(input_stream);
        vec<Lit> lits;
        int vars    = 0;
        int clauses = 0;
        int cnt     = 0;
        string str;
        while (getline(in, str)) {
            if (str[0] == 'c')
                continue;
            if (str[0] == 'p') {
                auto res = extractNumbers(str);
                if (res.size() != 2) cout << "p cnt failed";
                vars = res[0];
                clauses = res[1];
                continue;
            } else if (str[0] == '%')
                break;
            else {
                vector<int> data = extractNumbers(str);
                int var;
                lits.clear();
                for (auto n: data) {
                    if (n == 0)
                        break;
                    var = abs(n) - 1;
                    while (var >= S.nVars()) S.newVar();
                    lits.push( (n > 0) ? mkLit(var) : ~mkLit(var) );
                }
                cnt++;
                S.addClause_(lits);
            }
        }
        if (strictp && cnt != clauses)
            printf("PARSE ERROR! DIMACS header mismatch: wrong number of clauses\n");
    }

//=================================================================================================
}
//template<class Solver>
//static void parse_DIMACS(ifstream input_stream, Solver& S, bool strictp = false) {
////    StreamBuffer in(input_stream);
//    parse_DIMACS_main(input_stream, S, strictp); }
//
////=================================================================================================
//}

#endif
