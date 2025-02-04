//
// Created by bamboosfox on 2021/10/19.
//
#include "minisat/tabu/bqp.cpp"
#include "minisat/tabu/tabu_search.cpp"

#ifndef MINISAT_SOLVE_SAT_2_H
#define MINISAT_SOLVE_SAT_2_H

#endif //MINISAT_SOLVE_SAT_2_H

#include <vector>
#include <cstdlib>// Header file needed to use rand
#include "minisat/mtl/Vec.h"
#include "minisat/core/SolverTypes.h"
#include "minisat/core/Solver.h"
#include <list>
#include <algorithm>
#include <set>
#include <iostream>
#include "minisat/mtl/Sort.h"
#include "minisat/mtl/IntMap.h"
#include "Simulate.h"
#include <Python.h>
#include <thread>


using namespace std;
using namespace Minisat;

typedef int Var;  //why not unsigned int


// 定义64位整形
#if defined(_WIN32) && !defined(CYGWIN)
typedef __int64 int64_t;
#else
typedef long long int64t;
#endif  // _WIN32

// 获取系统的当前时间，单位微秒(us)
int64_t getSysTimeMicros() {
#ifdef _WIN32
    // 从1601年1月1日0:0:0:000到1970年1月1日0:0:0:000的时间(单位100ns)
#define EPOCHFILETIME   (116444736000000000UL)
    FILETIME ft;
    LARGE_INTEGER li;
    int64_t tt = 0;
    GetSystemTimeAsFileTime(&ft);
    li.LowPart = ft.dwLowDateTime;
    li.HighPart = ft.dwHighDateTime;
    // 从1970年1月1日0:0:0:000到现在的微秒数(UTC时间)
    tt = (li.QuadPart - EPOCHFILETIME) /10;
    return tt;
#else
    timeval tv;
    gettimeofday(&tv, 0);
    return (int64_t) tv.tv_sec * 1000000 + (int64_t) tv.tv_usec;
#endif // _WIN32
    return 0;
}

struct Node{
    Var var;
    double activity;
    Node(Var _var, double  _activity){
        var = _var;
        activity = _activity;
    }
};

bool operator<(Node a, Node b){
    return a.activity<b.activity;
}

char *lbool2string(lbool b) {
    if (b == l_Undef)
        return "u";
    if (b == l_True)
        return "t";
    if (b == l_False)
        return "f";
    assert(true);
}

string lit2string(Lit literal, Solver *s = NULL) {
    string var_satisfy_s = "";
    if (s) {
        lbool var_satisfy = s->value(literal);
        var_satisfy_s = lbool2string(var_satisfy);
    }
    Var variable = var(literal);
    string signal = sign(literal) ? "^" : "";  //important! true means reverse?
    return "[" + signal + to_string(variable) + " " + var_satisfy_s + "]";
}

template<class T>
string clause2string(T &lits, Solver *s = NULL) {
    string str = "Clause:";
    vec<Lit> temp_lits(lits.size());
    for (int j = 0; j < lits.size(); j++) {
        temp_lits[j] = lits[j];
    }
    sort(temp_lits);

    for (int j = 0; j < temp_lits.size(); j++) {
        Lit literal = temp_lits[j];
        str += lit2string(literal);
    }
    return str;
}

template<class T>
void printClause(T &lits, Solver *s = NULL) {
    printf("%s\n", clause2string(lits, s).c_str());
}

bool litSatisfy(Lit p, lbool v) {
    return (v ^ sign(p)) == l_True;
}

vector<Lit> clause2vectorLit(Clause clauses) {
    vector<Lit> new_vec(clauses.size());
    for (int i = 0; i < clauses.size(); i++) {
        new_vec[i] = clauses[i];
    }
    return new_vec;
}

void clause2vecLit(Clause &clause, vec<Lit> &new_vec) {
    new_vec.clear();
    new_vec.growTo(clause.size());
    for (int i = 0; i < clause.size(); i++) {
        new_vec[i] = clause[i];
    }
}

template<class T>
void vec2vector(vec<T> &_vec, vector<T> &new_vector) {
    new_vector.clear();
    new_vector.resize(_vec.size());
    for (int i = 0; i < _vec.size(); i++) {
        new_vector[i] = _vec[i];
    }
    return;
}

template<class T>
void vector2vec(vector<T> &_vector, vec<Lit> &new_vec) {
//    vec<T> _vec(_vector.size());
    new_vec.clear();
    new_vec.growTo(_vector.size());
    for (int i = 0; i < _vector.size(); i++) {
        new_vec[i] = _vector[i];
    }
    return;
}

template<class T>
void combineList(vec<T> &stl1, vec<T> &stl2, vec<T *> &combined_stl) {
    combined_stl.clear();
    combined_stl.growTo(stl1.size() + stl2.size());

    for (int i = 0; i < stl1.size(); i++) {
        combined_stl[i] = &stl1[i];
    }
    for (int i = 0; i < stl2.size(); i++) {
        combined_stl[stl1.size() + i] = &stl2[i];
    }

}

int gen_count = 0;
queue<vec<Lit> *> vec_lit_company;
vec<vec<Lit> *> all_assigned_vec_lits;

vec<Lit> *genVecLit(int size = 0) {
//    printf("+%d \n", ++gen_count);
    vec<Lit> *new_vec_lit = NULL;
    if (vec_lit_company.empty()) {
        new_vec_lit = new vec<Lit>();
    } else {
        new_vec_lit = vec_lit_company.front();
        vec_lit_company.pop();
    }
    new_vec_lit->growTo(size);
    all_assigned_vec_lits.push(new_vec_lit);
    return new_vec_lit;
}

int free_count = 0;

void freeVecLit(vec<Lit> *vec_lit) {
    vec_lit->clear();
    vec_lit_company.push(vec_lit);
//    printf("-%d \n", ++free_count);
}

void freeAllVecLit() {
    for (int i = 0; i < all_assigned_vec_lits.size(); i++) {
        vec<Lit> *vec_lit = all_assigned_vec_lits[i];
        freeVecLit(vec_lit);
    }
    all_assigned_vec_lits.clear();
}


bool clauseSatisfy(vec<Lit> &unsatisfy_lits, Clause &clause, Solver &S) {
    unsatisfy_lits.clear();
    for (int j = 0; j < clause.size(); j++) {
        lbool var_satisfy = S.value(clause[j]);
        if (var_satisfy == l_True) {
            unsatisfy_lits.clear();
            return true;
        } else if (var_satisfy == l_Undef) {
            unsatisfy_lits.push(clause[j]);
        }
    }
    return false;
}

void addClause2Solver(vec<Lit> &lits, Solver &solver) {
    for (int j = 0; j < lits.size(); j++) {
        Var original_var = var(lits[j]);
        while (original_var >= solver.nVars()) solver.newVar();
    }

    solver.addClause(lits);
}


void addClause2Simulator(vec<Lit> *lits, Simulate &simulator) {
    simulator.addClause(lits);
}

template<class STL, class OBJ>
bool has(STL stl, OBJ obj) {
    return stl.find(obj) != stl.end();
}

void printClauseToFile( char const* path, Solver& S)  {
    FILE *fp;
    if((fp=fopen(path,"a+"))==NULL){
        puts("Open file failure");
        exit(0);
    }


    if (true) {
        vec<CRef> &clauses = S.clauses;
        Minisat::ClauseAllocator &ca = S.ca;
        fprintf(fp,"clauses: \n");
        for(int i = 0; i<clauses.size();i++){
            Clause *c = &ca[clauses[i]];
//            printf("%s\n", clause2string(*c, NULL).c_str());
            fprintf(fp,"%s\n", clause2string(*c, NULL).c_str());
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}



void writeClauseTofile(vector<vec<Lit>*> &added_clauses ,Solver &S ,char const* outfilepre){
    char const* fileName = S.fileName;
    int num = S.solveNum;
    FILE *fp;
//    char const* outfilepre = "../failClauses/";
    string newFileName = string(fileName);
    newFileName.erase(newFileName.length()-4,4);
    string const&origin_path = string(outfilepre) + newFileName +"_"+ to_string(num) + string(".txt");
    char const* path = origin_path.c_str();
    if((fp=fopen(path,"a"))==NULL){
        puts("Open file failure");
        exit(0);
    }

    for(int i = 0; i<added_clauses.size(); i++){
        vec<Lit> &clause = *added_clauses[i];
        for(int j = 0; j<clause.size(); j++){
            if(S.value(clause[j]) != l_Undef) continue;
            Var v = var(clause[j]);
            if(sign(clause[j])){
                v = -(v+1);
            }
            else{
                v = v+1;
            }
            fprintf(fp,"%d ", v);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

bool my_rand(double percantage){
    return (rand()%100+1) < (int)(100 * percantage);
}

struct VarActivity
{
    Var v;
    double activity;
};
bool var_cmp(VarActivity x, VarActivity y)
{
    if(x.activity==y.activity)
        return x.activity>y.activity;
    return x.activity>y.activity;
}

PyObject* vectorToTuple_int(const vector<int>& data){
    PyObject *tuple = PyTuple_New(data.size());
    for(unsigned int i = 0; i<data.size(); i++){
        PyTuple_SetItem(tuple,i, Py_BuildValue("i", data[i]));
    }
    return tuple;
}

string itoa_self(int i)
{
    stringstream ss;
    ss << i;
    return ss.str();
}
// void dwave2python(vector<vector<int> > &clauses,int total_clause_size)
void dwave2python(string clauses, map<Var, lbool> &var_result, bool is_real)
{
    cout << "is_real:" << (is_real? "True":"False") << endl;
    PyObject* pModule = NULL;
    PyObject* pFunc = NULL;
    pModule = PyImport_ImportModule("test_noise");
    pFunc = PyObject_GetAttrString(pModule,"solve");
    // real
    cout << "this 1"<< endl;
    PyObject* pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, Py_BuildValue("(s)", clauses.c_str()));
    PyTuple_SetItem(pArgs, 1, Py_BuildValue("(s)", (is_real? "True":"False")));
    cout << "this 2"<< endl;
    // PyObject* pParams =  Py_BuildValue("(s)", clauses.c_str());
    // cout<<"HERE in dwave2python\n";
    // PyObject* pRet = PyObject_CallObject(pFunc,pParams);
    PyObject* pRet = PyObject_CallObject(pFunc, pArgs);
    cout << "this 3"<< endl;
    PyObject* pIsSatisfy = PyDict_GetItemString(pRet,"satisfy");
    char* isSatisfy;
    PyArg_Parse(pIsSatisfy,"s",&isSatisfy);
    if(strcmp(isSatisfy,"1")==0){
        Py_ssize_t i, j;
        i = 0;
        PyObject *key;
        PyObject *value;
        while (PyDict_Next(pRet, &i, &key, &value)) //C++中遍历python的dict对象。此为关键代码
        {
            char *skey;
            char *svalue;
            PyArg_Parse(key,"s",&skey);
            PyArg_Parse(value,"s",&svalue);
            if(strcmp(skey,"isSatisfy")==0 || strcmp(skey,"satisfy")==0) continue;
            var_result[atoi(skey)-1] = strcmp(svalue,"1")==0?l_True:l_False;
        }

    }


    Py_CLEAR(pModule);
    // Py_CLEAR(pParams);
    Py_CLEAR(pArgs);
    Py_CLEAR(pFunc);
    Py_CLEAR(pRet);
}

bool dwave(string clauses,  map<Var, lbool> &var_result, bool is_real){
    dwave2python(clauses, var_result, is_real);
    // cout<<"Here\n";
    if(var_result.size()==0) return false;
    return true;
}

string convert2string(vec<Lit> &clause){
    string new_clause = "";
    for(int i = 0; i<clause.size(); i++){
        bool s = sign(clause[i]);
        if(!s){
            new_clause+=to_string(var(clause[i])+1)+",";
        }else{
            new_clause+= to_string(-var(clause[i])-1)+",";
        }
    }
    new_clause[new_clause.size()-1] = ';';
    return new_clause;
}

//采用模拟器的版本
bool solveSATQuantum(Solver &S, vec<Lit> &learnt_clause, int &backtrack_level, int cnt_flag, vector<vec<Lit>* > & all_unsatisfy_lits, vector<vector<vec<Lit> *> *> & var2clauses, bool do_one_time_solve, bool do_set_phase, bool do_backtrack, int MAX_VAR_NUM, int MAX_CLAUSE_NUM) {
    if(all_unsatisfy_lits.size() == 0) return true;

    Solver minisat_solver;
    minisat_solver.verbosity = 0;

    int auxiNum = 0;

    Simulate simulator;

    if(cnt_flag==1){
        for (int i = 0; i < S.nVars(); i++) {
            reverse(var2clauses[i]->begin(), var2clauses[i]->end());
        }
        reverse(all_unsatisfy_lits.begin(), all_unsatisfy_lits.end());
    }
    else if(cnt_flag==0){
        for (int i = 0; i < S.nVars(); i++) {
            random_shuffle(var2clauses[i]->begin(), var2clauses[i]->end());
        }
        random_shuffle(all_unsatisfy_lits.begin(), all_unsatisfy_lits.end());
    }
    else if(cnt_flag==3 && all_unsatisfy_lits.size()>1){
        int maxlen = all_unsatisfy_lits.size() >= 10 ? 10 : all_unsatisfy_lits.size();
        int index = 0 + rand()%(maxlen-1);
//        cout<<index<<endl;
        vec<Lit>* tmp = all_unsatisfy_lits[index];
        all_unsatisfy_lits[index] = all_unsatisfy_lits[0];
        all_unsatisfy_lits[0] = tmp;
    }
    // TODO: 从刚变过的(propogate)不满足的clause里面挑

    vector<vec<Lit>* > added_clauses;
    string clauses2python;
    int total_clause_size = 0;


    vector<Var> added_vars;
    vector<Var> var_visit(S.nVars(),0);//TODO: 也改成mark

//    const int MAX_VAR_NUM = 128; //100; //64;

    for (int i = 0; i < all_unsatisfy_lits.size()  && added_vars.size() < MAX_VAR_NUM && added_clauses.size() < MAX_CLAUSE_NUM; i++) {
        vec<Lit> *unsatisfy_lits_p = all_unsatisfy_lits[i];
        vec<Lit> &unsatisfy_lits = *all_unsatisfy_lits[i];

        if(unsatisfy_lits_p->getmark() == 2)
            continue;

//        printf("scan to %s\n", clause2string(unsatisfy_lits, &S).c_str());
        unsatisfy_lits.setmark(2);

        addClause2Solver(unsatisfy_lits, minisat_solver);
        addClause2Simulator(unsatisfy_lits_p, simulator);
        added_clauses.push_back(&unsatisfy_lits);
        clauses2python+=convert2string(unsatisfy_lits);
        total_clause_size += unsatisfy_lits.size();

        if (unsatisfy_lits.size() == 3) auxiNum++;

        vector<Var> vars1, vars2;
        vector<Var> *unsatisfy_vars_p = &vars1;
        vector<Var> *next_level_vars_p = &vars2;

        for (int j = 0; j < unsatisfy_lits.size(); j++) {
            Var variable = var(unsatisfy_lits[j]);
            if (var_visit[variable] == 1) continue;
            unsatisfy_vars_p->push_back(variable);
            added_vars.push_back(variable);
            var_visit[variable] = 1;
        }

        while (unsatisfy_vars_p->size() != 0 && added_vars.size() < MAX_VAR_NUM &&
               added_clauses.size() < MAX_CLAUSE_NUM) {
            vector<Var> &unsatisfy_vars = *unsatisfy_vars_p;
            vector<Var> &next_level_vars = *next_level_vars_p;

            for (int j = 0; j < unsatisfy_vars.size() && added_vars.size() < MAX_VAR_NUM &&
                            added_clauses.size() < MAX_CLAUSE_NUM; j++) {
                Var variable = unsatisfy_vars[j];
                vector<vec<Lit> *> related_clauses = *var2clauses[variable];
                for (int k = 0; k < related_clauses.size() && added_vars.size() < MAX_VAR_NUM; k++) {
                    if (related_clauses[k]->getmark() == 2) {
                        continue;
                    }
                    related_clauses[k]->setmark(2);
                    vec<Lit> &cur_unsatisfy_lits = *(related_clauses[k]);

                    addClause2Solver(cur_unsatisfy_lits, minisat_solver);
                    addClause2Simulator(related_clauses[k], simulator);
                    added_clauses.push_back((&cur_unsatisfy_lits));
                    total_clause_size += cur_unsatisfy_lits.size();
                    clauses2python+=convert2string(cur_unsatisfy_lits);
                    for (int l = 0; l < cur_unsatisfy_lits.size(); l++) {
                        Var v = var(cur_unsatisfy_lits[l]);
                        if (var_visit[v] == 0) {
                            next_level_vars.push_back(v);
                            added_vars.push_back(v);
                            var_visit[v] = 1;
                        }
                    }
                    if (cur_unsatisfy_lits.size() == 3) auxiNum++;
                }
            }

            vector<Var> *temp = unsatisfy_vars_p;
            unsatisfy_vars_p = next_level_vars_p;
            next_level_vars_p = temp;
            next_level_vars_p->clear();
        }
    }

    bool PRINT_PROCESS = S.PRINT_PROCESS && S.verbosity > 0; //true;

    int unassigned_var_num = 0;
    for(Var v = 0; v < var2clauses.size(); v++){
        if(S.value(v)== l_Undef){
            unassigned_var_num++;
        }
    }

    if(added_clauses.size()==0)
        return true;

    bool success = false;
    minisat_solver.random_seed = rand();
    map<Var, lbool> success_Lit;
    success = dwave(clauses2python, success_Lit, S.use_realQC);


    S.solveNum++;

    if (success) {   //如果是true那一定是true的

        map<Var,lbool> successLit;
        successLit = success_Lit;
        if (all_unsatisfy_lits.size() == added_clauses.size() && S.quantum_effect) {
            //  all the variable remained are deduced
            if(do_one_time_solve){
                if (PRINT_PROCESS)
                    printf("ALL success");

                while(!S.force_decision_lits.empty())
                    S.force_decision_lits.pop();

                for (auto item : successLit) {
                    Var v = item.first;
                    lbool value = item.second;
                    if (S.value(v) != l_Undef) {
                        continue;
                    }
                    Lit lit = (value == l_True) ? mkLit(v) : ~mkLit(v);
                    if (S.quantum_effect) {
                        S.force_decision_lits.push(lit);
                    }
                }
            }

      }

      else if(do_set_phase){
            bool overlap = false;
//            for (auto item : successLit) {
//                Var v = item.first;
//                if (S.value(v) != l_Undef)
//                    continue;
//                if(S.user_pol[v] != l_Undef){
//                    overlap = true;
//                    break;
//                }
//            }
            if(!overlap){
                for (auto item : successLit) {  // todo: it should not be nVars, should be variable invoved
                    Var v = item.first;
                    lbool value = item.second;
                    if (S.value(v) != l_Undef)
                        continue;
//                    if(S.user_pol[v] != l_Undef)
//                        continue;
                    if (S.quantum_effect)
                        S.user_pol[v] = value;
                }
            }else{
//                    bool temp = true;
            }
        }
// */
        return true;
    } else {  // 如果是false, 有可能是错的
        if (PRINT_PROCESS)
            printf("quantum detect conflict\n");

        // 记录一下这一层的conflict数量
        int cur_level = S.decisionLevel();
        if(S.level2conficts.find(cur_level)!=S.level2conficts.end()){
            S.level2conficts[cur_level] ++;
        }else{
            S.level2conficts[cur_level] = 1;
        }
//        return false;

        if(S.quantum_effect && do_backtrack){

//            vec<Lit>& trail = S.trail;
//            vec<int>& trail_lim = S.trail_lim;
//            for(int i = trail.size()-1; i>=0; i--){
//                Lit lit = trail[i];
//                Var v = var(lit);
//            }

//          method 1
//            int unadded_var_num = var2clauses.size() - added_vars.size();
//            vector<VarActivity> var_activity(added_vars.size());
//            for(int i =0; i<added_vars.size();i++){
//                var_activity[i].v = added_vars[i];
//                var_activity[i].activity = S.activity[added_vars[i]];
//            }
//            sort(var_activity.begin(), var_activity.end(), var_cmp);  //TODO: 按照activity sort
//
//            while(!S.force_pick_vars.empty())
//                S.force_pick_vars.pop();
//
//            for(int i =0; i<added_vars.size();i++){
////                cout << var_activity[i].v << " " << var_activity[i].activity << endl;
//                S.force_pick_vars.push(var_activity[i].v);
//            }

//            if(PRINT_PROCESS)   //|| true
//                cout << "meet conflict and push" << len()

//            method 2
//            while(!S.force_decision_lits.empty())
//                S.force_decision_lits.pop();
//
//            // 这么写很可能会被马上的propogate掉
//            for(int i = 0; i < minisat_solver.current_conflict_clause->size(); i++){
//                Lit lit = (*minisat_solver.current_conflict_clause)[i];
//                S.force_decision_lits.push(~lit);
//            }

//             method 3, noise free
//            //  todo: now it will cause to many clause but do not very useful, making the restart so frequency
            int current_level = S.decisionLevel();

            int current_pick_index = -1;
            for (Var v = 0; v < S.nVars(); v++) {  //some of it can not be the reason of the first variable
                if(S.value(v) == l_Undef) continue;
                Lit lit = S.value(v) == l_True? mkLit(v) : ~mkLit(v);
                if(S.reason(v) == CRef_Undef){
                    learnt_clause.push(~lit);
                    S.varBumpActivity(var(lit));
                    if(S.level(v) == current_level){
                        current_pick_index = learnt_clause.size() - 1;
                    }
                }
            }

            if(S.correct_answer){
                bool satisfy = false;
                for(int i = 0; i < learnt_clause.size(); i++){
                    Lit lit = learnt_clause[i];
                    if( (sign(lit)? l_False: l_True) == (*S.correct_answer)[var(lit)]){
                        satisfy = true;
                    }
                }
                assert(satisfy);
            }

            if(current_pick_index == -1){
                return false;
            }
            Lit temp = learnt_clause[0];
            learnt_clause[0] = learnt_clause[current_pick_index];
            learnt_clause[current_pick_index] =  temp;

            backtrack_level = current_level - 1;

            assert(S.level(var(learnt_clause[0])) == current_level);

            if(PRINT_PROCESS)   //|| true
                printf("learn clause %s through trail, plan to backtrack to %d, then pick %s(%d)\n", clause2string(learnt_clause).c_str(), backtrack_level,
                       lit2string(learnt_clause[0]).c_str(), S.level(var(learnt_clause[0])));
        }

        return false;
    }
}