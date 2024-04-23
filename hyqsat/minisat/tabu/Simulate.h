//
// Created by qc1 on 2022/4/11.
//

#ifndef MINISAT_SIMULATE_H
#define MINISAT_SIMULATE_H


#include <iostream>
#include "minisat/core/SolverTypes.h"
#include <map>
#include "tabu_search.h"

using namespace std;
using namespace Minisat;

class Simulate {
public:
    Simulate();
    vec<vec<Lit>* > clauses;
    vec<Var> variabls;
    vec<Lit> lits;
    map<Var, lbool> successLit;
    void addClause(vec<Lit> *clause);
    void deleteAllClauses();
    void setAllclauses(vec<vec<Lit>* > addClauses);
//    vec<Clause*> getAllclauses();
    bool litSatisfy(Lit p, lbool v);
    void quantumSample(vector<TabuSearch *> &samplers, vector<vector<double>> &Q, int sample_number);
    bool solve(int auxiNum);
    map<Var,lbool> getSuccessLit();
};

Simulate::Simulate() {}

void Simulate:: addClause(vec<Lit> *clause){
    clauses.push(clause);
};

void Simulate::deleteAllClauses(){
    clauses.clear();
}

void Simulate::setAllclauses(vec<vec<Lit>* > addClauses){
    for(int i = 0; i<addClauses.size(); i++){
        clauses.push(addClauses[i]);
    }
}
map<Var,lbool> Simulate::getSuccessLit() {
    return successLit;
}
//vec<Clause*> Simulate::getAllclauses(){
//    return clauses;
//}

bool Simulate::litSatisfy(Lit p, lbool v) {
    return (v ^ sign(p)) == l_True;
}

void Simulate::quantumSample(vector<TabuSearch *> &samplers, vector<vector<double>> &Q, int sample_number) {
    vector<TabuSearch *> temp_samplers(sample_number);
    for (int t = 0; t < sample_number; t++) {
        int rand_int = rand();
        int var_num = Q.size();
        int initial_state_int = rand_int % var_num;
        vector<int> initial_state(var_num,0);
        int var_value, var_index = 0;
        while (initial_state_int && var_index<var_num) {
            var_value = (initial_state_int % 2);
            initial_state[var_index] = var_value;
            initial_state_int = initial_state_int / 2;
            var_index++;
        }
        TabuSearch *sampler = new TabuSearch(Q, initial_state, 0, (long) 60, 10000000, rand(), -9999999);
        temp_samplers[t] = sampler;
    }

    double MAX = 999999999;
    double best_energy = MAX;
//    int BORDER = 300;
    for (int i = 0; i < temp_samplers.size(); i++) {
        double energy = temp_samplers[i]->bestEnergy();
//        cout<<energy<<endl;
        if (energy < best_energy) {
            assert(energy < MAX);
            for (int j = 0; j < samplers.size(); j++) {
                delete samplers[j];
            }
            samplers.clear();
            best_energy = energy;
        }

        if (energy == best_energy) {
            samplers.push_back(temp_samplers[i]);
        } else {
            delete temp_samplers[i];
        }
    }
}

bool Simulate::solve(int auxiNum){
    map<int,int> var2index;
    map<int,int> index2var;
    int index = 0;

    for(int i = 0; i<clauses.size(); i++){
        vec<Lit> &clause = (*clauses[i]);
        for(int i = 0; i<clause.size(); i++){
            int v = var(clause[i]);
            if(var2index.find(v)==var2index.end()){
                var2index[v] = index;
                index2var[index] = v;
                index++;
            }
        }
    }

    int auxiindex = index;
    vector<vector<double> > Q(index+auxiNum+1,vector<double>(index+auxiNum+1,0));
    int should_best_energy = 0;
    for (int i = 0; i<clauses.size(); i++){
        vec<Lit> &clause = (*clauses[i]);
        Lit l0, l1, la,l2;
        int s0,s1,sa,s2;
        int i0,i1,ia,i2;
        Var v0,v1,va,v2;
        if(clause.size()==1){
            v0 = var(clause[0]);
            s0 = !sign(clause[0]);
            i0 = var2index[v0];
            if(s0){
                Q[i0][i0] -= 1;
                should_best_energy++;
            }
            else{
                Q[i0][i0] += 1;
            }
        }
        else if(clause.size()==2){
            l0 = clause[0], l1 = clause[1];
            s0 = !sign(l0), s1 = !sign(l1);
            v0 = var(l0), v1 = var(l1);
            i0 = var2index[v0], i1 = var2index[v1];
//            cout<<i0<<' '<<i1<<endl;
            // true for positive
            if(s0 && s1){
                Q[i0][i0] -= 1;
                Q[i1][i1] -= 1;
                Q[i0][i1] += 1;
                Q[i1][i0] += 1;
                should_best_energy += 1; // -=?
            }else if(!s0 && s1){
                Q[i0][i0] += 1;
                Q[i0][i1] -= 1;
                Q[i1][i0] -= 1;
            }else if(s0 && !s1){
                Q[i1][i1] += 1;
                Q[i0][i1] -= 1;
                Q[i1][i0] -= 1;
            }else if(!s0 && !s1){
                Q[i0][i1] += 1;
                Q[i1][i0] += 1;
            }
        }
        else if(clause.size()==3){
            l0 = clause[0], l1 = clause[1], l2 = clause[2];
            ia = auxiindex;
            auxiindex++;
            // sa = ((!sign(clause[0])) || (!sign(clause[1]))); //todo: false is positive?
            v0 = var(l0);
            i0 = var2index[v0];
            s0 = !sign(l0);
            v1 = var(l1);
            i1 = var2index[v1];
            s1 = !sign(l1);
            v2 = var(l2);
            i2 = var2index[v2];
            s2 = !sign(l2);
//            cout<<i0<<' '<<i1<<' '<<i2<<' '<<ia<<endl;
            if(s0 && s1){ //sa must be true
                Q[ia][ia] += 1;
                Q[i0][i0] += 1;
                Q[i1][i1] += 1;
                Q[ia][i0] -= 2;
                Q[i0][ia] -= 2;
                Q[ia][i1] -= 2;
                Q[i1][ia] -= 2;
                Q[i0][i1] += 1;
                Q[i1][i0] += 1;
            }
            else if(!s0 && s1){
                Q[ia][ia] -= 1;
                Q[i0][i0] -= 1;
                Q[i1][i1] += 2;
                Q[ia][i0] += 2;
                Q[i0][ia] += 2;
                Q[ia][i1] -= 2;
                Q[i1][ia] -= 2;
                Q[i0][i1] -= 1;
                Q[i1][i0] -= 1;
                should_best_energy += 1;
            }
            else if(s0 && !s1){
                Q[ia][ia] -= 1;
                Q[i0][i0] += 2;
                Q[i1][i1] -= 1;
                Q[ia][i0] -= 2;
                Q[i0][ia] -= 2;
                Q[ia][i1] += 2;
                Q[i1][ia] += 2;
                Q[i0][i1] -= 1;
                Q[i1][i0] -= 1;
                should_best_energy += 1;
            }
            else if(!s0 && !s1){
                Q[ia][ia] -= 3;
                Q[i0][i0] -= 2;
                Q[i1][i1] -= 2;
                Q[ia][i0] += 2;
                Q[i0][ia] += 2;
                Q[ia][i1] += 2;
                Q[i1][ia] += 2;
                Q[i0][i1] += 1;
                Q[i1][i0] += 1;
                should_best_energy += 3;
            }
            sa = true;
            if(sa && s2){
                Q[ia][ia] -= 1;
                Q[i2][i2] -= 1;
                Q[ia][i2] += 1;
                Q[i2][ia] += 1;
                should_best_energy += 1;
            }else if(sa && !s2) {
                Q[i2][i2] += 1;
                Q[ia][i2] -= 1;
                Q[i2][ia] -= 1;
            }
            else if(!sa && s2){
                Q[ia][ia] += 1;
                Q[ia][i2] -= 1;
                Q[i2][ia] -= 1;
            }
            else{
                Q[ia][i2] += 1;
                Q[i2][ia] += 1;
            }

        }

    }

    vector<TabuSearch*> samplers;
    quantumSample(samplers, Q, 10);  // not always true

    for(int i = 0; i < samplers.size(); i++){
        TabuSearch& sampler = *samplers[i];
        vector<int> solution = sampler.bestSolution();
        double best_energy = sampler.bestEnergy();

//        cout<<best_energy<< ' ' <<should_best_energy << endl;
        map <Var, lbool> var2assign;
        map <Var, bool> var2bool;

        for(int j = 0; j < solution.size(); j++){
            Var variable = index2var[j];
            var2assign[variable] = (solution[j] == 0)? l_True: l_False;
            var2bool[variable] = (solution[j] == 0);
        }

        bool satisfy = true;
        for (int i = 0; i<clauses.size(); i++) {
            vec<Lit>& clause = (*clauses[i]);
            bool caluse_satisfy = false;
            for (int k = 0; k < clause.size(); k++) {
                Var variable = var(clause[k]);
                bool var_satisfy = litSatisfy(clause[k], var2assign[variable]);
                if (var_satisfy) {
                    caluse_satisfy = true; //this clause is satisfy
                    break;
                }
            }

            if (!caluse_satisfy){
                satisfy = false;
                break;
            }
        }

        if(satisfy){
            successLit = var2assign;

            return true;// in this condition, all clauses are satisfy
            //如果有可满足情况就返回true，如果所有solution都是不满足的，就发现了冲突；
        }

    }
    return false;
}

#endif //MINISAT_SIMULATE_H
