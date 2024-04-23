
/***************************************************************************************[Solver.cc]
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

#include <math.h>
#include <stdio.h>

#include "minisat/mtl/Alg.h"
//#include "minisat/mtl/Sort.h"
#include "minisat/utils/System.h"
#include "minisat/core/Solver.h"
#include "minisat/mtl/Vec.h"
#include <algorithm>

//#include "minisat/tabu/sat2solver.cpp"

#include "minisat/tabu/solve_sat_2.h"
#include <iostream>

using namespace Minisat;
using namespace std;

//=================================================================================================
// Options:


static const char *_cat = "CORE";

static DoubleOption opt_var_decay(_cat, "var-decay", "The variable activity decay factor", 0.95,
                                  DoubleRange(0, false, 1, false));
static DoubleOption opt_clause_decay(_cat, "cla-decay", "The clause activity decay factor", 0.999,
                                     DoubleRange(0, false, 1, false));
static DoubleOption opt_random_var_freq(_cat, "rnd-freq",
                                        "The frequency with which the decision heuristic tries to choose a random variable",
                                        0, DoubleRange(0, true, 1, true));
static DoubleOption opt_random_seed(_cat, "rnd-seed", "Used by the random variable selection", 91648253,
                                    DoubleRange(0, false, HUGE_VAL, false));
static IntOption opt_ccmin_mode(_cat, "ccmin-mode", "Controls conflict clause minimization (0=none, 1=basic, 2=deep)",
                                2, IntRange(0, 2));
static IntOption opt_phase_saving(_cat, "phase-saving",
                                  "Controls the level of phase saving (0=none, 1=limited, 2=full)", 0, IntRange(0, 2));
static BoolOption opt_rnd_init_act(_cat, "rnd-init", "Randomize the initial activity", false);
static BoolOption opt_luby_restart(_cat, "luby", "Use the Luby restart sequence", true);
static IntOption opt_restart_first(_cat, "rfirst", "The base restart interval", 100, IntRange(1, INT32_MAX));
static DoubleOption opt_restart_inc(_cat, "rinc", "Restart interval increase factor", 2,
                                    DoubleRange(1, false, HUGE_VAL, false));
static DoubleOption opt_garbage_frac(_cat, "gc-frac",
                                     "The fraction of wasted memory allowed before a garbage collection is triggered",
                                     0.20, DoubleRange(0, false, HUGE_VAL, false));
static IntOption opt_min_learnts_lim(_cat, "min-learnts", "Minimum learnt clause limit", 0, IntRange(0, INT32_MAX));

bool compareClauseBasedOnActivity(Clause *a, Clause *b) {
    return a->activity() > b->activity(); //ascending order
};
//=================================================================================================
// Constructor/Destructor:


Solver::Solver() :

// Parameters (user settable):
//
        verbosity(0), var_decay(opt_var_decay), clause_decay(opt_clause_decay), random_var_freq(opt_random_var_freq),
        random_seed(opt_random_seed), luby_restart(opt_luby_restart), ccmin_mode(opt_ccmin_mode),
        phase_saving(opt_phase_saving), rnd_pol(false), rnd_init_act(opt_rnd_init_act), garbage_frac(opt_garbage_frac),
        min_learnts_lim(opt_min_learnts_lim), restart_first(opt_restart_first), restart_inc(opt_restart_inc)

        // Parameters (the rest):
        //
        , learntsize_factor((double) 1 / (double) 3), learntsize_inc(1.1)

        // Parameters (experimental):
        //
        , learntsize_adjust_start_confl(100), learntsize_adjust_inc(1.5)

        // Statistics: (formerly in 'SolverStats')
        //
        , solves(0), starts(0), decisions(0), rnd_decisions(0), propagations(0), conflicts(0), dec_vars(0),
        num_clauses(0), num_learnts(0), clauses_literals(0), learnts_literals(0), max_literals(0), tot_literals(0),
        watches(WatcherDeleted(ca)), order_heap(VarOrderLt(activity)), ok(true), cla_inc(1), var_inc(1), qhead(0),
        simpDB_assigns(-1), simpDB_props(0), progress_estimate(0), remove_satisfied(true), next_var(0)

        // Resource constraints:
        //
        , conflict_budget(-1), propagation_budget(-1), asynch_interrupt(false), use_quantum(false),
        current_conflict_clause(NULL), quantum_time_cost(0), quantum_count(0), my_start_time(getSysTimeMicros()),
        conflict_analysis_timecost(0), propagations_timecost(0), decision_timecost(0), quantum_success_number(0),
        quantum_conflict_number(0), PRINT_PROCESS(false), quantum_total_number(0), DLIS_timecost(0),
        quantum_onetime_solve_number(0), quantum_effect(false), correct_answer(NULL),propagate_error(false),print_clause(false)
{}


Solver::~Solver() {
}


//=================================================================================================
// Minor methods:


// Creates a new SAT variable in the solver. If 'decision' is cleared, variable will not be
// used as a decision variable (NOTE! This has effects on the meaning of a SATISFIABLE result).
//
Var Solver::newVar(lbool upol, bool dvar) {
    Var v;
    if (free_vars.size() > 0) {
        v = free_vars.last();
        free_vars.pop();
    } else
        v = next_var++;
    watches.init(mkLit(v, false));
    watches.init(mkLit(v, true));
    sat2_activity.insert(v, 0);
    assigns.insert(v, l_Undef);
    vardata.insert(v, mkVarData(CRef_Undef, 0));
    activity.insert(v, rnd_init_act ? drand(random_seed) * 0.00001 : 0);
    seen.insert(v, 0);
    polarity.insert(v, true);
    user_pol.insert(v, upol);
    decision.reserve(v);
    trail.capacity(v + 1);
    setDecisionVar(v, dvar);
    return v;
}


// Note: at the moment, only unassigned variable will be released (this is to avoid duplicate
// releases of the same variable).
void Solver::releaseVar(Lit l) {
    if (value(l) == l_Undef) {
        addClause(l);
        released_vars.push(var(l));
    }
}


bool Solver::addClause_(vec<Lit> &ps) {
    vec<Lit> original_ps;
    ps.copyTo(original_ps);
//     for(int i = 0; i<original_ps.size(); i++){
//         cout<<var(original_ps[i])<<' ';
//     }
//     cout<<endl;
    assert(ps.size() != 0);
    assert(decisionLevel() == 0);
    if (!ok)
        return false;

    // Check if clause is satisfied and remove false/duplicate literals:
    sort(ps);
    Lit p;
    int i, j;
    for (i = j = 0, p = lit_Undef; i < ps.size(); i++)
        if (value(ps[i]) == l_True || ps[i] ==~p) // two types of ensuring the clause is always true, clause with true value does not need to be added
            return true;
        else if (value(ps[i]) != l_False && ps[i] != p)
            ps[j++] = p = ps[i];
    ps.shrink(i - j);

    if (ps.size() == 0) { // can not satisfy
//        printClause(ps);
//        printf("WARNING: can bot be satisfy after adding Clause\n");
        original_ps.copyTo(current_conflict_clause_instance);
        current_conflict_clause = &current_conflict_clause_instance; //todo: should be remember in the future
        return ok = false;
    } else if (ps.size() == 1) {  // can be applied to propagate
        uncheckedEnqueue(ps[0]);
        ok = (propagate() == CRef_Undef);
        if (!ok) {
            original_ps.copyTo(current_conflict_clause_instance);
            current_conflict_clause = &current_conflict_clause_instance; //todo: should be remember in the future
//            printf("WARNING: can bot be satisfy after adding Clause\n");
        }
        return ok;
    } else {
        CRef cr = ca.alloc(ps); //false
//        printClause(ca[cr]);
//        assert(ca[cr].activity() == 0);
        clauses.push(cr);
        attachClause(cr);
//        assert(ca[cr].activity() == 0);
    }

    return true;
}


void Solver::attachClause(CRef cr) {
    const Clause &c = ca[cr];
    assert(c.size() > 1);
    watches[~c[0]].push(Watcher(cr, c[1]));
    watches[~c[1]].push(Watcher(cr, c[0]));
    if (c.learnt()) num_learnts++, learnts_literals += c.size();
    else num_clauses++, clauses_literals += c.size();
}


void Solver::detachClause(CRef cr, bool strict) {
    const Clause &c = ca[cr];
    assert(c.size() > 1);
    // Strict or lazy detaching:
    if (strict) {
        remove(watches[~c[0]], Watcher(cr, c[1]));
        remove(watches[~c[1]], Watcher(cr, c[0]));
    } else {
        watches.smudge(~c[0]);
        watches.smudge(~c[1]);
    }

    if (c.learnt()) num_learnts--, learnts_literals -= c.size();
    else num_clauses--, clauses_literals -= c.size();
}


void Solver::removeClause(CRef cr) {
    Clause &c = ca[cr];
    detachClause(cr);
    // Don't leave pointers to free'd memory!
    if (locked(c)) vardata[var(c[0])].reason = CRef_Undef;
    c.mark(1);
    ca.free(cr);
}


bool Solver::satisfied(const Clause &c) const {
    for (int i = 0; i < c.size(); i++)
        if (value(c[i]) == l_True)
            return true;
    return false;
}


// Revert to the state at given level (keeping all assignment at 'level' but not beyond).
void Solver::cancelUntil(int level) {
    if (decisionLevel() > level && level != -1) {
        for (int c = trail.size() - 1; c >= trail_lim[level]; c--) {
            Var x = var(trail[c]);
            assigns[x] = l_Undef;
            if (phase_saving > 1 || (phase_saving == 1 && c > trail_lim.last()))
                polarity[x] = sign(trail[c]);  //store the preferred value of variable x
            insertVarOrder(x);
        }
        qhead = trail_lim[level];
        trail.shrink(trail.size() - trail_lim[level]);
        trail_lim.shrink(trail_lim.size() - level);
    }else if(level == -1){
        for (int c = trail.size() - 1; c >= 0; c--) {
            Var x = var(trail[c]);
            assigns[x] = l_Undef;
//            because we only backtrack one step, I do not think we should remember the phase_saving
            if (phase_saving > 1 || (phase_saving == 1 && c > trail_lim.last()))
                polarity[x] = sign(trail[c]);  //store the preferred value of variable x
            insertVarOrder(x);
        }
        qhead = 0;
        trail.shrink(trail.size() - trail_lim[level]);
        trail_lim.shrink(trail_lim.size() );
    }else{
    }

}

//=================================================================================================
// Major methods:


Lit Solver::pickBranchLit() {
    Var next = var_Undef;

    if (!force_pick_vars.empty()) {
        next = force_pick_vars.front();
        force_pick_vars.pop();
        while(value(next) != l_Undef and !force_pick_vars.empty()){
            next = force_pick_vars.front();
            force_pick_vars.pop();
        }
        if(value(next) != l_Undef){
            next = var_Undef;
        }
    }

    // Random decision:
    if (next == var_Undef && drand(random_seed) < random_var_freq && !order_heap.empty()) {
        next = order_heap[irand(random_seed, order_heap.size())];
        if (value(next) == l_Undef && decision[next])
            rnd_decisions++;
    }

    while (next == var_Undef || value(next) != l_Undef || !decision[next]){   //todo: what's decision?
        if (order_heap.empty()){
            next = var_Undef;
            break;
        }else{
            next = order_heap.removeMin();  // order heap do not only obey the activity if there are insert order heap it will use the var insert at first
        }
    }

    if (next == var_Undef)
        return lit_Undef;
    else if(value(next)!=l_Undef){
        return lit_Undef;
    }
    else if (user_pol[next] != l_Undef) {
        Lit assignment = mkLit(next, user_pol[next] == l_True);
//        user_pol[next] = l_Undef;
//        assert(false);
        return assignment;
    } else if (rnd_pol)
        return mkLit(next, drand(random_seed) < 0.5);
    else {
        Lit lit = mkLit(next, polarity[next]);
//        cout << lit2string(lit) << endl;
        return lit;
    }

}

// Siwei: actually it's different from the DLIS, it does not consider the negative or positive of the varibale
Var Solver::pickBranchVarDLIS() {
    vec<CRef *> all_clauses;
    IntMap<Var, int> var2count;
    for (int i = 0; i < nVars(); i++) {
        var2count.insert(i, 0);
    }

    combineList(clauses, learnts, all_clauses);
    vec<Lit> un_satisfy_lits;
    for (int i = 0; i < all_clauses.size(); i++) {
        CRef &clause_ref = *(all_clauses[i]);
        Clause &clause = ca[clause_ref];

        un_satisfy_lits.clear();
        bool is_satisfy = false;
        for (int j = 0; j < clause.size(); j++) {
            lbool var_satisfy = value(clause[j]);
            if (var_satisfy == l_True) {
                is_satisfy = true;
                break;
            }
            if (var_satisfy == l_Undef) {
                un_satisfy_lits.push(clause[j]);
            }
        }

        if (is_satisfy || un_satisfy_lits.size() <= 2)
            continue;

        for (int j = 0; j < un_satisfy_lits.size(); j++) {
            var2count[var(un_satisfy_lits[j])]++;
        }
    }

    Var max_var = -1;
    int max_num = -1;
    for (int i = 0; i < nVars(); i++) {
        if (max_num < var2count[i]) {
            max_var = i;
            max_num = var2count[i];
        }
    }

    assert(max_var != -1);
    return max_var;
}


/*_________________________________________________________________________________________________
|
|  analyze : (confl : Clause*) (out_learnt : vec<Lit>&) (out_btlevel : int&)  ->  [void]
|
|  Description:
|    Analyze conflict and produce a reason clause.
|
|    Pre-conditions:
|      * 'out_learnt' is assumed to be cleared.
|      * Current decision level must be greater than root level.
|
|    Post-conditions:
|      * 'out_learnt[0]' is the asserting literal at level 'out_btlevel'.
|      * If out_learnt.size() > 1 then 'out_learnt[1]' has the greatest decision level of the
|        rest of literals. There may be others from the same level though.
|
|________________________________________________________________________________________________@*/
void Solver::analyze(CRef confl, vec<Lit> &out_learnt, int &out_btlevel) {
    int pathC = 0;
    Lit p = lit_Undef;

    // Generate conflict clause:
    //
    out_learnt.push();      // (leave room for the asserting literal)
    int index = trail.size() - 1;

    do {
        assert(confl != CRef_Undef); // (otherwise should be UIP)
        Clause &c = ca[confl];

//        string conf_clause_str = clause2string(c, this);
//        string learnt_str = clause2string(out_learnt, this);
        if (c.learnt())
            claBumpActivity(c);
//        else
//            claBumpActivity(c);  // siwei

        for (int j = (p == lit_Undef) ? 0 : 1; j < c.size(); j++) {
            Lit q = c[j];

            if (!seen[var(q)] && level(var(q)) > 0) {  // level(var(q)) < 0 means q is the root
                varBumpActivity(var(q)); // for heuristics of decision
//                assert(seen[var(q)] !=  1);
                seen[var(q)] = 1;
                assert(level(var(q)) <= decisionLevel());
                if (level(var(q)) >= decisionLevel()){ //I think should be ==, do not know why, to find uip?
//                    printf("pathC++ %d %d level: %d, decision level: %d\n", var(q), pathC, level(var(q)), decisionLevel());
                    pathC++;
                }else
                    out_learnt.push(q);
                // literals that do not belong to the decision level are all put in the learnt clause
            }
        }

        //  siwei: to get all the 2-sat learnts clause, do not show speed up, also do not become slower
//        if(out_learnt.size() == 2 && p != lit_Undef && use_quantum){
//            vec<Lit>& sat2learnt = *genVecLit();  //*(new vec<Lit>()); //
//            out_learnt.copyTo(sat2learnt);
//            sat2learnt[0] = ~p;
//            sat2learnts.push_back(&sat2learnt);
//        }

        // Select next clause to look at:
        while (!seen[var(trail[index--])]);  // start from the next assigment
        // seen determine whether a var is analyzed
        p = trail[index + 1];
        assert(level(var(p)) == decisionLevel());
        confl = reason(var(p));  // return clause determine the value of p

        // all the variable deduced is 1, because it is deduced by a unit clause, literal neg x => 0, literal x => 1 (do not know what I want to say?)
        seen[var(p)] = 0;
        pathC--;
//        printf("pathC-- %d %d level:%d decision level:%d\n", var(p), pathC, level(var(p)), decisionLevel());

//
        if (confl == CRef_Undef && pathC != 0) {  // it means that there is a circle, a variable have been enqueued twice
            printf("p: %d %d\n", var(p), pathC);
            for (Var v = 0; v < nVars(); ++v) {
                if(value(v) != l_Undef && level(v) == decisionLevel()){
                    printf("%d %d %s\n", v, decisionLevel(), reason(v) != CRef_Undef? clause2string(ca[reason(v)]).c_str(): "None");
                }
            }
            assert(false);
        }
    } while (pathC > 0);  //not UIP  //todo: important do not know why should add confl == CRef_Undef   && confl != CRef_Undef
    out_learnt[0] = ~p; // out_learnt[0] is null due to the out_learnt.push() at first
//    so p is 1-UIP
//    printClause(out_learnt);



    // Simplify conflict clause:

    int i, j;
    out_learnt.copyTo(analyze_toclear);
    if (ccmin_mode == 2) {  // ccmin_mode: Controls conflict clause minimization (0=none, 1=basic, 2=deep).
        for (i = j = 1; i < out_learnt.size(); i++)
            if (reason(var(out_learnt[i])) == CRef_Undef || !litRedundant(out_learnt[i]))
                out_learnt[j++] = out_learnt[i];

    } else if (ccmin_mode == 1) {
        for (i = j = 1; i < out_learnt.size(); i++) {
            Var x = var(out_learnt[i]);

            if (reason(x) == CRef_Undef)
                out_learnt[j++] = out_learnt[i];
            else {
                Clause &c = ca[reason(var(out_learnt[i]))];
                for (int k = 1; k < c.size(); k++)
                    if (!seen[var(c[k])] && level(var(c[k])) > 0) {
                        out_learnt[j++] = out_learnt[i];
                        break;
                    }
            }
        }
    } else
        i = j = out_learnt.size();

    max_literals += out_learnt.size();
    out_learnt.shrink(i - j);
    tot_literals += out_learnt.size();

    // Find correct backtrack level:
    //
    if (out_learnt.size() == 1)
        out_btlevel = 0;
    else {
        int max_i = 1;
        // Find the first literal assigned at the next-highest level:
        for (int i = 2; i < out_learnt.size(); i++)
            if (level(var(out_learnt[i])) > level(var(out_learnt[max_i])))
                max_i = i;
        // Swap-in this literal at index 1:
        Lit p = out_learnt[max_i];
        out_learnt[max_i] = out_learnt[1];
        out_learnt[1] = p;
        out_btlevel = level(var(p));
    }

    for (int j = 0; j < analyze_toclear.size(); j++) seen[var(analyze_toclear[j])] = 0;    // ('seen[]' is now cleared)
}

//do not quite understand ??
// Check if 'p' can be removed from a conflict clause.
bool Solver::litRedundant(Lit p) {
    enum {
        seen_undef = 0, seen_source = 1, seen_removable = 2, seen_failed = 3
    };
    assert(seen[var(p)] == seen_undef || seen[var(p)] == seen_source);
    assert(reason(var(p)) != CRef_Undef);

    Clause *c = &ca[reason(var(p))];  // why
    vec<ShrinkStackElem> &stack = analyze_stack;
    stack.clear();

    for (uint32_t i = 1;; i++) {
        if (i < (uint32_t) c->size()) {
            // Checking 'p'-parents 'l':
            Lit l = (*c)[i];

            // Variable at level 0 or previously removable:
            if (level(var(l)) == 0 || seen[var(l)] == seen_source || seen[var(l)] == seen_removable) {
                continue;
            }

            // Check variable can not be removed for some local reason:
            if (reason(var(l)) == CRef_Undef || seen[var(l)] == seen_failed) {
                stack.push(ShrinkStackElem(0, p));
                for (int i = 0; i < stack.size(); i++)
                    if (seen[var(stack[i].l)] == seen_undef) {
                        seen[var(stack[i].l)] = seen_failed;
                        analyze_toclear.push(stack[i].l);
                    }

                return false;
            }

            // Recursively check 'l':
            stack.push(ShrinkStackElem(i, p));
            i = 0;
            p = l;
            c = &ca[reason(var(p))];
        } else {
            // Finished with current element 'p' and reason 'c':
            if (seen[var(p)] == seen_undef) {
                seen[var(p)] = seen_removable;
                analyze_toclear.push(p);
            }

            // Terminate with success if stack is empty:
            if (stack.size() == 0) break;

            // Continue with top element on stack:
            i = stack.last().i;
            p = stack.last().l;
            c = &ca[reason(var(p))];

            stack.pop();
        }
    }

    return true;
}


/*_________________________________________________________________________________________________
|
|  analyzeFinal : (p : Lit)  ->  [void]
|
|  Description:
|    Specialized analysis procedure to express the final conflict in terms of assumptions.
|    Calculates the (possibly empty) set of assumptions that led to the assignment of 'p', and
|    stores the result in 'out_conflict'.
|________________________________________________________________________________________________@*/
void Solver::analyzeFinal(Lit p, LSet &out_conflict) {
    out_conflict.clear();
    out_conflict.insert(p);

    if (decisionLevel() == 0)
        return;

    seen[var(p)] = 1;

    for (int i = trail.size() - 1; i >= trail_lim[0]; i--) {
        Var x = var(trail[i]);
        if (seen[x]) {
            if (reason(x) == CRef_Undef) {
                assert(level(x) > 0);
                out_conflict.insert(~trail[i]);
            } else {
                Clause &c = ca[reason(x)];
                for (int j = 1; j < c.size(); j++)
                    if (level(var(c[j])) > 0)
                        seen[var(c[j])] = 1;
            }
            seen[x] = 0;
        }
    }

    seen[var(p)] = 0;
}


void Solver::uncheckedEnqueue(Lit p, CRef from)  // CRef is CRef_Undef by default
{
//    if (PRINT_PROCESS && verbosity > 0)
//        printf("uncheckedEnqueue %s from: %s at %d level\n", lit2string(p).c_str(),
//               from != CRef_Undef ? clause2string(ca[from]).c_str() : "None", decisionLevel());

    assert(value(p) == l_Undef);
    assigns[var(p)] = lbool(!sign(p));
    vardata[var(p)] = mkVarData(from, decisionLevel());  // Stores clause reasoned and level for each variable.
    trail.push_(p); // Assignment stack; stores all assigments made in the order they were made.
//    if (quantum_effect && (var(p) == 0 || var(p) == 99) && trail.size()>80){ //
//        cout<<"Push:"<<var(p)<<' '<<trail.size()<<endl;
//        for (int i=0; i<trail.size(); i++) cout<<var(trail[i])<<' ';
//        cout<<endl;
//    }
//    if(quantum_effect && (var(p) == 99)){
//        cout<<"Push:"<<var(p)<<' '<<trail.size()<<endl;
//    }
}


vec<Lit>* Solver::erase_lit(vec<Lit>* &a, Var target){
    vec<Lit>* newVec =  new vec<Lit>();
    for(int i = 0; i<a->size();i++){
        //cout<<&&a[i]<<endl;
        if(var((*a)[i])==target) continue;
        newVec->push((*a)[i]);
    }
    return newVec;
}


/*_________________________________________________________________________________________________
|
|  propagate : [void]  ->  [Clause*]
|
|  Description:
|    Propagates all enqueued facts. If a conflict arises, the conflicting clause is returned,
|    otherwise CRef_Undef.
|
|    Post-conditions:
|      * the propagation queue is empty, even if there was a conflict.
|________________________________________________________________________________________________@*/
CRef Solver::propagate() {
    CRef confl = CRef_Undef;
    int num_props = 0;

    while (qhead < trail.size()) {
        Lit p = trail[qhead++];     // 'p' is enqueued fact to propagate.
        vec<Watcher> &ws = watches.lookup(p);
        Watcher *i, *j, *end;
        num_props++;

        for (i = j = (Watcher *) ws, end = i + ws.size(); i != end;) {
            // Try to avoid inspecting the clause:
            Lit blocker = i->blocker;
            if (value(blocker) == l_True) {
                *j++ = *i++;
                continue;
            }

            // Make sure the false literal is data[1]:
            CRef cr = i->cref;
            Clause &c = ca[cr];

//            cout<<var(p)<<endl;
//            for(int i = 0; i<c.size();i++){
//                cout<<var(c[i])<<' ';
//            }
//            cout<<endl;

            Lit false_lit = ~p;
            if (c[0] == false_lit)
                c[0] = c[1], c[1] = false_lit;

            if(c[1] != false_lit){  //|| my_rand(0.2)
                confl = cr;
                this->propagate_error = true;
                return confl;
            }
            assert(c[1] == false_lit);
            i++;

            // If 0th watch is true, then clause is already satisfied.
            Lit first = c[0];
            Watcher w = Watcher(cr, first);
            if (first != blocker && value(first) == l_True) {
                *j++ = w;
                continue;
            }

            // Look for new watch:
            for (int k = 2; k < c.size(); k++)
                if (value(c[k]) != l_False) {
                    c[1] = c[k];
                    c[k] = false_lit;
                    watches[~c[1]].push(w);
                    goto NextClause;
                }

            // Did not find watch -- clause is unit under assignment:
            *j++ = w;
            if (value(first) == l_False) {
                confl = cr;
                qhead = trail.size();
                // Copy the remaining watches:
                while (i < end)
                    *j++ = *i++;
            } else
                uncheckedEnqueue(first, cr);

            NextClause:;
        }
        ws.shrink(i - j);
    }
    propagations += num_props;
    simpDB_props -= num_props;

    return confl;
}


/*_________________________________________________________________________________________________
|
|  reduceDB : ()  ->  [void]
|
|  Description:
|    Remove half of the learnt clauses, minus the clauses locked by the current assignment. Locked
|    clauses are clauses that are reason to some assignment. Binary clauses are never removed.
|________________________________________________________________________________________________@*/
struct reduceDB_lt {
    ClauseAllocator &ca;

    reduceDB_lt(ClauseAllocator &ca_) : ca(ca_) {}

    bool operator()(CRef x, CRef y) {
        return ca[x].size() > 2 && (ca[y].size() == 2 || ca[x].activity() < ca[y].activity());
    }
};

void Solver::reduceDB() {
    int i, j;
    double extra_lim = cla_inc / learnts.size();    // Remove any clause below this activity

    sort(learnts, reduceDB_lt(ca));
    // Don't delete binary or locked clauses. From the rest, delete clauses from the first half
    // and clauses with activity smaller than 'extra_lim':
    for (i = j = 0; i < learnts.size(); i++) {
        Clause &c = ca[learnts[i]];
//        if(c.size() == 2){
//            printClause(c);
//        }
//      siwei: now the size 2 learnts will not be removed, but it's very rare.
        if (c.size() > 2 && !locked(c) && (i < learnts.size() / 2 || c.activity() < extra_lim)) {
            removeClause(learnts[i]);
        } else
            learnts[j++] = learnts[i];
    }

//    todo: max2sat clause will not be garbage
//    if(learnt_clause.size()==2){
//        vec<Lit>& temp = *genVecLit();
//        learnt_clause.copyTo(temp);
//        sat2learnts.push_back(&temp);
//    }

    learnts.shrink(i - j);
    checkGarbage();
}


void Solver::removeSatisfied(vec<CRef> &cs) {
    int i, j;
    for (i = j = 0; i < cs.size(); i++) {
        Clause &c = ca[cs[i]];
        if (satisfied(c)){
//            bool flag = false;
//            if(var(c[0])==90 || var(c[0])==43){
//                printClause(c);
//                flag = true;
//            }
            removeClause(cs[i]);
//            if(flag){
//                printClause(c);
//            }
        }
        else {
            // Trim clause:

//            for (int k = 0; k < c.size(); k++){
//                cout<<var(c[k])<<' ';
//            }
//            cout<<endl;
//            assert(value(c[0]) == l_Undef && value(c[1]) == l_Undef);
            for (int k = 0; k < c.size(); k++) //todo:k=2?
                if (value(c[k]) == l_False) {
                    c[k--] = c[c.size() - 1];
                    c.pop();
                }
            cs[j++] = cs[i];
        }
    }
    cs.shrink(i - j);
}


void Solver::rebuildOrderHeap() {
    vec<Var> vs;
    for (Var v = 0; v < nVars(); v++)
        if (decision[v] && value(v) == l_Undef)
            vs.push(v);
    order_heap.build(vs);
}


/*_________________________________________________________________________________________________
|
|  simplify : [void]  ->  [bool]
|
|  Description:
|    Simplify the clause database according to the current top-level assigment. Currently, the only
|    thing done here is the removal of satisfied clauses, but more things can be put here.
|________________________________________________________________________________________________@*/
bool Solver::simplify() {
    assert(decisionLevel() == 0);

    if (!ok || propagate() != CRef_Undef)
        return ok = false;

    if (nAssigns() == simpDB_assigns || (simpDB_props > 0))
        return true;

    // Remove satisfied clauses:
    removeSatisfied(learnts);
    if (remove_satisfied) {       // Can be turned off.
        removeSatisfied(clauses);

        // TODO: what todo in if 'remove_satisfied' is false?

        // Remove all released variables from the trail:
//        what are released variables?
        for (int i = 0; i < released_vars.size(); i++) {
            assert(seen[released_vars[i]] == 0);
            seen[released_vars[i]] = 1;
        }

        int i, j;
        for (i = j = 0; i < trail.size(); i++)
            if (seen[var(trail[i])] == 0)
                trail[j++] = trail[i];
        trail.shrink(i - j);
        //printf("trail.size()= %d, qhead = %d\n", trail.size(), qhead);
        qhead = trail.size();

        for (int i = 0; i < released_vars.size(); i++)
            seen[released_vars[i]] = 0;

        // Released variables are now ready to be reused:
        append(released_vars, free_vars);
        released_vars.clear();
    }
    checkGarbage();
    rebuildOrderHeap();

    simpDB_assigns = nAssigns();
    simpDB_props = clauses_literals + learnts_literals;   // (shouldn't depend on stats really, but it will do for now)

    return true;
}

void Solver::clearForceDecisionLits() {
    while (!force_decision_lits.empty())
        force_decision_lits.pop();
}

bool Solver::clauseSatisfy2(vec<Lit> &unsatisfy_lits, Clause &clause) {
    unsatisfy_lits.clear();
    for (int j = 0; j < clause.size(); j++) {
        lbool var_satisfy = this->value(clause[j]);
        if (var_satisfy == l_True) {
            unsatisfy_lits.clear();
            return true;
        }
        if(var_satisfy == l_Undef){
            unsatisfy_lits.push(clause[j]);
        }
    }
    return false;
}
bool Solver::clauseSatisfy(Clause &clause) {
//    unsatisfy_lits.clear();
    for (int j = 0; j < clause.size(); j++) {
        lbool var_satisfy = this->value(clause[j]);
        if (var_satisfy == l_True) {
//            unsatisfy_lits.clear();
            return true;
        }
    }
    return false;
}

map<Var, lbool> Solver::getSuccessLit(){
    map<Var,lbool> successLit;
    for(int i = 0; i<this->nVars(); i++){
        successLit[i] = this->model[i];
    }
    return successLit;
}

void Solver::prepareQuantum(vector<vec<Lit> *> &all_unsatisfy_lits,  vector<vector<vec<Lit> *> *> &var2clauses){
    for (int i = 0; i < this->nVars(); i++) {
        var2clauses[i]->clear();
    }
    all_unsatisfy_lits.clear();

    vec<CRef> &clauses = this->clauses;
    Minisat::ClauseAllocator &ca = this->ca;

    vector<vec<Lit> *> sat2_clauses;

    int all_clause_num = clauses.size();// + learnts.size();
    Clause *all_clauses[all_clause_num];


    for (int i = 0; i < clauses.size(); i++) {
        all_clauses[i] = &ca[clauses[i]];
    }

    sort(all_clauses, all_clause_num, compareClauseBasedOnActivity);  // todo: very time-consuming

    for (int i = 0; i < all_clause_num; i++) {
        Clause &clause = *(all_clauses[i]);

        vec<Lit> &unsatisfy_lits = *genVecLit();
        bool is_satisfy = clauseSatisfy2(unsatisfy_lits, clause);

        if(is_satisfy){
            continue;
        }

        if(unsatisfy_lits.size()==0) continue;
        assert(unsatisfy_lits.size() != 0);

        if(unsatisfy_lits.size() > 5){
            continue;
        }
        all_unsatisfy_lits.push_back(&unsatisfy_lits);
        for (int j = 0; j < unsatisfy_lits.size(); j++) {
            Var variable = var(unsatisfy_lits[j]);
            var2clauses[variable]->push_back(&unsatisfy_lits);  //it is sorted based on the activity
        }
//        cout<<endl;
    }

}

void Solver::prepareQuantumv2(vector<Clause *> &all_unsatisfy_lits,  vector<vector<Clause *> *> &var2clauses){
//    writeClauseTofile();
    vec<CRef> &clauses = this->clauses;
    Minisat::ClauseAllocator &ca = this->ca;

//    vector<Clause *> sat2_clauses;

    int all_clause_num = clauses.size();// + learnts.size();
    Clause *all_clauses[all_clause_num];


    for (int i = 0; i < clauses.size(); i++) {
        all_clauses[i] = &ca[clauses[i]];
    }

    sort(all_clauses, all_clause_num, compareClauseBasedOnActivity);  // todo: very time-consuming

    for (int i = 0; i < all_clause_num; i++) {
        Clause &clause = *(all_clauses[i]);

        bool is_satisfy = clauseSatisfy(clause);

        if(is_satisfy){
            continue;
        }
//        cout<<clause.size();
        if(clause.size()==0) continue;
        assert(clause.size() != 0);

        if(clause.size() > 4){
            continue;
        }
        all_unsatisfy_lits.push_back(&clause);
        for (int j = 0; j < clause.size(); j++) {
            Var variable = var(clause[j]);
//            cout<<variable<<' ';
            var2clauses[variable]->push_back(&clause);  //it is sorted based on the activity
        }
//        cout<<endl;
    }


}

void Solver::writeClauseTofile(){
    char const* fileName = this->fileName;
    int num = this->solveNum;
    FILE *fp;
    char const* outfilepre = "../testClausesSize/";
    string newFileName = string(fileName);
    newFileName.erase(newFileName.length()-4,4);
    string const&origin_path = string(outfilepre) + newFileName + string(".txt");
    char const* path = origin_path.c_str();
    if((fp=fopen(path,"a"))==NULL){
        puts("Open file failure");
        exit(0);
    }
    vec<CRef> &clauses = this->clauses;
    Minisat::ClauseAllocator &ca = this->ca;

    int all_clause_num = clauses.size();// + learnts.size();
    Clause *all_clauses[all_clause_num];

    for (int i = 0; i < clauses.size(); i++) {
        all_clauses[i] = &ca[clauses[i]];
        Clause &clause = *(all_clauses[i]);
        for(int j = 0; j<clause.size(); j++){
            Lit p = clause[j];
            if(value(p) != l_Undef) continue;
            Var v = var(p);
            if(sign(p)){
                v = -v;
            }
            fprintf(fp,"%d ", v);
        }
        fprintf(fp,"\n");
    }

    fclose(fp);
}

/*_________________________________________________________________________________________________
|
|  search : (nof_conflicts : int) (params : const SearchParams&)  ->  [lbool]
|
|  Description:
|    Search for a model the specified number of conflicts.
|    NOTE! Use negative value for 'nof_conflicts' indicate infinity.
|
|  Output:
|    'l_True' if a partial assigment that is consistent with respect to the clauseset is found. If
|    all variables are decision variables, this means that the clause set is satisfiable. 'l_False'
|    if the clause set is unsatisfiable. 'l_Undef' if the bound on number of conflicts is reached.
|________________________________________________________________________________________________@*/
lbool Solver::search(int nof_conflicts) {
//    my_test();
    assert(ok);
    int backtrack_level = 0;
    int conflictC = 0;
    vec<Lit> learnt_clause;
    map<int,int> visit;
    starts++;


    for (;;) {
        int64_t start_time = getSysTimeMicros();
        bool has_conflict = false;
        CRef confl;
        confl = propagate();
//        cout<<order_heap.size()<<endl;
        if (PRINT_PROCESS && verbosity > 0)
            printf("propagate\n");

        propagations_timecost += getSysTimeMicros() - start_time;

        if(this->propagate_error){
            clause2vecLit(ca[confl], current_conflict_clause_instance);
            current_conflict_clause = &current_conflict_clause_instance;  // siwei
            conflicts++;
            conflictC++;
            learnt_clause.clear();
            return l_False;
        }

        learnt_clause.clear();

        if (confl != CRef_Undef) {  // conflict
            int64_t start_time = getSysTimeMicros();
            if(this->minisat_conflict_time == -1 && this->quantum_conflict_time != -1){
                this->minisat_conflict_time = start_time;
                this->delta_conflict_time = (double)(start_time-this->quantum_conflict_time)/1000;
            }
            if (PRINT_PROCESS && verbosity > 0)
                printf("conflict with %s\n", clause2string(ca[confl]).c_str());
            clause2vecLit(ca[confl], current_conflict_clause_instance);
            current_conflict_clause = &current_conflict_clause_instance;  // siwei
//            string conflict_str = clause2string(ca[confl]);

            // CONFLICT
            conflicts++;
            conflictC++;  // conflictC: current conflict of this restart
            if (decisionLevel() == 0){
                if (PRINT_PROCESS && verbosity > 0)
                    printf("backtrack at 0 level, can not be solved\n");
                return l_False;
            }

            analyze(confl, learnt_clause,
                    backtrack_level);  // learnt_clause e just temp to store one-time clause analysis, learnts are global

            conflict_analysis_timecost += getSysTimeMicros() - start_time;
//            && (decisions < 100 || my_rand(0.6))
        } else if (use_quantum && force_decision_lits.empty() && force_pick_vars.empty()  &&decisionLevel() < 5000) {  //&& decisionLevel() < 20  // && my_rand(0.3)
            this->iteration += 1;

            // int MAX_ITER = 200000;

            // if(this->iteration >= MAX_ITER){
            //     printf("iteration more than 200000, time out\n");
            //     return l_False;
            // }

            int64_t start_time = getSysTimeMicros();

            if (PRINT_PROCESS && verbosity > 0)
                printf("use quantum\n");

//            quantum_count++;

            prepareQuantum(all_unsatisfy_lits,var2clauses);

            int unassigned_var_num = 0;
            for(Var v = 0; v < var2clauses.size(); v++){
                if(value(v)== l_Undef){
                    unassigned_var_num++;
                }
            }

            
            bool do_one_time_solve = true;
            bool set_phase = true;
            int grid = 16;
            int base = (int)pow(double(grid/16),2);
            int MAX_VAR_NUM = 128*base;
            int MAX_CLAUSE_NUM = 250*base;

            int times = 0;
            if(decisions < 100){
                set_phase = true;
                times = 20;
            }else if(decisions < 5000){
//                if(unassigned_var_num>=128)
//                    MAX_VAR_NUM = unassigned_var_num/2;
//                else
//                    MAX_VAR_NUM = 128;
                times = 20;
            }else{
                times = 0;
            }

//`            if(my_rand(0.3)){
//                set_phase = true;
//            }`

            for(auto item: all_unsatisfy_lits){
                item->setmark(0);
            }
            bool sub_satisfy = false;
            if(times!=0){
                sub_satisfy = solveSATQuantum(*this, learnt_clause, backtrack_level, 2,
                                                   all_unsatisfy_lits, var2clauses, do_one_time_solve,set_phase, true, MAX_VAR_NUM, MAX_CLAUSE_NUM);
            }


            // can somehow speed up with noise-free backtrack
            // for uf, time should not be large, because it will affect the phase setting

            MAX_CLAUSE_NUM = 10000;
            for (int i = 0; i < times; i++) {
                if(!sub_satisfy || !force_decision_lits.empty() || !force_pick_vars.empty())
                    break;
                for(auto item: all_unsatisfy_lits){
                    item->setmark(0);
                }
                sub_satisfy = solveSATQuantum(*this, learnt_clause, backtrack_level, 0,
                                all_unsatisfy_lits, var2clauses, do_one_time_solve, set_phase, true, MAX_VAR_NUM, MAX_CLAUSE_NUM);
            }

            if(sub_satisfy && times!=0){
                quantum_success_number++;
            }else{
                quantum_conflict_number++;
            }
            quantum_total_number++;

            for (int i = 0; i < this->nVars();i++) {
//                delete var2clauses[i];
                var2clauses[i]->clear();
            }
            all_unsatisfy_lits.clear();

            freeAllVecLit();

            int64_t time_cost = getSysTimeMicros() - start_time;
            quantum_time_cost += double(time_cost) / 1000000;

            if(decisionLevel()==0 && sub_satisfy == false){
                confl = ca.alloc(learnt_clause, true);
                clause2vecLit(ca[confl], current_conflict_clause_instance);
                current_conflict_clause = &current_conflict_clause_instance;
                conflicts++;
                conflictC++;
                return l_False;
            }

        }

        if (learnt_clause.size() != 0) {  //
            while (!force_decision_lits.empty()) force_decision_lits.pop();
            while(!force_pick_vars.empty()) force_pick_vars.pop();
            for (int v = 0; v < nVars(); v++)
                user_pol[v] = l_Undef;

            if (PRINT_PROCESS && verbosity > 0)
                printf("backtrack from %d to %d, with learnt %s\n", decisionLevel(), backtrack_level, clause2string(learnt_clause).c_str());

            cancelUntil(backtrack_level);

//            string learnt_str = clause2string(learnt_clause);
            if (learnt_clause.size() == 1) {  // unity clause, which can be applied to propagate
                if(PRINT_PROCESS && verbosity > 0)
                    printf("Variable %d must be %d\n", var(learnt_clause[0]), sign(learnt_clause[0]) == false);
                if(!sign(learnt_clause[0])){
                    visit[var(learnt_clause[0])+1] = 1;
                }
                else if(sign(learnt_clause[0])){
                    visit[-(var(learnt_clause[0])+1)] = 1;
                }

                uncheckedEnqueue(learnt_clause[0]);

            } else {
                CRef cr = ca.alloc(learnt_clause, true);
                learnts.push(cr);
                attachClause(cr);

                claBumpActivity(ca[cr]);

                uncheckedEnqueue(learnt_clause[0], cr);
            }

            varDecayActivity();
            claDecayActivity();

            if (--learntsize_adjust_cnt == 0) {
                learntsize_adjust_confl *= learntsize_adjust_inc;
                learntsize_adjust_cnt = (int) learntsize_adjust_confl;
                max_learnts *= learntsize_inc;

                my_time_cost = double(getSysTimeMicros() - my_start_time) / 1000000;
                if (verbosity >= 1){
                    double now_timecost = (my_time_cost-quantum_time_cost) * 1000; //+ 2 * (double)quantum_count / 100000 - DLIS_timecost;
                    printf("| %9d | %7d %8d %8d | %8d %8d %6.0f | %6.3f %% | %6.3f  |\n",
                           (int) conflicts,
                           (int) dec_vars - (trail_lim.size() == 0 ? trail.size() : trail_lim[0]), nClauses(),
                           (int) clauses_literals,
                           (int) max_learnts, nLearnts(), (double) learnts_literals / nLearnts(),
                           progressEstimate() * 100,
                           now_timecost
                    );
//                    printStats();

                    if(now_timecost > 3000000){
                        printf("exceed the original timecost");
                        return l_False;
                    }
                }
            }

        } else {
//            // NO CONFLICT
            if ((nof_conflicts >= 0 && conflictC >= nof_conflicts) || !withinBudget()) {
                // Reached bound on number of conflicts:
                progress_estimate = progressEstimate();
                cancelUntil(0);
                return l_Undef;
            }

            // Simplify the set of problem clauses:
            if (decisionLevel() == 0 && !simplify())
                return l_False;

            if (learnts.size() - nAssigns() >= max_learnts)
                // Reduce the set of learnt clauses:
                reduceDB();

            Lit next = lit_Undef;
            while (decisionLevel() < assumptions.size()) {  // Current set of assumptions provided to solve by the user.
                // Perform user provided assumption:
                Lit p = assumptions[decisionLevel()];  //?? feel strange, the key of assumptionsis not Lit?
                if (value(p) == l_True) {
                    // Dummy decision level:
                    newDecisionLevel();
                } else if (value(p) == l_False) {
                    analyzeFinal(~p, conflict);
                    return l_False;
                } else {
                    next = p;
                    break;
                }
            }

            if (next == lit_Undef) {
                while (next == lit_Undef) {
                    if (force_decision_lits.empty()) {
                        int64_t start_time = getSysTimeMicros();
                        next = pickBranchLit();

                        if (PRINT_PROCESS && verbosity > 0)
                            printf("pick %s\n", lit2string(next).c_str());

                        decision_timecost += getSysTimeMicros() - start_time;
                        if (next == lit_Undef)
                            // Model found:
                            return l_True;
                        decisions++;
                    } else {
                        assert(use_quantum);
                        next = force_decision_lits.front();
                        force_decision_lits.pop();
                        if (PRINT_PROCESS && verbosity > 0)
                            printf("pick pop %s\n", lit2string(next).c_str());
                        assert(next != lit_Undef);
                        if (value(next) != l_Undef) {
                            next = lit_Undef;
                        }
                    }
                }

                if (next == lit_Undef){
                    // Model found:
                    return l_True;
                }

            }

            // Increase decision level and enqueue 'next'
            newDecisionLevel();
            uncheckedEnqueue(next);

//        TODO: try again
//        become slower, do not know why
//        while(!force_decision_lits.empty()){
//            Lit next_lit = force_decision_lits.front();
//            force_decision_lits.pop();
//            if(value(next_lit) == l_Undef)
//                uncheckedEnqueue(next_lit);
//        }
        }
    }
}


double Solver::progressEstimate() const {
    double progress = 0;
    double F = 1.0 / nVars();

    for (int i = 0; i <= decisionLevel(); i++) {
        int beg = i == 0 ? 0 : trail_lim[i - 1];
        int end = i == decisionLevel() ? trail.size() : trail_lim[i];
        progress += pow(F, i) * (end - beg);
    }

    return progress / nVars();
}

/*
  Finite subsequences of the Luby-sequence:

  0: 1
  1: 1 1 2
  2: 1 1 2 1 1 2 4
  3: 1 1 2 1 1 2 4 1 1 2 1 1 2 4 8
  ...


 */

static double luby(double y, int x) {
    // Find the finite subsequence that contains index 'x', and the
    // size of that subsequence:
    int size, seq;
    for (size = 1, seq = 0; size < x + 1; seq++, size = 2 * size + 1);

    while (size - 1 != x) {
        size = (size - 1) >> 1;
        seq--;
        x = x % size;
    }

    return pow(y, seq);
}

// NOTE: assumptions passed in member-variable 'assumptions'.
lbool Solver::solve_() {


    model.clear();
    conflict.clear();
    if (!ok)
        return l_False;

    solves++;

    max_learnts = nClauses() * learntsize_factor;
    if (max_learnts < min_learnts_lim)
        max_learnts = min_learnts_lim;

    learntsize_adjust_confl = learntsize_adjust_start_confl;
    learntsize_adjust_cnt = (int) learntsize_adjust_confl;
    lbool status = l_Undef;

    if (verbosity >= 1) {
        printf("============================[ Search Statistics ]==============================\n");
        printf("| Conflicts |          ORIGINAL         |          LEARNT          | Progress |\n");
        printf("|           |    Vars  Clauses Literals |    Limit  Clauses Lit/Cl |          |\n");
        printf("===============================================================================\n");
    }

    my_start_time = getSysTimeMicros();
    // Search:


    var2clauses.resize(this->nVars());
    for (int i = 0; i < this->nVars(); i++) {
        vector<vec<Lit> *> *clause_set = new vector<vec<Lit> *>;
        var2clauses[i] = clause_set;
    }
    int curr_restarts = 0;  //restart technique
    while (status == l_Undef) {
        double rest_base = luby_restart ? luby(restart_inc, curr_restarts) : pow(restart_inc, curr_restarts);
        int64_t restart_start_time = getSysTimeMicros();
        status = search(rest_base * restart_first);
        if(PRINT_PROCESS  && verbosity > 0)
            cout << "restart " << double(getSysTimeMicros() - restart_start_time) / 1000000 << endl;
        if (!withinBudget()) break;
        curr_restarts++;
    }
    for (int i = 0; i < this->nVars(); i++) {
        delete var2clauses[i];
    }
    my_time_cost = double(getSysTimeMicros() - my_start_time) / 1000000;

    if (verbosity >= 1)
        printf("===============================================================================\n");


    if (status == l_True) {
        // Extend & copy model:
        model.growTo(nVars());  // If problem is satisfiable, this vector contains the model (if any).
        for (int i = 0; i < nVars(); i++)
            model[i] = value(i);

        for (int i = 0; i < clauses.size(); i++) {
            CRef &clause_ref = clauses[i];
            Clause &clause = ca[clause_ref];

            bool is_satisfy = false;
            for (int j = 0; j < clause.size(); j++) {
                Lit lit = clause[j];
                if (litSatisfy(lit, assigns[var(lit)])) {
                    is_satisfy = true;
                    break;
                }
            }
            assert(is_satisfy);
        }
    } else if (status == l_False && conflict.size() == 0)
        ok = false;

    cancelUntil(0);  //the assignment of level 0 remain, it is a problem


//    printf("final solve");
    if (current_conflict_clause && status != l_True) {
//        printf("Current conflict clause ");
//        printClause(*current_conflict_clause);
    }
    if (status != l_True) {
//        printf("Error but no conflict!\n");
//        assert(current_conflict_clause != NULL);
    }

    assert(status != l_Undef);
    return status;
}

//todo: y be the conflict of the propogate can help solve other variable


bool Solver::implies(const vec<Lit> &assumps, vec<Lit> &out) {
    trail_lim.push(trail.size());
    for (int i = 0; i < assumps.size(); i++) {
        Lit a = assumps[i];

        if (value(a) == l_False) {
            cancelUntil(0);
            return false;
        } else if (value(a) == l_Undef)
            uncheckedEnqueue(a);
    }

    unsigned trail_before = trail.size();
    bool ret = true;
    if (propagate() == CRef_Undef) {
        out.clear();
        for (int j = trail_before; j < trail.size(); j++)
            out.push(trail[j]);
    } else
        ret = false;

    cancelUntil(0);
    return ret;
}

//=================================================================================================
// Writing CNF to DIMACS:
//
// FIXME: this needs to be rewritten completely.

static Var mapVar(Var x, vec<Var> &map, Var &max) {
    if (map.size() <= x || map[x] == -1) {
        map.growTo(x + 1, -1);
        map[x] = max++;
    }
    return map[x];
}


void Solver::toDimacs(FILE *f, Clause &c, vec<Var> &map, Var &max) {
    if (satisfied(c)) return;

    for (int i = 0; i < c.size(); i++)
        if (value(c[i]) != l_False)
            fprintf(f, "%s%d ", sign(c[i]) ? "-" : "", mapVar(var(c[i]), map, max) + 1);
    fprintf(f, "0\n");
}


void Solver::toDimacs(const char *file, const vec<Lit> &assumps) {
    FILE *f = fopen(file, "wr");
    if (f == NULL)
        fprintf(stderr, "could not open file %s\n", file), exit(1);
    toDimacs(f, assumps);
    fclose(f);
}


void Solver::toDimacs(FILE *f, const vec<Lit> &assumps) {
    // Handle case when solver is in contradictory state:
    if (!ok) {
        fprintf(f, "p cnf 1 2\n1 0\n-1 0\n");
        return;
    }

    vec<Var> map;
    Var max = 0;

    // Cannot use removeClauses here because it is not safe
    // to deallocate them at this point. Could be improved.
    int cnt = 0;
    for (int i = 0; i < clauses.size(); i++)
        if (!satisfied(ca[clauses[i]]))
            cnt++;

    for (int i = 0; i < clauses.size(); i++)
        if (!satisfied(ca[clauses[i]])) {
            Clause &c = ca[clauses[i]];
            for (int j = 0; j < c.size(); j++)
                if (value(c[j]) != l_False)
                    mapVar(var(c[j]), map, max);
        }

    // Assumptions are added as unit clauses:
    cnt += assumps.size();

    fprintf(f, "p cnf %d %d\n", max, cnt);

    for (int i = 0; i < assumps.size(); i++) {
        assert(value(assumps[i]) != l_False);
        fprintf(f, "%s%d 0\n", sign(assumps[i]) ? "-" : "", mapVar(var(assumps[i]), map, max) + 1);
    }

    for (int i = 0; i < clauses.size(); i++)
        toDimacs(f, ca[clauses[i]], map, max);

    if (verbosity > 0)
        printf("Wrote DIMACS with %d variables and %d clauses.\n", max, cnt);
}

void Solver::printStatsToFile(char const* path) const {
    FILE *fp;
    if((fp=fopen(path,"a+"))==NULL){
        puts("Open file failure");
        exit(0);
    }
    double cpu_time = cpuTime();
//    double mem_used = memUsedPeak();
    int64_t finish_time = getSysTimeMicros();

    double this_problem_time = (my_time_cost - quantum_time_cost) * 1000; //+ 2 * quantum_count / 100000;  //ms

    if (true) { //this_problem_time > 3
        fprintf(fp,"restarts              : %" PRIu64"\n", starts);
        fprintf(fp,"conflicts             : %-12" PRIu64"   (%.0f /sec)\n", conflicts, conflicts / cpu_time);
        fprintf(fp,"conflict cost         : %g ms \n", double(conflict_analysis_timecost) / 1000);
        fprintf(fp,"decisions             : %-12" PRIu64"   (%4.2f %% random) (%.0f /sec)\n", decisions,
                (float) rnd_decisions * 100 / (float) decisions, decisions / cpu_time);
        fprintf(fp,"decisions cost        : %g ms \n", double(decision_timecost) / 1000);
        fprintf(fp,"propagations          : %-12" PRIu64"   (%.0f /sec)\n", propagations, propagations / cpu_time);
        fprintf(fp,"propagations cost     : %g ms \n", double(propagations_timecost) / 1000);
        fprintf(fp,"conflict literals     : %-12" PRIu64"   (%4.2f %% deleted)\n", tot_literals,
                (max_literals - tot_literals) * 100 / (double) max_literals);
        //    if (mem_used != 0) printf("Memory used           : %.2f MB\n", mem_used);
        fprintf(fp,"actual CPU time       : %g s\n", cpu_time);
        //printf("principle all CPU time    : %g s\n", cpu_time-quantum_time_cost);
        fprintf(fp,"this problem time     : %g ms\n", this_problem_time);
        fprintf(fp,"annealing time        : %g ms\n", double(2 * quantum_count) / 100000);
        fprintf(fp,"quantum count         : %d\n", quantum_count);
        fprintf(fp,"simulate time         : %g s\n", quantum_time_cost);
        fprintf(fp,"quantum success number   : %d\n", quantum_success_number);
        fprintf(fp,"quantum conflict number   : %d\n", quantum_conflict_number);
        fprintf(fp,"quantum one time solve number   : %d\n", quantum_onetime_solve_number);
//        fprintf(fp,"level2conflicts : \n");
//        for(auto item : level2conficts){
//            fprintf(fp,"level:%d ",item.first);
//            fprintf(fp,"conflict : %d\n",item.second);
//        }
    }
    fclose(fp);
}

void Solver::printStats() const {
    double cpu_time = cpuTime();
//    double mem_used = memUsedPeak();
    int64_t finish_time = getSysTimeMicros();

    int annealing_count = quantum_success_number + quantum_conflict_number + quantum_onetime_solve_number;

    double annealing_time = double(quantum_success_number + quantum_conflict_number + quantum_onetime_solve_number) * 0.02;

    double this_problem_time = (my_time_cost - quantum_time_cost) * 1000;// + 2 * quantum_count / 100000;  //ms
    double solving_time = this_problem_time + annealing_time;

    if (true) { //this_problem_time > 3
        printf("restarts              : %" PRIu64"\n", starts);
        printf("conflicts             : %-12" PRIu64"   (%.0f /sec)\n", conflicts, conflicts / cpu_time);
        printf("conflict cost         : %g ms \n", double(conflict_analysis_timecost) / 1000);
        printf("decisions             : %-12" PRIu64"   (%4.2f %% random) (%.0f /sec)\n", decisions,
               (float) rnd_decisions * 100 / (float) decisions, decisions / cpu_time);
        printf("decisions cost        : %g ms \n", double(decision_timecost) / 1000);
        printf("propagations          : %-12" PRIu64"   (%.0f /sec)\n", propagations, propagations / cpu_time);
        printf("propagations cost     : %g ms \n", double(propagations_timecost) / 1000);
        printf("conflict literals     : %-12" PRIu64"   (%4.2f %% deleted)\n", tot_literals,
               (max_literals - tot_literals) * 100 / (double) max_literals);
//    if (mem_used != 0) printf("Memory used           : %.2f MB\n", mem_used);
        printf("actual CPU time       : %g s\n", cpu_time);
//printf("principle all CPU time    : %g s\n", cpu_time-quantum_time_cost);
        printf("solving time     : %g ms\n", solving_time);
        printf("annealing time        : %g ms\n", annealing_time);
        // printf("quantum count         : %d\n", quantum_count);
        printf("simulation time         : %g s\n", quantum_time_cost);
        printf("quantum success number   : %d\n", quantum_success_number);
        printf("quantum conflict number   : %d\n", quantum_conflict_number);
        printf("quantum one time solve number   : %d\n", quantum_onetime_solve_number);
        // printf("delta conflict time              : %g ms\n", delta_conflict_time);
//        for(auto item : level2conficts){
//            printf("level:%d ",item.first);
//            printf("conflict : %d\n",item.second);
//        }
    }
}


//=================================================================================================
// Garbage Collection methods:

void Solver::relocAll(ClauseAllocator &to) {
    // All watchers:
    //
    watches.cleanAll();
    for (int v = 0; v < nVars(); v++)
        for (int s = 0; s < 2; s++) {
            Lit p = mkLit(v, s);
            vec<Watcher> &ws = watches[p];
            for (int j = 0; j < ws.size(); j++)
                ca.reloc(ws[j].cref, to);
        }

    // All reasons:
    //
    for (int i = 0; i < trail.size(); i++) {
        Var v = var(trail[i]);

        // Note: it is not safe to call 'locked()' on a relocated clause. This is why we keep
        // 'dangling' reasons here. It is safe and does not hurt.
        if (reason(v) != CRef_Undef && (ca[reason(v)].reloced() || locked(ca[reason(v)]))) {
//            assert(!isRemoved(reason(v)));
            ca.reloc(vardata[v].reason, to);
        }
    }

    // All learnt:
    //
    int i, j;
    for (i = j = 0; i < learnts.size(); i++)
        if (!isRemoved(learnts[i])) {
            ca.reloc(learnts[i], to);
            learnts[j++] = learnts[i];
        }
    learnts.shrink(i - j);

    // All original:
    //
    for (i = j = 0; i < clauses.size(); i++)
        if (!isRemoved(clauses[i])) {
            ca.reloc(clauses[i], to);
            clauses[j++] = clauses[i];
        }
    clauses.shrink(i - j);
}


void Solver::garbageCollect() {
    // Initialize the next region to a size corresponding to the estimated utilization degree. This
    // is not precise but should avoid some unnecessary reallocations for the new region:
    ClauseAllocator to(ca.size() - ca.wasted());

    relocAll(to);
    if (verbosity >= 2)
        printf("|  Garbage collection:   %12d bytes => %12d bytes             |\n",
               ca.size() * ClauseAllocator::Unit_Size, to.size() * ClauseAllocator::Unit_Size);
    to.moveTo(ca);
}
