from collections import defaultdict
from functools import lru_cache
import random
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from janusq.tools.ray_func import map
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.approximation.maxcut as maxcut
from sklearn.naive_bayes import GaussianNB
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from janusq.optimizations.readout_mitigation.fem.tools import all_bitstrings, npformat_to_statuscnt, statuscnt_to_npformat, to_int


def downsample_statuscnt(statscnt: tuple[np.ndarray, np.ndarray], qubits: list) -> tuple[np.ndarray, np.ndarray]:
    measured_np, count_np = statscnt
    new_measured_np = measured_np[:, qubits]
    return [new_measured_np, count_np]


def construct_cpd(args: tuple[int, list[int]], bench_results: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]):
    qubit, related_qubits = args

    data = np.zeros(shape=(3, 3**len(related_qubits)))

    for real_bstr, statuscnt in zip(*bench_results):
        real_bstr = [int(real_bstr[qubit]) for qubit in related_qubits]

        col_index = to_int(real_bstr, base=3)

        for m_bstr, count in zip(*statuscnt):
            read_value = int(m_bstr[qubit])
            data[read_value][col_index] += count

    def extend(bench_results, related_qubits, qubit):
        X = []
        Y = []
        for real_bstr, statuscnt in bench_results.items():
            set_values = [int(real_bstr[qubit])
                          for qubit in related_qubits]
            for measure_bitstring, count in statuscnt.items():
                read_value = int(measure_bitstring[qubit])
                if read_value == 2:
                    continue
                if count < 1:
                    count = int(count*5000)
                for _ in range(count):
                    X.append(set_values)
                    Y.append(read_value)
        return X, Y

    infer_model = None
    for col_index in range(3**len(related_qubits)):
        if np.sum(data[:, col_index]) == 0:
            op_types = []
            temp_colum_index = col_index
            for _ in related_qubits:
                op_types.append(temp_colum_index % 3)
                temp_colum_index = temp_colum_index // 3
            op_types.reverse()

            if op_types[related_qubits.index(qubit)] == 2:
                data[2, col_index] = 1
            else:
                if infer_model is None:
                    X, Y = extend(bench_results, related_qubits, qubit)
                    infer_model = GaussianNB()
                    infer_model.fit(X, Y)
                probs = infer_model.predict_proba([op_types])[0]
                data[0:2, col_index] = probs
        else:
            data[:, col_index] /= np.sum(data[:, col_index])

    # generate a cpd table with following format
    # +-----------+----------+----------+-----+----------+----------+----------+
    # | 0_set     | 0_set(0) | 0_set(0) | ... | 0_set(2) | 0_set(2) | 0_set(2) |
    # +-----------+----------+----------+-----+----------+----------+----------+
    # | 1_set     | 1_set(0) | 1_set(1) | ... | 1_set(0) | 1_set(1) | 1_set(2) |
    # +-----------+----------+----------+-----+----------+----------+----------+
    # | 0_read(0) | 0.0      | 0.0      | ... | 942.0    | 41.0     | 0.0      |
    # +-----------+----------+----------+-----+----------+----------+----------+
    # | 0_read(1) | 0.0      | 0.0      | ... | 58.0     | 959.0    | 0.0      |
    # +-----------+----------+----------+-----+----------+----------+----------+
    # | 0_read(2) | 1.0      | 1.0      | ... | 0.0      | 0.0      | 0.0      |
    # +-----------+----------+----------+-----+----------+----------+----------+

    # P(qubit_read | qubit_set, other qubit_set)
    qubit_cpd = TabularCPD(f"{qubit}_read", 3,
                           data,
                           evidence=[
                               f"{related_qubit}_set" for related_qubit in related_qubits],
                           evidence_card=[3, ] * len(related_qubits),
                           )

    return qubit_cpd


def construct_bayesian_network(bench_results, n_qubits, groups, multi_process=True):
    qubit_to_group = {}
    for group in groups:
        for qubit in group:
            qubit_to_group[qubit] = group

    cpds = []
    network_edges = []

    for qubit in range(n_qubits):
        cpds.append(TabularCPD(f"{qubit}_set", 3, [[1/3]] * 3,))

        for related_qubit in qubit_to_group[qubit]:
            network_edges.append((f'{related_qubit}_set', f'{qubit}_read'))

    cpds += map(construct_cpd, [(qubit, qubit_to_group[qubit]) for qubit in range(
        n_qubits)], bench_results=bench_results, multi_process=multi_process)

    model = BayesianNetwork(network_edges)  
    model.add_cpds(*cpds)
    infer = VariableElimination(model)

    return model, infer


def correlation_based_partation(bench_results, group_size, n_qubits, draw_grouping = False):
    error_count = np.zeros(shape=(n_qubits, 3, n_qubits, 1))
    all_count = np.zeros(shape=(n_qubits, 3, n_qubits, 1))

    for real, statuscnt in tqdm(zip(*bench_results)):
        for bstr, count in zip(*statuscnt):
            error_qubits = np.argwhere(real != bstr)

            for error_qubit in error_qubits:
                for qubit in range(n_qubits):
                    error_count[qubit][real[qubit]][error_qubit] += count

            for qubit1 in range(n_qubits):
                for qubit2 in range(n_qubits):
                    all_count[qubit1][real[qubit1]][qubit2] += count

    error_freq = error_count / all_count

    freq_diff = np.abs(error_freq[:, 0, :]-error_freq[:, 1, :]) + np.abs(
        error_freq[:, 0, :]-error_freq[:, 2, :]) + np.abs(error_freq[:, 1, :]-error_freq[:, 2, :])
    large_corr_qubit_pairs = np.where(freq_diff > 0.01)  # (q1, q2, 0)

    graph = nx.Graph()

    graph.add_nodes_from([[qubit, {'qubit': qubit}]
                         for qubit in range(n_qubits)])
    for q1, q2, _ in zip(*large_corr_qubit_pairs):
        if q1 == q2:
            continue
        graph.add_edge(q1, q2, freq_diff=np.round(freq_diff[q1][q2][0], 4))
    
    if n_qubits < 10 and draw_grouping:
        plt.clf()
        plt.figure(figsize=(7, 3))
        plt.subplot(1, 2, 1)
        nx.draw(graph, with_labels=True, font_weight='bold')
        labels = nx.get_edge_attributes(graph, 'freq_diff')
        nx.draw_networkx_edge_labels(graph, pos=nx.spring_layout(graph), edge_labels=labels)
        plt.show()

    def partition(group):
        small_partitions = []
        for sub_group in maxcut.one_exchange(graph.subgraph(group), weight='freq_diff')[1]:
            if len(sub_group) == len(group):
                return [
                    [qubit]
                    for qubit in sub_group
                ]
            if len(sub_group) <= group_size:
                small_partitions.append(sub_group)
            else:
                small_partitions += partition(sub_group)
        return small_partitions

    groups = [
        list(group)
        for group in partition(list(range(n_qubits)))
    ]

    groups = partition(list(graph.nodes()))

# Create a new graph
    partitioned_graph = nx.Graph()

    # Add nodes
    for group in groups:
        for node in group:
            partitioned_graph.add_node(node)

    # Add edges
    for edge in graph.edges():
        for group in groups:
            if edge[0] in group and edge[1] in group:
                partitioned_graph.add_edge(edge[0], edge[1])

    if n_qubits < 10:
        plt.subplot(1, 2, 2)
    nx.draw(partitioned_graph, with_labels=True, font_weight='bold')
    # plt.show()

    return groups


def hamming_distance(str1: np.ndarray, str2: np.ndarray):
    return np.sum(np.abs(str1 - str2))

def to_bitstring(integer, n_qubits):
    measure_bitstring = bin(integer).replace('0b', '')
    measure_bitstring = (n_qubits - len(measure_bitstring)) * '0' + measure_bitstring
    return measure_bitstring

def permute(statscnt: tuple[np.ndarray, np.ndarray], qubit_order: list):
    if isinstance(statscnt, dict):
        permuted_stat_counts = {}
        for bstr, count in statscnt.items():
            new_bitstring = ['0'] * len(bstr)
            for now_pos, old_pos in enumerate(qubit_order):
                new_bitstring[now_pos] = bstr[old_pos]
            permuted_stat_counts[''.join(new_bitstring)] = count
        return permuted_stat_counts
    else:
        measured_np, count_np = statscnt
        permuted_measured_np = np.array([
            measure_bitstring[qubit_order]
            for measure_bitstring in measured_np
        ], dtype=np.int8)
        return np.array(permuted_measured_np), np.array(count_np)


def kron_basis(arr1, arr2, offest):
    grid = np.meshgrid(arr2, arr1)
    return grid[1].ravel() << offest | grid[0].ravel()


class TPEngine():
    '''
        tensor-product engine
    '''
    def __init__(self, n_qubits, group_to_M):
        self.n_qubits = n_qubits
        self.group_to_M = group_to_M

        self.group_to_invM = {
            group: np.linalg.inv(M)
            for group, M in group_to_M.items()
        }

        self.groups = []
        self.qubit_map = []  # Internal order -> External order
        for group in group_to_M:
            self.qubit_map += list(group)  
            self.groups.append(group)

        self.invqubit_map = [0] * self.n_qubits
        for real_pos, old_pos in enumerate(self.qubit_map):
            self.invqubit_map[old_pos] = real_pos

    def run(self, statscnts: dict, threshold: float = None, group_to_invM=None):
        '''Assuming there are no overlapping qubits between groups'''
        statscnts = permute(statscnts, self.qubit_map)

        if group_to_invM is None:
            group_to_invM = self.group_to_invM

        groups = self.groups

        if threshold is None:
            sum_count = sum(statscnts.values())
            threshold = sum_count * 0  # 1e-152

        # TODO: jax.jit
        rm_prob = defaultdict(float)
        for basis, count in zip(*statscnts):
            now_basis = None
            now_values = None

            pointer = 0
            for group in groups:
                invM = group_to_invM[group]
                group_size = len(group)

                group_basis = basis[pointer: pointer + group_size]

                pointer += group_size

                group_mitigated_vec = invM[:, to_int(group_basis)]
                group_basis = np.arange(2**group_size)

                if now_basis is None:
                    next_basis = group_basis
                    next_values = group_mitigated_vec * count
                else:
                    next_basis = kron_basis(now_basis, group_basis, group_size)
                    next_values = np.kron(now_values, group_mitigated_vec)

                filter = np.logical_or(
                    next_values > threshold, next_values < -threshold)
                now_basis = next_basis[filter]
                now_values = next_values[filter]

                # now_basis = next_basis
                # now_values = next_values

            for basis, value in zip(now_basis, now_values):
                rm_prob[basis] += value  # the basis is in the order of the groups

        rm_prob = {
            basis: value
            for basis, value in rm_prob.items()
            if value > 0
        }
        sum_prob = sum(rm_prob.values())
        rm_prob = {
            basis: value / sum_prob
            for basis, value in rm_prob.items()
        }

        rm_prob = {
            to_bitstring(bstr, self.n_qubits): prob
            for bstr, prob in rm_prob.items()
        }
        rm_prob = statuscnt_to_npformat(rm_prob)
        rm_prob = permute(rm_prob, self.invqubit_map)

        return rm_prob


class Iteration():
    def __init__(self, n_qubits, threshold = 1e-3):   #threshold=1e-10
        self.n_qubits = n_qubits
        self.threshold = threshold

    def init(self, bench_results, groups: list[list[int]], multi_process: bool = False):
        self.bayesian_network_model, self.bayesian_infer_model = construct_bayesian_network(
            bench_results, self.n_qubits, groups, multi_process=multi_process)
        self.groups = [
            sorted(group)
            for group in groups
        ]

    @lru_cache
    def unmeasureed_index(self, group_size):
        return np.array([index for index, bstr in enumerate(all_bitstrings(group_size, base=3)) if 2 not in bstr])

    @lru_cache
    def get_engine(self, measured_qubits: list) -> TPEngine:
        n_measured_qubits = len(measured_qubits)

        bayesian_infer_model: VariableElimination = self.bayesian_infer_model

        group_to_M = {}
        for group in self.groups:
            group_measured_qubits = [
                qubit for qubit in group if qubit in measured_qubits]
            n_group_measured_qubits = len(group_measured_qubits)
            if n_group_measured_qubits == 0:
                continue
            M = np.zeros(shape=(2**n_group_measured_qubits,
                         2**n_group_measured_qubits))
            for bstr in all_bitstrings(n_group_measured_qubits):
                posterior_p = bayesian_infer_model.query([f'{qubit}_read' for qubit in group_measured_qubits],   # if qubit in measured_qubits
                                                         evidence={
                    f'{qubit}_set': bstr[group_measured_qubits.index(qubit)] if qubit in measured_qubits else 2
                    for qubit in group
                }
                )
                posterior_v: np.ndarray = posterior_p.values.reshape(
                    3**n_group_measured_qubits)  # It has become data in the columns of matrix M
                posterior_v = posterior_v[self.unmeasureed_index(
                    n_group_measured_qubits)]  

                assert abs(sum(posterior_v) -
                           1) < 1e-2, sum(posterior_v)  

                M[:, to_int(bstr, base=2)] = posterior_v

            remap_group = tuple([measured_qubits.index(qubit)
                                for qubit in group_measured_qubits])
            group_to_M[remap_group] = M

        return TPEngine(n_measured_qubits, group_to_M)

    def mitigate(self, statscnt: tuple[np.ndarray, np.ndarray], measured_qubits: list = None, cho = None):
        threshold = self.threshold

        if measured_qubits is not None:
            measured_qubits = tuple(measured_qubits)

        if isinstance(statscnt, dict):
            statscnt = statuscnt_to_npformat(statscnt)

        engine = self.get_engine(measured_qubits)
        statscnt = downsample_statuscnt(
            statscnt, measured_qubits)  # Trim the unmeasured qubits
    

        mitigated_statscnts = engine.run(statscnt, threshold=threshold)



        extend_statscnts = np.zeros(
            (len(mitigated_statscnts[0]), self.n_qubits), dtype=np.int8) * 2
        for index, bstr in enumerate(mitigated_statscnts[0]):
            extend_statscnts[index, measured_qubits] = bstr
            
        if cho is not None:
            # return npformat_to_statuscnt((extend_statscnts, mitigated_statscnts[0]))
            return mitigated_statscnts
        else:
            return extend_statscnts, mitigated_statscnts[1]


class Mitigator():
    def __init__(self, n_qubits, n_iters=2, threshold=8e-4):
        self.n_qubits = n_qubits
        self.n_iters = n_iters
        self.threshold = threshold

        self.iters: list[Iteration] = None

    def random_group(self, group_size):
        qubits = list(range(self.n_qubits))

        groups = []
        while len(qubits) != 0:
            now_group = []
            for _ in range(group_size):
                now_group.append(random.choice(qubits))
                qubits = [
                    qubit
                    for qubit in qubits
                    if qubit != now_group[len(now_group)-1]
                ]
                if len(qubits) == 0:
                    break
            now_group.sort()
            groups.append(now_group)

        return groups

    def eval_statuscnt(self, bench_results: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]):
        reals, statuscnt = bench_results

        total_dist = 0
        n_total = 0

        for real_bstr, (m_bstrs, counts) in zip(reals, statuscnt):
            n_total += sum(counts)
            for m_bstr, count in zip(m_bstrs, counts):
                total_dist += hamming_distance(m_bstr, real_bstr) * count

        return total_dist/n_total

    def eval_partation(self, groups: list[list[int]], bench_results: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]], multi_process=False) -> tuple[iter, float, float]:
        iter = Iteration(self.n_qubits, threshold=self.threshold)
        iter.init(bench_results, groups, multi_process=multi_process)

        reals, statuscnts = bench_results

        def mitigate(elm):
            real, probdist = elm
            measured_qubits = [int(elm) for elm in np.argwhere(real != 2)]
            mitig_statuscnt = iter.mitigate(probdist, measured_qubits)
            return mitig_statuscnt

        opt_statuscnts = map(mitigate, list(
            zip(reals, statuscnts)), multi_process=multi_process)

        opt_score = self.eval_statuscnt((reals, opt_statuscnts))

        return iter, opt_score, (reals, opt_statuscnts)

    def init(self, bench_results, group_size=2, partation_methods=['random', 'max-cut'], multi_process=True, draw_grouping = False):
        
        real_bstrs, statuscnts = bench_results
        if isinstance(statuscnts[0], dict):
            statuscnts: list[tuple[np.ndarray, np.ndarray]] = map(statuscnt_to_npformat, statuscnts)
            bench_results = (real_bstrs, statuscnts)
        
        n_qubits = self.n_qubits
        n_iters = self.n_iters

        self.iters = []
        self.scores = []

        for iter_index in range(n_iters):
            candidate_groups = []
            if 'random' in partation_methods:
                for _ in range(1):
                    groups = self.random_group(group_size)
                    candidate_groups.append(groups)

            if 'max-cut' in partation_methods:
                groups = correlation_based_partation(
                    bench_results, group_size, n_qubits, draw_grouping = draw_grouping)
                candidate_groups.append(groups)


            candidate_iter_score_results = [
                self.eval_partation(group, bench_results, multi_process)
                for group in candidate_groups
            ]

            candidate_iter_score_results.sort(key=lambda elm: elm[1])

            # select the best candidate
            self.iters.append(candidate_iter_score_results[0][0])
            self.scores.append(candidate_iter_score_results[0][1])
            bench_results = candidate_iter_score_results[0][2]

        best_plm_index = np.argmin(self.scores)
        self.iters = self.iters[:best_plm_index+1]
        self.scores = self.scores[:best_plm_index+1]

        return self.scores[-1]
    
    def mitigate(self, statscnt: dict, measured_qubits = None, cho = None):
        if isinstance(statscnt, dict):
            statscnt = statuscnt_to_npformat(statscnt)

        if measured_qubits is None:
            assert len(statscnt[0][0]) == self.n_qubits
            measured_qubits = list(range(self.n_qubits))

        

        opt_statscnt = statscnt
        for iter in self.iters:
            opt_statscnt = iter.mitigate(
                opt_statscnt, measured_qubits = measured_qubits,cho = cho)

            if iter != self.iters[-1]:
                opt_statscnt = opt_statscnt[0], opt_statscnt[1] * 1000

        if isinstance(statscnt, dict):
            opt_statscnt = npformat_to_statuscnt(opt_statscnt)

        return opt_statscnt
