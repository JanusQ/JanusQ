import numpy as np
import tensorflow as tf
import jax.numpy as jnp
from typing import List, Tuple
from tqdm import trange
from datetime import datetime
from collections import defaultdict


def get_log_dir(params):
    """
        Generates the path to the directory in which to store experimental
        results.
    """
    if params['exp_name'] == 'raw':
        logdir = f"results/raw"
    elif params['exp_name'] == "m3":
        logdir = f"results/m3"
    else:
        if params["init"] == 'unif':
            init_name = 'unif_init'
        elif params["init"] == 'obs':
            init_name = f"obs{params['smoothing']}_init"
        else:
            raise "Unknown init type!"
        if params["method"] == 'reduced':
            suffix = f"_HD{params['ham_dist']}"
        else:
            suffix = ''

        iteration = f"{params['max_iters']}maxiters_{params['tol']}tol"

        logdir = f"results/{params['exp_name']}/{params['num_qubits']}_qubits/" \
                 f"{init_name}/{params['method']}{suffix}/{params['library']}_" \
                 f"{int(params['use_log'])}log_{iteration}_" + \
                 datetime.now().strftime("%Y%m%d-%H%M%S")

    return logdir


################################################################################
#                    DATA PREPROCESSING (reduced subspace)
################################################################################


def resampler(num_samples, probs, num_qubits, use_log=False):
    """
        Draws samples from a vector and returns a counts dictionary
    :param num_samples: number of samples to draw
    :param probs: 2^N by 1 probability vector
    :param num_qubits: number of qubits N
    :param use_log: whether probability in probs is in log space
    :return: a dictionary of bitstring keys and count values
    """
    if use_log:
        samples = list(np.random.choice(np.arange(2 ** num_qubits),
                                        size=(num_samples,),
                                        p=np.exp(np.asarray(
                                            probs.reshape(-1)))))
    else:
        samples = list(np.random.choice(np.arange(2 ** num_qubits),
                                        size=(num_samples,),
                                        p=np.asarray(probs.reshape(-1))))

    obs_dict = defaultdict(int)
    for s in range(len(samples)):
        obs_dict[format(samples[s], f'0{num_qubits}b')[::-1]] += 1

    return obs_dict


def gen_ham_strings(string, d, prefix='', i=0):
    """
        Generate all strings that are Hamming distance d away from a given str
    :param string: the string from which to compute all other strings
    :param d: the desired Hamming distance
    :param prefix: recursive parameters
    :param i: recursive parameter
    :return: list of strings hamming distance d from string
    """
    # Base Case
    if d == 0:
        return [prefix + string[i:]]

    words = []

    if '0' != string[i]:
        words += gen_ham_strings(string, d - 1, prefix + '0', i + 1)
    if '1' != string[i]:
        words += gen_ham_strings(string, d - 1, prefix + '1', i + 1)

    if len(string) - i > d:
        words += gen_ham_strings(string, d, prefix + string[i], i + 1)

    return words


def expand_strs_by_hamdist(strs_list, ham_dist, desc):
    """
        Given a list of strings strs_list, return a list of all strings that
        are hamming distance ham_dist away from at least one string in strs_list
    :param strs_list: list of strings to compute hamming distance form
    :param ham_dist: hamming distance from string
    :param desc: (optional) description
    :return: list of strings that are hamming distance ham_dist from every
             string in strs_list
    """
    expanded_bitstrs = []

    for s in trange(len(strs_list), desc=desc, disable=desc is None):
        for d in range(1, ham_dist + 1):
            expanded_bitstrs += gen_ham_strings(strs_list[s], d)
    expanded_bitstrs = list(set(expanded_bitstrs))
    return expanded_bitstrs


################################################################################
#                             DATA PREPROCESSING
################################################################################

def ghz_dense(dimension, library='numpy', use_log=False):
    if library == 'tensorflow':
        ghz = tf.transpose((tf.one_hot([0], dimension, dtype=tf.double) +
                            tf.one_hot([dimension - 1], dimension,
                                       dtype=tf.double)) / 2)
        if use_log:
            ghz = tf.math.log(ghz)
    elif library == 'jax':
        ghz = jnp.zeros((dimension, 1))
        ghz = ghz.at[0, 0].set(0.5)
        ghz = ghz.at[-1, 0].set(0.5)
        if use_log:
            ghz = jnp.log(ghz)
    else:
        ghz = np.zeros((dimension, 1))
        ghz[0, 0] = 0.5
        ghz[-1, 0] = 0.5
        if use_log:
            ghz = np.log(ghz)

    return ghz


def arbitrary_qubit_op(library='numpy'):
    if library == 'tensorflow':
        op_true = tf.constant([[0.9, 0.2], [0.1, 0.8]], dtype='float64')
    elif library == 'jax':
        op_true = jnp.array([[0.9, 0.2], [0.1, 0.8]])
    else:
        op_true = np.array([[0.9, 0.2], [0.1, 0.8]], dtype='float64')

    return op_true


def unif_dense(dimension, library='numpy', use_log=False):
    if library == 'tensorflow':
        unif = tf.ones((dimension, 1), dtype=tf.double) / dimension
        if use_log:
            unif = tf.math.log(unif)
    elif library == 'jax':
        unif = jnp.ones((dimension, 1)) / dimension
        if use_log:
            unif = jnp.log(unif)
    else:
        unif = np.ones((dimension, 1)) / dimension
        if use_log:
            unif = np.log(unif)

    return unif


################################################################################
#                             DICT MANIPULATION
################################################################################

def counts_to_vec_full(data_dict):
    """
        Converts a dictionary of bitstrings mapped to counts to a 2^N x 1 numpy
        vector of counts; the vector indexes bitstrings in the REVERSE order.
        For e.g. vec[0] corresponds to 0000..., vec[1] to 1000..., vec[2] to
        0100..., vec[4] to 1100..., and so on.
    """
    dimension = 2 ** len(list(data_dict.keys())[0])
    vec = np.zeros([dimension, 1])
    for key, val in data_dict.items():
        vec[int(key[::-1], 2)] = val
    return vec


def counts_to_vec_subspace(obs_dict, bitstrs_list, verbose):
    """
        Converts a dictionary of bitstrings mapped to counts to a K x 1 numpy
        vector of counts, where K is the length of bitstrs_list.
        The vector indexes bitstrings in the REVERSE order. For e.g. vec[0]
        corresponds to 0000..., vec[1] to 1000..., vec[2] to 0100..., vec[4] to
        1100..., and so on.
        NOTE: Only keys of obs_dict that appear in bitstrs_list will have an
        entry in the returned vector. So if there are bitstrings in obs_dict
        but NOT in bitstrs_list, these will not be tracked in the output.
    """
    vec = np.zeros([len(bitstrs_list), 1])
    obs_bitstrs = obs_dict.keys()

    for s in trange(len(bitstrs_list), desc="Counts to vector",
                    disable=not verbose):
        if bitstrs_list[s] in obs_bitstrs:
            vec[s] = obs_dict[bitstrs_list[s]]

    return vec


def normalize_vec(vec, library, use_log):
    """
        Normalizes a numpy vector and converts it to the desired library
        (np, tf, jax). use_log determines whether to return the log of the
        normalized vector.
    """
    if library == 'tensorflow':
        if use_log:
            norm_vec = tf.convert_to_tensor(np.log(vec) - np.log(np.sum(vec)),
                                            dtype=tf.double)
        else:
            norm_vec = tf.convert_to_tensor(vec / np.sum(vec), dtype=tf.double)
    elif library == 'jax':
        if use_log:
            norm_vec = jnp.array(np.log(vec) - np.log(np.sum(vec)))
        else:
            norm_vec = jnp.array(vec / np.sum(vec))
    elif library == 'numpy':
        if use_log:
            norm_vec = np.array(np.log(vec) - np.log(np.sum(vec)))
        else:
            norm_vec = np.array(vec / np.sum(vec))
    else:
        raise "Unknown library!"

    return norm_vec


def strs_to_mat(strs_list, library, desc=None):
    """
        Converts a K-list of N-length bitstrings into a K x N matrix of
        0s and 1s in the desired library (only jax currently supported).
    """
    mat = np.zeros([len(strs_list), len(strs_list[0])])
    for i in trange(len(strs_list), desc=desc, disable=desc is None):
        mat[i, :] = [int(x) for x in list(strs_list[i])[::-1]]

    if library == 'jax':
        mat = jnp.array(mat, dtype=int)
    else:
        raise "Unsupported library!"

    return mat


def vec_to_dict(vec, tol=None, bitstrs_indexed=None):
    """
        Converts a vector of probabilities/counts to a dictionary; essentially
        reverses counts_to_vec_full() and counts_to_vec_subspace().
    :param vec: the vector to convert to a dictionary
    :param tol: any entry in the vector with value less than tol is ignored and
                not included to the dictionary.
    :param bitstrs_indexed: when using IBUReduced, the bitstrings corresponding
                            to vec are not immediately obvious; bitstrs_indexed
                            specifies the bitstring corresponding to each entry
                            in vec; must be same length as vec.
    :param verbose: whether to print progress bar
    :return: dictionary of bitstrings mapped to counts
    """
    dim = vec.shape[0]
    # Figure out which elements of vector to select
    if tol is None:
        inds = range(dim)
        values = vec
    else:
        x = np.ma.masked_less_equal(vec, tol)
        if type(x.mask) == np.bool_:
            # when no values are less than mask, must collect all values
            inds = range(dim)
            values = vec
        else:
            inds = np.where(~x.mask)[0]
            values = vec[inds]

    # Figure out the corresponding bitstrings
    if bitstrs_indexed is None:
        keys = [format(i, f'0{int(np.log2(dim))}b')[::-1] for i in inds]
    else:
        assert len(bitstrs_indexed) == dim
        keys = [bitstrs_indexed[i] for i in inds]
    data = dict(zip(keys, values))

    return data


def resample_from_dict(orig_dict: dict, num_samples: int = None):
    """
        Given a dictionary of bitstrings and their counts, resample
        a new dictionary from the empirical distribution; useful for
        bootstrapping.
    :param orig_dict: a dictionary mapping bitstrings to counts
    :return: a dictionary mapping bitstrings to counts
    """
    bitstrs, counts = list(zip(*orig_dict.items()))
    total_counts = np.sum(counts)
    probs = counts / total_counts
    if num_samples is None:
        sampled = np.random.choice(bitstrs, total_counts, p=probs)
    else:
        sampled = np.random.choice(bitstrs, num_samples, p=probs)
    sampled_dict = defaultdict(int)
    for s in sampled:
        sampled_dict[s] += 1
    return sampled_dict


def get_l1_error_between_dicts(dict_a: dict, dict_b: dict,
                               log_probs: bool = False) -> float:
    dict_a_d = defaultdict(int, dict_a)
    dict_b_d = defaultdict(int, dict_b)
    err = 0
    bitstrs_union = set(dict_a.keys()) | set(dict_b.keys())
    for bitstr in bitstrs_union:
        if log_probs:
            err += float(np.absolute(np.exp(dict_a_d[bitstr]) -
                               np.exp(dict_b_d[bitstr])))
        else:
            err += float(np.absolute(dict_a_d[bitstr] - dict_b_d[bitstr]))

    return err


def get_linf_error_between_dicts(dict_a: dict, dict_b: dict,
                               log_probs: bool = False) -> float:
    dict_a_d = defaultdict(int, dict_a)
    dict_b_d = defaultdict(int, dict_b)
    err = 0
    bitstrs_union = set(dict_a.keys()) | set(dict_b.keys())
    for bitstr in bitstrs_union:
        if log_probs:
            err = max(np.absolute(np.exp(dict_a_d[bitstr]) -
                                  np.exp(dict_b_d[bitstr])), err)
        else:
            err = max(np.absolute(dict_a_d[bitstr] - dict_b_d[bitstr]), err)

    return float(err)


def marginalize_out(counts_dict: dict,
                    idx_list: List[int]) -> Tuple[list, defaultdict]:
    """
        Given a dictionary of bitstring keys and count values, marginalize out
        the bits in idx_list
    :param counts_dict: a dictionary of bitstrings mapped to counts/probs
    :param idx_list: a list of indices in the bitstring to marginalize out
    :return: a tuple of (kept indices, reduced counts dictionary)
    """
    new_dict = defaultdict(int)
    kept_idxs = []
    for bitstr, count in counts_dict.items():
        new_bitstr = "".join([bitstr[i]
                              for i in range(len(bitstr)) if i not in idx_list])
        new_dict[new_bitstr] += count
        kept_idxs = [x for x in range(len(bitstr)) if x not in idx_list]
    return kept_idxs, new_dict
