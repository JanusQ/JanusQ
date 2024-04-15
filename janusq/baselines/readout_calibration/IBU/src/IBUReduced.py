import logging
from janusq.baselines.readout_calibration.IBU.src.IBUBase import IBUBase
from collections import namedtuple
from typing import NamedTuple, Union
from janusq.baselines.readout_calibration.IBU.utils.data_utils import *
from janusq.baselines.readout_calibration.IBU.src.kron_matmul import *
from functools import partial
from tqdm import tqdm


class IBUReduced(IBUBase):

    def __init__(self, mats_raw: List[np.ndarray], params: dict,
                 mem_constrained: bool = False):

        self._num_qubits = params['num_qubits']
        self._library = params['library']
        self._use_log = params['use_log']
        self.mem_constrained = mem_constrained

        self._verbose = params['verbose']

        self._mats = self.mats_to_kronstruct(mats_raw)
        self._obs = None
        self._init = None
        self._guess = None
        self.ReducedBitstrs = namedtuple('ReducedBitstrs',
                                         ['obs_bitstrs', 'obs_mat',
                                          'exp_bitstrs', 'exp_mat', 'obs_vec'])

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def library(self):
        return self._library

    @property
    def use_log(self):
        return self._use_log

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool):
        self._verbose = value

    @property
    def mats(self):
        return self._mats

    @property
    def obs(self):
        if self._obs is None:
            return self._obs
        return self._obs.obs_vec

    @property
    def init(self):
        if self._init is None:
            return self._init
        if self.library == 'tensorflow':
            t_init = tf.reshape(self._init, (-1, 1))
        elif self.library == 'jax':
            t_init = self._init.reshape(-1, 1).block_until_ready()
        else:
            t_init = self._init.reshape(-1, 1)
        return t_init

    @property
    def guess(self):
        if self._guess is None:
            return self._guess
        if self.library == 'tensorflow':
            t_guess = tf.reshape(self._guess, (-1, 1))
        elif self.library == 'jax':
            t_guess = self._guess.reshape(-1, 1).block_until_ready()
        else:
            t_guess = self._guess.reshape(-1, 1)
        return t_guess

    def guess_as_dict(self, tol: float = 1e-6):
        if self._guess is None:
            return self._guess
        return vec_to_dict(self.guess, tol, self._obs.exp_bitstrs)

    ############################################################################
    #                     DATA PROCESSING / GENERATION
    ############################################################################

    def set_obs(self, obs: Union[NamedTuple, dict], ham_dist: int = 1):
        """
            Sets the observed counts for each bitstring to be used during IBU
            Bitstrings should be in reverse order of the single-qubit error
            matrices.
        :param obs: can be a dict mapping bitstrings to counts, or a NamedTuple
        :param ham_dist: the hamming distance from the observed bitstrings up
                         to which need to be tracked
        called ReducedBitstrs (see self.process_obs_dict())
        """
        if type(obs) == dict:
            if self.verbose:
                logging.info("Processing dictionary of counts...")
            obs = self.process_obs_dict(obs, ham_dist)

        self._obs = obs

    def generate_obs(self, t_raw: jnp.ndarray, num_resamples: int = 1000,
                     ham_dist: int = 1):
        """
            Using the single-qubit error matrices, generate
        :param t_raw: a 2^N x 1 jax ndarray of the true distribution over
                      bitstrings from which to generate noisy measurements
        :param num_resamples: number of samples to draw from the noisy
                              distribution
        :param ham_dist: the max hamming distance from observed bitstrings of
                         the bitstrings that will be tracked
        :return: NamedTuple called ReducedBitstrs (see self.process_obs_dict())
        """

        if self.verbose:
            logging.info("Generating noisy distribution over counts...")
        obs_true = self.kron_matmul_full(self._mats, t_raw)
        obs_dict = resampler(num_resamples, obs_true, self.num_qubits,
                             self.use_log)
        obs = self.process_obs_dict(obs_dict, ham_dist)

        return obs

    def initialize_guess(self, init: Union[None, list, dict, tuple,
                                           jnp.ndarray] = None,
                         smoother: float = 0.0):
        """
            Initialize a guess to be iterated with IBU. There are four ways of
            initializing:
                - Uniform distribution over all K TRACKED bitstrings: no
                arguments need to be passed in
                - Uniform distribution over select bitstrings: pass in init as
                a list of select bitstrings; smoother should be 0, unless some
                 smoothing over the unselected bitstrings is desired.
                - Arbitrary non-uniform distribution over select bitstrings:
                pass in init as a dict of bitstring keys to counts/probabilities
                (NOT log probabilities). smoother may be set non-zero if needed.
                - Vector of probabilities: direct initialization by passing init
                as a vector of probabilities/log probabilities, as a K x 1 jax
                ndarray.
            Bitstrings should be in reverse order as the single-qubit error
            matrices passed in.
        :param init: a list, dict, or vec specifying how to initialize the guess
        for IBU.
        :param smoother: adds (typically) small probability mass to every
        bitstring to smooth the distribution given by init.
        """
        if init is None:
            if self.verbose:
                logging.info("Initializing guess with uniform distribution over the "
                      "expanded set of bitstrings...")
            t_init = unif_dense(len(self._obs.exp_bitstrs),
                                library=self.library, use_log=self.use_log)
        elif type(init) == list or type(init) == tuple:
            if self.verbose:
                logging.info(f"Initializing guess with uniform distribution over"
                      f" {len(init)} bitstrings...")
            obs_dict = {key: 1 for key in init}
            t_init = counts_to_vec_subspace(obs_dict, self._obs.exp_bitstrs,
                                            self.verbose)
            t_init = normalize_vec(t_init + smoother, self.library,
                                   self.use_log)
            t_init = (t_init + smoother) / (1 + smoother*(2**self.num_qubits))
        elif type(init) == dict:
            if self.verbose:
                logging.info(f"Initializing guess with empirical distribution from "
                      f"dictionary of counts...")
            t_init = counts_to_vec_subspace(init, self._obs.exp_bitstrs,
                                            self.verbose)
            t_init = normalize_vec(t_init, self.library, self.use_log)
            t_init = (t_init + smoother) / (1 + smoother*(t_init.shape[0]))
        else:
            if self.verbose:
                logging.info(f"Initializing guess with given vector...")
            t_init = init

        self._init = t_init
        if self.library == 'tensorflow':
            self._guess = tf.Variable(t_init, trainable=False)
        else:
            self._guess = t_init

    def reduce_to_top_guess(self, k: int):
        """
            Reduce the number of tracked bitstrings in self._obs.exp_bitstrs
            to just the k bitstrings with the highest probabilities (given in
            self._guess).
        """
        if k is None:
            k = len(self._obs.obs_bitstrs)
        inds = jnp.array(np.argpartition(np.array(self._guess).reshape(-1),
                                         -k)[-k:])
        self._guess = self._guess[inds]
        self._guess = self._guess / jnp.sum(self._guess)
        exp_bitstrs = [self._obs.exp_bitstrs[s] for s in inds]
        exp_mat = self._obs.exp_mat[inds, :]
        self._obs = self.ReducedBitstrs(self._obs.obs_bitstrs,
                                        self._obs.obs_mat, exp_bitstrs,
                                        exp_mat, self._obs.obs_vec)

    ############################################################################
    #                     MATMUL WITH KRONECKER STRUCTURE
    ############################################################################

    def kron_matmul_full(self, mat: jnp.ndarray,
                         vec: jnp.ndarray) -> jnp.ndarray:
        """
            Kronecker matrix multiplication dispatcher WITHOUT any subspace
            reduction; used to generate_obs(). See kron_matmul() in IBUFull.py
            for more details.
        """

        if self.library == 'jax':
            result = self._kron_matmul_jax_full(mat, vec)
        else:
            raise "Unsupported library!"

        return result

    def _kron_matmul_jax_full(self, mat: jnp.ndarray,
                              vec: jnp.ndarray) -> jnp.ndarray:
        """
            Kronecker matrix multiplication in jax WITHOUT any subspace
            reduction; used to generate_obs(). See kron_matmul_jax() in
            IBUFull.py for more details.
        """

        if self.use_log:
            max_vec = jnp.max(vec)
            exp_vec = jnp.exp(vec - max_vec)
        else:
            max_vec = None
            exp_vec = vec

        result = jnp.transpose(exp_vec)
        for i in jnp.arange(mat.shape[0] - 1, -1, -1):
            op = mat[i, :, :]

            dim = int(result.shape[-2] * result.shape[-1] // 2)
            result_shape = (dim, 2)
            result = jnp.reshape(result, result_shape)  # 2**n-1 x 2
            result = jnp.matmul(op, jnp.transpose(result))  # 2 x 2**n-1
        result = jnp.reshape(result, exp_vec.shape)

        if self.use_log:
            result = jnp.log(result) + max_vec

        return result

    ############################################################################
    #                       TRAIN AND ITERATION LOOPS
    ############################################################################

    def train(self, max_iters: int = 100,
              tol: float = 1e-4,
              soln: Union[None, list, dict] = None,
              hd_reduce: Tuple[int, int] = (None, None))\
            -> Tuple[jnp.ndarray, int, jnp.ndarray]:
        """
            Train IBU.
        :param max_iters: maximum number of iterations to run IBU for
        :param tol: tolerance for convergence; IBU halts when norm difference
                    of update difference is less than this amount
        :param soln: (optional) solution for validating learned model. This can
                     either be a list of bitstrs of the correct probability or a
                     dictionary mapping bitstrings to correct probability
                     (only jax currently supported).
        :param hd_reduce: (optional) a tuple of 2 ints that enable hybrid
                          Hamming distance approaches; the first int in the
                          tuple specifies after *which iteration* to narrow down
                          the list of tracked/supported bitstrings, and the
                          second int specifies *how many* bitstrings to narrow
                          to.
        :return: a 3-tuple:
                - the solution after iteration (as jax array)
                - # iterations (may be less than max_iters if converged)
                - an array tracking the performance of the guess wrt the soln
                  (if provided), either as probability assigned to "right"
                  answer or as norm error from correct solution
        """
        tracker = self.initialize_tracker(max_iters)

        iteration = 0
        diff = tol + 1
        if self.verbose:
            pbar = tqdm(total=max_iters, desc='IBU Iteration')
        else:
            pbar = None
        if hd_reduce[0] is not None and hd_reduce[0] == -1:
            self.reduce_to_top_guess(hd_reduce[1])

        # Main train loop
        while iteration < max_iters and diff > tol:
            tracker = self.log_performance(tracker, soln, iteration)
            diff = self.train_iter()

            if hd_reduce[0] is not None and iteration == hd_reduce[0]:
                self.reduce_to_top_guess(hd_reduce[1])
            iteration += 1
            if self.verbose:
                pbar.update()

        tracker = self.log_performance(tracker, soln, iteration)
        if self.library == 'jax':
            if self.verbose:
                logging.info("Waiting for JAX to return control flow...")
            tracker.block_until_ready()
        return self.guess, iteration, tracker[:iteration + 1, 0]

    def train_iter(self) -> jnp.float32:
        """
            Dispatcher for IBU iteration based on library (currently on jax is
            supported)
        :return: the norm difference between the updated parameters and previous
                 parameters
        """
        if self.library == 'jax':
            self._guess, diff = self._train_iter_jax()
            return diff
        else:
            raise "Unsupported library!"

    def _train_iter_jax(self) -> jnp.float32:
        """
            Dispatcher that selects the JITted function to call based on
            whether there are memory constraints and log-space is being used
            for numerical precision
        :return: the norm difference between the updated parameters and previous
                 parameters
        """
        if self.use_log:
            if self.mem_constrained:
                return self._train_iter_jax_log_compact(self._mats, self._guess,
                                                        self._obs.exp_mat,
                                                        self._obs.obs_mat,
                                                        self._obs.obs_vec)
            else:
                try:
                    return self._train_iter_jax_log_fast(self._mats,
                                                         self._guess,
                                                         self._obs.exp_mat,
                                                         self._obs.obs_mat,
                                                         self._obs.obs_vec)
                except:
                    logging.warning("May have run into memory issues, switching to memory" 
                          " efficient implementation...")
                    self.mem_constrained = True
                    return self._train_iter_jax_log_compact(self._mats,
                                                            self._guess,
                                                            self._obs.exp_mat,
                                                            self._obs.obs_mat,
                                                            self._obs.obs_vec)
        else:
            if self.mem_constrained:
                return self._train_iter_jax_compact(self._mats, self._guess,
                                                    self._obs.exp_mat,
                                                    self._obs.obs_mat,
                                                    self._obs.obs_vec)
            else:
                try:
                    return self._train_iter_jax_fast(self._mats, self._guess,
                                                     self._obs.exp_mat,
                                                     self._obs.obs_mat,
                                                     self._obs.obs_vec)
                except:
                    logging.warning("May have run into memory issues, switching to memory" 
                          " efficient implementation...")
                    self.mem_constrained = True
                    return self._train_iter_jax_compact(self._mats, self._guess,
                                                        self._obs.exp_mat,
                                                        self._obs.obs_mat,
                                                        self._obs.obs_vec)

    @partial(jit, static_argnums=0)
    def _train_iter_jax_fast(self, mats: jnp.ndarray, guess: jnp.ndarray,
                             exp_mat: jnp.ndarray, obs_mat: jnp.ndarray,
                             obs_vec: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.float32]:
        """
            A single (jax) iteration of IBU, optimized for speed and assuming
        no memory constraints.
        :param mats: N x 2 x 2 jax.ndarray storing single-qubit error matrices
        :param guess: K' x 1 vector representing current guess of distribution
                      over tracked bitstrings
        :param exp_mat: a K x N matrix where each row encodes a tracked
                        bitstring as an array
        :param obs_mat: a K x N matrix where each row encodes an observed
                        bitstring as an array
        :param obs_vec: a K x 1 vector representing normalized probabilities of
                        each observed bitstring
        :return: the new guess and norm difference between the previous guess
                 and new guess
        """
        obs_guess = fast_kron_matmul(mats, guess, exp_mat, obs_mat)

        # Update estimate of P(t)
        eq1 = obs_vec / obs_guess
        eq1 = jnp.nan_to_num(eq1)
        eq2 = fast_kron_matmul(jnp.transpose(mats, (0, 2, 1)), eq1,
                               obs_mat, exp_mat)
        diff = jnp.linalg.norm((guess * eq2) - guess, ord=1)
        guess = guess * eq2

        return guess, diff

    @partial(jit, static_argnums=0)
    def _train_iter_jax_compact(self, mats: jnp.ndarray, guess: jnp.ndarray,
                                exp_mat: jnp.ndarray, obs_mat: jnp.ndarray,
                                obs_vec: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.float32]:
        """
            A single (jax) iteration of IBU that is more memory efficient but
            often slower than _train_iter_jax_fast(...). See that function for
            documentation.
        """
        obs_guess = compact_kron_matmul(mats, guess, exp_mat, obs_mat)

        # Update estimate of P(t)
        eq1 = obs_vec / obs_guess
        eq1 = jnp.nan_to_num(eq1)
        eq2 = compact_kron_matmul(jnp.transpose(mats, (0, 2, 1)), eq1,
                                  obs_mat, exp_mat)
        diff = jnp.linalg.norm((guess * eq2) - guess, ord=1)
        guess = guess * eq2

        return guess, diff

    @partial(jit, static_argnums=0)
    def _train_iter_jax_log_fast(self, mats: jnp.ndarray, guess: jnp.ndarray,
                                 exp_mat: jnp.ndarray, obs_mat: jnp.ndarray,
                                 obs_vec: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.float32]:
        """
            A single (jax) iteration of IBU, optimized for speed and assuming
            no memory constraints, when the computation happens in log space.
            See _train_iter_jax_log_fast(...) for full documentation.
        """
        # Compute kron matmul in log space
        max_guess = jnp.max(guess)
        exp_guess = jnp.exp(guess - max_guess)
        result = fast_kron_matmul(mats, exp_guess, exp_mat, obs_mat)
        obs_guess = jnp.log(result) + max_guess

        # Update estimate of P(t)
        eq1 = obs_vec - obs_guess
        eq1 = jnp.nan_to_num(eq1)

        # Again, kron matmul in log space
        max_eq1 = jnp.max(eq1)
        exp_eq1 = jnp.exp(eq1 - max_eq1)
        eq2 = fast_kron_matmul(jnp.transpose(mats, (0, 2, 1)), exp_eq1, obs_mat,
                               exp_mat)
        eq2 = jnp.log(eq2) + max_eq1

        diff = jnp.linalg.norm(jnp.exp(guess + eq2) - jnp.exp(guess), ord=1)
        guess = guess + eq2

        return guess, diff

    @partial(jit, static_argnums=0)
    def _train_iter_jax_log_compact(self, mats: jnp.ndarray, guess: jnp.ndarray,
                                    exp_mat: jnp.ndarray, obs_mat: jnp.ndarray,
                                    obs_vec: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.float32]:
        """
            A single (jax) iteration of IBU that is more memory efficient but
            often slower than _train_iter_jax_fast(...), when the computation
            happens in log space. See that function for documentation.
        """
        # Compute kron matmul in log space
        max_guess = jnp.max(guess)
        exp_guess = jnp.exp(guess - max_guess)
        result = compact_kron_matmul(mats, exp_guess, exp_mat, obs_mat)
        obs_guess = jnp.log(result) + max_guess

        # Update estimate of P(t)
        eq1 = obs_vec - obs_guess
        eq1 = jnp.nan_to_num(eq1)

        # Again, kron matmul in log space
        max_eq1 = jnp.max(eq1)
        exp_eq1 = jnp.exp(eq1 - max_eq1)
        eq2 = compact_kron_matmul(jnp.transpose(mats, (0, 2, 1)), exp_eq1,
                                  obs_mat, exp_mat)
        eq2 = jnp.log(eq2) + max_eq1

        diff = jnp.linalg.norm(jnp.exp(guess + eq2) - jnp.exp(guess), ord=1)
        guess = guess + eq2

        return guess, diff

    ############################################################################
    #                                LOGGING
    ############################################################################

    def initialize_tracker(self, max_iters: int) -> jnp.ndarray:
        """
            Initialize jax ndarray of length max_iters to track progress after
            each iteration of IBU.
        :param max_iters: maximum number of iterations that may be tracked
        :return: a jax ndarray of zeros
        """
        if self.library == 'jax':
            return jnp.zeros([max_iters, 1])
        else:
            raise "Unsupported library!"

    def log_performance(self, tracker: jnp.ndarray, soln: Union[list, dict],
                        idx: int):
        """
            Logs the performance of the current self._guess.
            If soln is a list of bitstrs, tracker tracks the total probability
            assigned to these bitstrs. If soln is a dictionary mapping
            bitstrings to correct probability/log probabilities, tracker tracks
            the l1-norm error with the current guess (in the original space, not
            the log space).

        :param tracker: the array in which to log performance
        :param soln: the solution (either list of correct keys or true prob vec
                     or dictionary mapping bitstrings to correct probability)
        :param idx: the index at which to log performance in tracker
        :return: the updated tracker as jax ndarray/tensorflow Tensor.
        """
        if soln is not None:
            if type(soln) == list or type(soln) == tuple:
                res = self.get_prob(soln)
            else:
                res = self.get_l1_error(soln)
            if self.library == 'jax':
                tracker = tracker.at[idx].set(res)
            else:
                raise "Unsupported Library!"

        return tracker

    def get_prob(self, soln: list) -> jnp.float32:
        """
            Given a list of bitstrings, returns the total probability assigned
            by the current guess to those bitstrings
        :param soln: A list of bitstrings for which to get total probability for
        :return: a jax DeviceArray of a float representing probability
        """
        if self.library == 'jax':
            prob = jnp.zeros([1, 1])
            for sol in soln:
                if self.use_log:
                    if sol in self._obs.exp_bitstrs:
                        prob += jnp.exp(
                            self._guess[self._obs.exp_bitstrs.index(sol)])
                else:
                    if sol in self._obs.exp_bitstrs:
                        prob += self._guess[self._obs.exp_bitstrs.index(sol)]
            return prob[0, 0]
        else:
            raise "Unrecognized library!"

    def get_l1_error(self, soln: dict) -> jnp.float32:
        """
            Returns the norm error between the current guess and the provided
            solution. If the current guess and provided solution are log
            probabilities, the log probabilities are first exponentiated before
            taking the l1-norm difference.
        :param soln: a dictionary mapping bitstrings to their true probabilities
                     or log probabilities
        :return: float, l1-norm error between guess and provided soln
        """
        if self.library == 'jax':
            bitstrs_intsect = set(soln.keys()) & set(self._obs.exp_bitstrs)
            err = jnp.zeros([1, 1])

            guess_copy = (jnp.copy(self._guess),
                          jnp.exp(self._guess))[self.use_log]
            for bitstr in soln.keys():
                soln_prob = (-soln[bitstr],
                             -jnp.exp(soln[bitstr]))[self.use_log]
                if bitstr in bitstrs_intsect:
                    guess_copy = guess_copy.at[
                        self._obs.exp_bitstrs.index(bitstr)].add(soln_prob)
                else:
                    err += jnp.absolute(soln_prob)
            err += jnp.sum(jnp.absolute(guess_copy))

        else:
            raise "Unrecognized library!"

        return err[0, 0]

    def get_linf_error(self, soln: dict) -> jnp.float32:
        """
            Returns the norm error between the current guess and the provided
            solution. If the current guess and provided solution are log
            probabilities, the log probabilities are first exponentiated before
            taking the l_inf-norm difference.
        :param soln: a dictionary mapping bitstrings to their true probabilities
                     or log probabilities
        :return: float, l_inf-norm error between guess and provided soln
        """
        if self.library == 'jax':
            bitstrs_intsect = set(soln.keys()) & set(self._obs.exp_bitstrs)
            err = 0.0

            guess_copy = (jnp.copy(self._guess),
                          jnp.exp(self._guess))[self.use_log]
            for bitstr in soln.keys():
                soln_prob = (-soln[bitstr],
                             -jnp.exp(soln[bitstr]))[self.use_log]
                if bitstr in bitstrs_intsect:
                    guess_copy = guess_copy.at[
                        self._obs.exp_bitstrs.index(bitstr)].add(soln_prob)
                else:
                    err = max(jnp.absolute(soln_prob), err)
            err = max(jnp.max(jnp.absolute(guess_copy)), err)

        else:
            raise "Unrecognized library!"

        return float(err)

    ############################################################################
    #                                 UTILS
    ############################################################################

    def mats_to_kronstruct(self, mats_raw: List[np.ndarray]) -> jnp.ndarray:
        """
            Helper function to convert list of numpy matrices of single-qubit
            error probabilities to jax ndarray.
        :param mats_raw: list of 2x2 numpy arrays of single-qubit error
         probabilities, in reverse order their respective qubits appear in
         bitstrings.
        :return: jax ndarray
        """
        if self.library == 'jax':
            kronmats = jnp.array(mats_raw)
        else:
            raise "Unsupported library!"

        return kronmats

    def process_obs_dict(self, obs_dict: dict, ham_dist: int) -> NamedTuple:
        """
            A helper function to convert a given dictionary of observations
            into the namedTuple ReducedBitstrs that stores the following
            relevant information:
            obs_bitstrs: the dictionary of observed bitstrings and their counts
            obs_mat: a K' x N matrix where each row encodes an observed
                     bitstring as a vector of 0s and 1s
            exp_bitstrs: obs_bitstrs + all bitstrings that are no more than
                              hamming distance ham_dist from any bitstring in
                              obs_bitstrs
            exp_mat: a K x N matrix where each row encodes a bitstring from
                     exp_bitstrs as a vector of 0s and 1s
            obs_vec: a K x 1 vector of counts corresponding to each bitstring
                     in obs_bitstrs, in the same order as the keys
        :param obs_dict: a dictionary of measured bitstrings mapped to counts
        :param ham_dist: the hamming distance from the observed bitstrings that
                         tracked bitstrings will fall under
        :return: a namedtuple called ReducedBitstrs consisting of obs_bitstrs,
                 obs_mat, exp_bitstrs, exp_mat, obs_vec
        """
        obs_bitstrs = sorted(obs_dict.keys())
        desc_ham, desc_obs, desc_exp = None, None, None
        if self.verbose:
            desc_ham = f"Computing strings within Hamming radius {ham_dist}"
            desc_obs = "Encoding observed bitstrings as matrix"
            desc_exp = "Encoding expanded set of bitstrings as matrix"
        exp_bitstrs = expand_strs_by_hamdist(obs_bitstrs, ham_dist,
                                             desc=desc_ham)
        exp_bitstrs = sorted(set(obs_bitstrs + exp_bitstrs))
        obs_vec = counts_to_vec_subspace(obs_dict, obs_bitstrs, self.verbose)
        obs_vec = normalize_vec(obs_vec, self.library, self.use_log)
        obs_mat = strs_to_mat(obs_bitstrs, self.library, desc=desc_obs)
        exp_mat = strs_to_mat(exp_bitstrs, self.library, desc=desc_exp)
        obs = self.ReducedBitstrs(obs_bitstrs, obs_mat, exp_bitstrs,
                                  exp_mat, obs_vec)

        return obs
