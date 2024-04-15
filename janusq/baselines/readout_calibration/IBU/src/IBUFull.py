import logging
from janusq.baselines.readout_calibration.IBU.utils.data_utils import *
from janusq.baselines.readout_calibration.IBU.src.IBUBase import IBUBase
from typing import Union, List, Tuple
from functools import partial
from jax import jit
from tqdm import tqdm


class IBUFull(IBUBase):

    def __init__(self, mats_raw: List[np.ndarray], params: dict):

        self._num_qubits = params['num_qubits']
        self._library = params['library']
        self._use_log = params['use_log']

        self._verbose = params['verbose']

        self._mats = self.mats_to_kronstruct(mats_raw, transpose=False)
        self._matsT = self.mats_to_kronstruct(mats_raw, transpose=True)
        self._obs = None
        self._init = None
        self._guess = None

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
        return self._obs

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
        return vec_to_dict(self.guess, tol)

    ############################################################################
    #                     DATA PROCESSING / GENERATION
    ############################################################################

    def set_obs(self, obs: Union[dict, np.array, jnp.array, tf.Tensor]):
        """
            Sets the observed counts for each bitstring to be used during IBU
            Bitstrings should be in reverse order of the single-qubit error
             matrices.
        :param obs: can be a dict mapping bitstrings to counts, a numpy array,
        a jax array, or a tensorflow Tensor. If computations are done in log
        space, and obs is a vector (not dict), it should be a vector of log
        probabilities.
        """
        if type(obs) == dict:
            if self.verbose:
                logging.info("Converting dictionary of counts to vector...")
            obs_vec = counts_to_vec_full(obs)
            obs_vec = normalize_vec(obs_vec, self.library, self.use_log)
        else:
            obs_vec = obs

        if self.verbose:
            logging.info("Setting counts distribution...")
        self._obs = obs_vec

    def generate_obs(self, t_raw: Union[np.ndarray, jnp.ndarray, tf.Tensor]) \
            -> Union[np.ndarray, jnp.ndarray, tf.Tensor]:
        """
            Generates the distribution over bitstrings observed for a given
            ground truth probability distribution t_raw, with the noise model
            given by the single-qubit errors at instantiation.
            Bitstrings will be in reverse order as the single-qubit error
            matrices passed in, but this ordering will be reversed in the vector
            representation, which counts up from right-to-left.
        :param t_raw: the "true" probability vector over bitstrings, which will
        be noised as per self.mats, as numpy/jax ndarray or tensorflow Tensor.
        :return: a 2**num_qubits x 1 vector of probabilities (or
        log-probabilities) over bitstrings of the same dtype as input.
        """

        if self.verbose:
            logging.info("Generating noisy distribution over counts...")
        obs_true = self.kron_matmul(self.mats, t_raw)

        return obs_true

    def initialize_guess(self, init: Union[None, list, dict, tuple, np.ndarray,
                                           jnp.ndarray, tf.Tensor] = None,
                         smoother: float = 0.0):
        """
            Initialize a guess to be iterated with IBU. There are four ways of
            initializing:
                - Uniform distribution over all bitstrings: no arguments need to
                be passed in
                - Uniform distribution over select bitstrings: pass in init as
                a list of select bitstrings; smoother should be 0, unless some
                 smoothing over the unselected bitstrings is desired.
                - Arbitrary non-uniform distribution over select bitstrings:
                pass in init as a dict of bitstring keys to counts/probabilities
                (NOT log probabilities). smoother may be set non-zero if needed.
                - Vector of probabilities: direct initialization by passing init
                as a vector of probabilities/log probabilities, as a numpy/jax
                ndarray, or tensorflow Tensor (MUST match library choice for
                IBU.)
            Bitstrings should be in reverse order as the single-qubit error
            matrices passed in.
        :param init: a list, dict, or vec specifying how to initialize the guess
        for IBU.
        :param smoother: adds (typically) small probability mass to every
        bitstring to smooth the distribution given by init.
        """
        if init is None:
            if self.verbose:
                logging.info("Initializing guess with uniform distribution over all "
                      "bitstrings...")
            t_init = unif_dense(2 ** self.num_qubits, library=self.library,
                                use_log=self.use_log)

        elif type(init) == list or type(init) == tuple:
            if self.verbose:
                logging.info(f"Initializing guess with uniform distribution over"
                      f" {len(init)} bitstrings...")
            obs_dict = {key: 1 for key in sorted(init)}
            t_init = counts_to_vec_full(obs_dict)
            t_init = normalize_vec(t_init, self.library, self.use_log)
            t_init = (t_init + smoother) / (1 + smoother*(2**self.num_qubits))

        elif type(init) == dict:
            if self.verbose:
                logging.info(f"Initializing guess with empirical distribution from "
                      f"dictionary of counts...")
            t_init = counts_to_vec_full(init)
            t_init = normalize_vec(t_init, self.library, self.use_log)
            t_init = (t_init + smoother) / (1 + smoother*(2**self.num_qubits))

        else:
            if self.verbose:
                logging.info(f"Initializing guess with given vector...")
            t_init = (init + smoother) / (1 + smoother*(2**self.num_qubits))


        self._init = t_init

        if self.library == 'tensorflow':
            self._guess = tf.Variable(t_init, trainable=False)
        else:
            self._guess = t_init

    ############################################################################
    #                     MATMUL WITH KRONECKER STRUCTURE
    ############################################################################

    def kron_matmul(self, mat: Union[np.ndarray, jnp.ndarray,
                                     tf.linalg.LinearOperatorKronecker],
                    vec: Union[np.ndarray, jnp.ndarray, tf.Tensor]) \
            -> Union[np.ndarray, jnp.ndarray, tf.Tensor]:
        """
            Dispatcher for fast matrix multiplication with a vector when the
            matrix is the kronecker product of N sub-matrices of identical
            dimension. jax/tensorflow supported, numpy not supported currently.

        :param mat: a [N, 2, 2]-array (jax/tensorflow) of N-qubit ops
        :param vec: a [2**N, 1]-array
        :return: the product mat @ vec
        """

        if self.library == 'tensorflow':
            result = self._kron_matmul_tf(mat, vec)
        elif self.library == 'jax':
            result = self._kron_matmul_jax(mat, vec)
        elif self.library == 'numpy':
            result = self._kron_matmul_numpy(mat, vec)
        else:
            raise "Unsupported library!"

        return result

    @tf.function
    def _kron_matmul_tf(self, mat: tf.Tensor,
                        vec: tf.linalg.LinearOperatorKronecker) -> tf.Tensor:
        """
            Fast matrix multiplication (tensorflow) with a vector when the
            matrix is the kronecker product of N sub-matrices of identical
            dimension.

        :param mat: a [N, 2, 2]-tf Tensor of N-qubit ops
        :param vec: a [2**N, 1]-tf Tensor
        :return: the product mat @ vec
        """
        if self.use_log:
            max_vec = tf.reduce_max(vec)
            exp_vec = tf.math.exp(vec - max_vec)
            result = mat.matmul(exp_vec)
            result = tf.math.log(result) + max_vec
        else:
            result = mat.matmul(vec)

        return result

    @partial(jit, static_argnums=(0,))
    def _kron_matmul_jax(self, mat: jnp.ndarray, vec: jnp.ndarray) \
            -> jnp.ndarray:
        """
            Fast matrix multiplication (jax) with a vector when the matrix is
            the kronecker product of N sub-matrices of identical dimension.

        :param mat: a [N, 2, 2]-jax ndarray of N-qubit ops
        :param vec: a [2**N, 1]-jax ndarray
        :return: the product mat @ vec
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

    def _kron_matmul_numpy(self, mat: np.ndarray, vec: np.ndarray):
        raise NotImplementedError

    ############################################################################
    #                               TRAIN
    ############################################################################

    def train(self, max_iters: int = 100, tol: float = 1e-4,
              soln: Union[dict, List[str], jnp.ndarray, tf.Tensor] = None)\
            -> Tuple[Union[jnp.ndarray, tf.Tensor], int,
                     Union[jnp.ndarray, tf.Tensor]]:
        """
            Train IBU.
        :param max_iters: maximum number of iterations to run IBU for
        :param tol: tolerance for convergence; IBU halts when norm difference
                    of update difference is less than this amount
        :param soln: solution for validating learned model. This can either be
                     a list of bitstrs of the correct probability or the
                     true solution vector (only jax/tensorflow currently
                     supported) or a dictionary mapping bitstrings to their
                     true probabilities
        :return: a 3-tuple:
                - the solution after iteration (as jax array/tensorflow tensor)
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

        while iteration < max_iters and diff > tol:
            tracker = self.log_performance(tracker, soln, iteration)
            diff = self.train_iter()
            iteration += 1
            if self.verbose:
                pbar.update()

        tracker = self.log_performance(tracker, soln, iteration)
        if self.library == 'jax':
            if self.verbose:
                logging.info("Waiting for JAX to return control flow...")
            tracker.block_until_ready()

        return self.guess, iteration, tracker[:iteration + 1, 0]

    def train_iter(self) -> float:
        """
            Dispatcher for IBU iteration
        :return: the norm difference between the updated parameters and previous
                 parameters
        """
        if self.library == 'tensorflow':
            return self._train_iter_tf()
        elif self.library == 'jax':
            return self._train_iter_jax()
        else:
            raise "Unsupported library!"

    @tf.function
    def _train_iter_tf(self) -> float:
        """
            A single (tensorflow) iteration of IBU
        :return: the norm difference between the updated parameters and previous
                 parameters
        """
        # Compute renormalizer P(o) needed to compute P(t|o)
        obs_guess = self._kron_matmul_tf(self.mats, self._guess)

        # Update estimate of P(t)
        if self.use_log:
            # if priors have 0, log will take them to inf; be careful
            # if adding/subtracting from infinity!
            eq1 = self.obs - obs_guess
            eq2 = self._kron_matmul_tf(self._matsT, eq1)
            diff = tf.linalg.norm(tf.math.exp(self._guess + eq2) -
                                  tf.math.exp(self._guess), ord=1)
            self._guess.assign(self._guess + eq2)
            return diff
        else:
            eq1 = tf.math.divide(self.obs, obs_guess)
            eq2 = self._kron_matmul_tf(self._matsT, eq1)
            diff = tf.linalg.norm((self._guess * eq2) - self._guess, ord=1)
            self._guess.assign(tf.math.multiply(self._guess, eq2))
            return diff

    def _train_iter_jax(self) -> float:
        """
            A single (jax) iteration of IBU
        :return: the norm difference between the updated parameters and previous
                 parameters
        """
        # Compute renormalizer P(o) needed to compute P(t|o)
        obs_guess = self._kron_matmul_jax(self.mats, self._guess)

        # Update estimate of P(t)
        if self.use_log:
            # if priors have 0, log will take them to inf; be careful
            # if adding/subtracting from infinity!
            eq1 = self.obs - obs_guess
            eq1 = jnp.nan_to_num(eq1)
            eq2 = self._kron_matmul_jax(self._matsT, eq1)
            diff = jnp.linalg.norm(jnp.exp(self._guess + eq2)
                                   - jnp.exp(self._guess), ord=1)
            self._guess = self._guess + eq2
        else:
            eq1 = jnp.divide(self.obs, obs_guess)
            eq1 = jnp.nan_to_num(eq1)
            eq2 = self._kron_matmul_jax(self._matsT, eq1)
            diff = jnp.linalg.norm((self._guess * eq2) - self._guess, ord=1)
            self._guess = self._guess * eq2

        return diff

    ############################################################################
    #                                LOGGING
    ############################################################################

    def initialize_tracker(self, max_iters: int) \
            -> Union[jnp.ndarray, tf.Variable]:
        """
            Initialize jax ndarray or tensorflow Tensor of length max_iters
            to track progress after each iteration of IBU.
        :param max_iters: maximum number of iterations that may be tracked
        :return: a jax ndarray/tensorflow tensor of zeros
        """

        if self.library == 'tensorflow':
            return np.zeros([max_iters, 1])
        elif self.library == 'jax':
            return jnp.zeros([max_iters, 1])
        else:
            raise "Unsupported library!"

    def log_performance(self, tracker: Union[jnp.ndarray, tf.Variable],
                        soln: Union[dict, List[str], jnp.ndarray, tf.Tensor],
                        idx: int) -> Union[jnp.ndarray, tf.Variable]:
        """
            Logs the performance of the current self._guess.
            If soln is a list of bitstrs, tracker tracks the probability
            assigned to these bitstrs. If soln is a vector or a dictionary
            mapping bitstrings to their true probabilities or
            log probabilities, tracker tracks the l1-norm error with the current
            guess (in the original space/not the log space).

        :param tracker: the array in which to log performance
        :param soln: the solution (either list of correct keys/true prob vec/
                     dict of keys and true probs)
        :param idx: the index at which to log performance in tracker
        :return: the updated tracker as jax ndarray/tensorflow Tensor.
        """
        if soln is not None:
            if type(soln) == list or type(soln) == tuple:
                res = self.get_prob(soln)
            else:
                res = self.get_l1_error(soln)

            if self.library == 'tensorflow':
                tracker[idx] = float(res)
            elif self.library == 'jax':
                tracker = tracker.at[idx].set(res)
            else:
                raise "Unsupported Library!"

        return tracker

    def get_prob(self, soln: List[str]):
        """
            Given a list of bitstrings, returns the total probability assigned
            by the current guess to those bitstrings
        :param soln: A list of bitstrings for which to get total probability for
        :return: a tf constant or jax DeviceArray of a float representing
                 probability
        """
        if self.library == 'tensorflow':
            prob = tf.constant(0.0, dtype=tf.double)
            for sol in soln:
                if self.use_log:
                    prob += tf.math.exp(self._guess[int(sol[::-1], 2)])
                else:
                    prob += self._guess[int(sol[::-1], 2)]
            return prob
        elif self.library == 'jax':
            prob = jnp.zeros([1, 1])
            for sol in soln:
                if self.use_log:
                    prob += jnp.exp(self._guess[int(sol[::-1], 2)])
                else:
                    prob += self._guess[int(sol[::-1], 2)]
            return prob[0, 0]
        else:
            raise "Unsupported Library!"

    def get_l1_error(self, soln: Union[dict, jnp.ndarray, tf.Tensor]) -> float:
        """
            Returns the norm error between the current guess and the provided
            solution. If the current guess and provided solution are log
            probabilities, the vector is first elementwise-exponentiated before
            taking the norm.
        :param soln: a jax ndarray or tensorflow Tensor or a dictionary mapping
                     bitstrings to their true probabilities/log probabilities
        :return: float, norm error between guess and provided soln
        """
        if self.library == 'jax':
            if type(soln) == dict:
                guess_copy = (jnp.copy(self.guess),
                              jnp.exp(self.guess))[self.use_log]
                for key, val in soln.items():
                    soln_prob = (-val, -jnp.exp(val))[self.use_log]
                    guess_copy = guess_copy.at[int(key[::-1], 2)].add(soln_prob)
                err = jnp.linalg.norm(guess_copy, ord=1)
            else:
                if self.use_log:
                    err = jnp.linalg.norm(jnp.exp(self.guess) - jnp.exp(soln),
                                          ord=1)
                else:
                    err = jnp.linalg.norm(self.guess - soln, ord=1)

        elif self.library == 'tensorflow':
            if type(soln) == dict:
                guess_cp = (tf.Variable(tf.identity(self.guess)),
                            tf.Variable(tf.math.exp(self.guess)))[self.use_log]
                for key, val in soln.items():
                    soln_prob = (-val, -tf.math.exp(val))[self.use_log]
                    rev_key = int(key[::-1], 2)
                    guess_cp[rev_key].assign(guess_cp[rev_key] + soln_prob)
                err = tf.linalg.norm(guess_cp, ord=1)
            else:
                if self.use_log:
                    err = tf.linalg.norm(tf.math.exp(self.guess)
                                         - tf.math.exp(soln), ord=1)
                else:
                    err = tf.linalg.norm(self.guess - soln, ord=1)

        else:
            raise "Unrecognized library!"

        return err

    def get_linf_error(self, soln: Union[dict]) -> float:
        """
            Returns the norm error between the current guess and the provided
            solution. If the current guess and provided solution are log
            probabilities, the vector is first elementwise-exponentiated before
            taking the norm.
        :param soln: a dictionary mapping bitstrings to their true
        probabilities/log probabilities
        :return: float, norm error between guess and provided soln
        """
        if self.library == 'jax':
            guess_copy = (jnp.copy(self.guess),
                          jnp.exp(self.guess))[self.use_log]
            for key, val in soln.items():
                soln_prob = (-val, -jnp.exp(val))[self.use_log]
                guess_copy = guess_copy.at[int(key[::-1], 2)].add(soln_prob)
            err = jnp.linalg.norm(guess_copy, ord=jnp.inf)

        elif self.library == 'tensorflow':
            guess_cp = (tf.Variable(tf.identity(self.guess)),
                        tf.Variable(tf.math.exp(self.guess)))[self.use_log]
            for key, val in soln.items():
                soln_prob = (-val, -tf.math.exp(val))[self.use_log]
                rev_key = int(key[::-1], 2)
                guess_cp[rev_key].assign(guess_cp[rev_key] + soln_prob)
            err = tf.linalg.norm(guess_cp, ord=np.inf)

        else:
            raise "Unrecognized library!"

        return float(err)

    ############################################################################
    #                                 UTILS
    ############################################################################

    def mats_to_kronstruct(self, mats_raw: List[np.ndarray], transpose: bool) \
            -> Union[jnp.ndarray, tf.linalg.LinearOperatorKronecker]:
        """
            Helper function to convert list of numpy matrices of single-qubit
            error probabilities to jax/tensorflow tensor.
        :param mats_raw: list of 2x2 numpy arrays of single-qubit error
         probabilities, in reverse order their respective qubits appear in
         bitstrings.
        :param transpose: whether to transpose each matrix
        :return: jax ndarray or tensorflow LinearOperatorKronecker
        """

        if self.library == 'tensorflow':
            if transpose:
                kronmats = [tf.linalg.LinearOperatorFullMatrix(
                    tf.transpose(mat)) for mat in mats_raw]
            else:
                kronmats = [tf.linalg.LinearOperatorFullMatrix(mat)
                            for mat in mats_raw]
            kronmats = tf.linalg.LinearOperatorKronecker(kronmats)

        elif self.library == 'jax':
            if transpose:
                kronmats = jnp.array([mat.transpose() for mat in mats_raw])
            else:
                kronmats = jnp.array(mats_raw)
        else:
            raise "Unsupported library!"

        return kronmats

    def trace_out(self,  idx):
        result = jnp.transpose(self.guess)  # 1 x 2**n
        j = 1
        for i in jnp.arange(self.num_qubits - 1, -1, -1):
            result = jnp.reshape(result, (2 ** (self.num_qubits-j), 2))
            if i == idx:
                result = jnp.transpose(result).sum(0)  # 2 x 2**n-1
                j = 2
        result = jnp.reshape(result, (2 ** (self.num_qubits-1), 1))
        return result
