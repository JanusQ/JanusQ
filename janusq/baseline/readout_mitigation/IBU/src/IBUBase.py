from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import jax.numpy as jnp
import tensorflow as tf


class IBUBase(ABC):

    ############################################################################
    #                                PROPERTIES
    ############################################################################

    @property
    @abstractmethod
    def num_qubits(self):
        pass

    @property
    @abstractmethod
    def library(self):
        pass

    @property
    @abstractmethod
    def use_log(self):
        pass

    @property
    @abstractmethod
    def verbose(self):
        pass

    @verbose.setter
    @abstractmethod
    def verbose(self, value: bool):
        pass

    @property
    @abstractmethod
    def mats(self):
        pass

    @property
    @abstractmethod
    def obs(self):
        pass

    @property
    @abstractmethod
    def init(self):
        pass

    @property
    @abstractmethod
    def guess(self):
        pass

    ############################################################################
    #                     DATA PROCESSING / GENERATION
    ############################################################################

    @abstractmethod
    def set_obs(self, obs: Union[dict, np.array, jnp.array, tf.Tensor]):
        pass

    @abstractmethod
    def generate_obs(self, t_raw):
        pass

    @abstractmethod
    def initialize_guess(self, init=None, smoother=0.0):
        pass

    ############################################################################
    #                                 TRAIN
    ############################################################################

    @abstractmethod
    def train(self, max_iters: int = 100, tol: float = 1e-3):
        pass

    @abstractmethod
    def train_iter(self):
        pass

    ############################################################################
    #                                 LOGGING
    ############################################################################

    @abstractmethod
    def log_performance(self, tracker, soln, idx):
        pass

    @abstractmethod
    def get_l1_error(self, soln):
        pass

    @abstractmethod
    def get_linf_error(self, soln):
        pass
