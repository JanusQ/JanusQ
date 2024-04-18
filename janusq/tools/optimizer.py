import logging
import jax.numpy as jnp
import time


class OptimizingHistory(object):
    def __init__(self, params: jnp.ndarray, learning_rate: float = 0.01, unchange_tol=0.001, n_iter_unchange=100, max_epoch=10000, allowed_dist=0.0001, verbose: bool = False):
        self.learning_rate = learning_rate
        self.unchange_tol = unchange_tol
        self.n_iter_unchange = n_iter_unchange
        self.max_epoch = max_epoch
        self.allowed_dist = allowed_dist
        self.verbose = verbose

        self.min_loss = 1e2
        self.best_params = params

        self.loss_decrement_history = []
        self.epcoh_time_costs = []
        self.epoch = 0
        self.epoch_start_time = time.time()

        self.should_break: bool = False

    def update(self, loss_value: float, params: jnp.ndarray):
        loss_decrement_history = self.loss_decrement_history
        n_iter_unchange = self.n_iter_unchange
        epoch = self.epoch

        loss_decrement_history.append(self.min_loss - loss_value)

        if self.min_loss > loss_value and epoch != 0:  # 如果很开始就set了，就可能会一下子陷入全局最优
            self.min_loss = loss_value
            self.best_params = params

        if epoch < n_iter_unchange:
            loss_unchange = False
        else:
            loss_unchange = True
            for loss_decrement in loss_decrement_history[-n_iter_unchange:]:
                if loss_decrement > self.unchange_tol:
                    loss_unchange = False

        if loss_unchange or epoch > self.max_epoch or loss_value < self.allowed_dist:
            self.should_break = True
        else:
            self.should_break = False

        self.epoch += 1
        self.epcoh_time_costs.append(time.time() - self.epoch_start_time)
        self.epoch_start_time = time.time()

        if self.verbose and epoch % 100 == 0:
            epcoh_time_costs = self.epcoh_time_costs[-50:]
            logging.info('Epoch: {:5d} | Loss: {:.5f}  | Dist: {:.5f} | Time: {:.3f}'.format(
                epoch, loss_value, loss_value, sum(epcoh_time_costs)/(len(epcoh_time_costs))))
