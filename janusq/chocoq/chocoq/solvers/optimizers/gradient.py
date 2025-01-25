import numpy as np
from tqdm import tqdm

from chocoq.utils import iprint
from chocoq.utils.gadget import get_main_file_info, create_directory_if_not_exists

from chocoq.solvers.options.optimizer_option import AdamOptimizerOption as OptimizerOption
from .abstract_optimizer import Optimizer


class AdamOptimizer(Optimizer):
    def __init__(
        self,
        *,
        max_iter: int = 50,
        learning_rate: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999
    ):
        super().__init__()
        self.optimizer_option: OptimizerOption = OptimizerOption(max_iter=max_iter)
        self.optimizer_option.learning_rate = learning_rate
        self.optimizer_option.beta1 = beta1
        self.optimizer_option.beta2 = beta2
        # optimizer_option.opt_id

    def minimize(self):
        optimizer_option = self.optimizer_option
        obj_dir = optimizer_option.obj_dir
        cost_func = optimizer_option.cost_func
        cost_func_trans = self.obj_dir_trans(obj_dir, cost_func)
        params = self._initialize_params(optimizer_option.num_params)
        max_iter = optimizer_option.max_iter
        learning_rate = optimizer_option.learning_rate
        beta1 = optimizer_option.beta1
        beta2 = optimizer_option.beta2
        num_consecutive_iter = 5
        early_stopping_threshold = 0.0001

        def gradient_by_param_shift(params, cost_function):
            num_params = len(params)
            gradients = np.zeros(num_params)
            shift = 0.01
            for i in range(num_params):
                shifted_params = params.copy()
                shifted_params[i] += shift
                forward = cost_function(shifted_params)
                shifted_params[i] -= 2 * shift
                backward = cost_function(shifted_params)
                gradients[i] = (forward - backward) / (2 * shift)

            print(gradients)
            return gradients

        eps = 1e-8
        m = np.zeros(len(params))
        v = np.zeros(len(params))
        best_params = params
        best_cost = cost_func(params)
        prev_cost = best_cost  # 跟踪先前成本
        consecutive_no_improvement = 0  # 没有明显提升的连续迭代次数
        costs_list = []

        with tqdm(total=max_iter) as pbar:
            for iter in range(max_iter):
                gradients = gradient_by_param_shift(params, cost_func_trans)
                m = beta1 * m + (1 - beta1) * gradients
                v = beta2 * v + (1 - beta2) * gradients**2
                m_hat = m / (1 - beta1 ** (iter + 1))
                v_hat = v / (1 - beta2 ** (iter + 1))
                params -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
                cost = cost_func(params)
                costs_list.append(float(cost))
                pbar.set_postfix(cost=cost)
                pbar.update(1)

                if cost < best_cost:
                    best_cost = cost
                    best_params = params

                if abs(prev_cost - cost) < early_stopping_threshold:
                    consecutive_no_improvement += 1
                    # consecutive iterations
                    if consecutive_no_improvement >= num_consecutive_iter:
                        break
                else:
                    consecutive_no_improvement = 0
                    prev_cost = cost

        if consecutive_no_improvement >= num_consecutive_iter:
            iprint("early stopping: loss change is below threshold.")

        return best_params, iter

    # create_directory_if_not_exists("cost_loss")
    # import csv

    # with open("cost_loss/" + opt_id + ".csv", mode="w", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["method", "loss"])
    #     for row in costs_list:
    #         writer.writerow([opt_id, row])

    # iprint("====")
    # iprint(costs_list)
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(10, 5))
    # plt.plot(range(len(costs_list)), costs_list, marker="o")
    # plt.xlabel("Iteration")
    # plt.ylabel("Cost")
    # plt.grid(True)
    # plt.show()
    # # +
    # return best_params, iter


# def train_gradient(num_params, cost_function, max_iter, learning_rate, beta1, beta2):


# def gradient_by_param_shift_pauli(params, cost_function):
#     num_params = len(params)
#     gradients = np.zeros(num_params)
#     shift = np.pi / 2
#     for i in range(num_params):
#         shifted_params = params.copy()
#         shifted_params[i] += shift
#         forward = cost_function(shifted_params)
#         shifted_params[i] -= 2 * shift
#         backward = cost_function(shifted_params)
#         gradients[i] = 0.5 * (forward - backward)
#     return gradients

# def gradient_by_param_shift(params, cost_function):
#     num_params = len(params)
#     gradients = np.zeros(num_params)
#     shift = 0.01
#     for i in range(num_params):
#         shifted_params = params.copy()
#         shifted_params[i] += shift
#         forward = cost_function(shifted_params)
#         shifted_params[i] -= 2 * shift
#         backward = cost_function(shifted_params)
#         gradients[i] = (forward - backward)/(2*shift)
#     return gradients

# def adam_optimizer(params, cost_function, max_iter, learning_rate, beta1, beta2, num_consecutive_iter = 5, early_stopping_threshold=0.0001, opt_id = None):
#     eps = 1e-8
#     m = np.zeros(len(params))
#     v = np.zeros(len(params))
#     best_params = params
#     best_cost = cost_function(params)
#     prev_cost = best_cost  # 跟踪先前成本
#     consecutive_no_improvement = 0  # 没有明显提升的连续迭代次数
#     costs_list = []
#     with tqdm(total=max_iter) as pbar:
#         for iter in range(max_iter):
#             gradients = gradient_by_param_shift(params, cost_function)
#             m = beta1 * m + (1 - beta1) * gradients
#             v = beta2 * v + (1 - beta2) * gradients ** 2
#             m_hat = m / (1 - beta1 ** (iter + 1))
#             v_hat = v / (1 - beta2 ** (iter + 1))
#             params -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
#             cost = cost_function(params)
#             costs_list.append(float(cost))
#             pbar.set_postfix(cost=cost)
#             pbar.update(1)
#             if cost < best_cost:
#                 best_cost = cost
#                 best_params = params
#             if abs(prev_cost - cost) < early_stopping_threshold:
#                 consecutive_no_improvement += 1
#                 if consecutive_no_improvement >= num_consecutive_iter:  # consecutive iterations
#                     iprint("Early stopping: Loss change is below threshold.")
#                     break
#             else:
#                 consecutive_no_improvement = 0
#                 prev_cost = cost

#     create_directory_if_not_exists('cost_loss')
#     import csv
#     with open('cost_loss/' + opt_id + '.csv', mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['method', 'loss'])
#         for row in costs_list:
#             writer.writerow([opt_id, row])

#     iprint('====')
#     iprint(costs_list)
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(len(costs_list)), costs_list, marker='o')
#     plt.xlabel('Iteration')
#     plt.ylabel('Cost')
#     plt.grid(True)
#     plt.show()
#     #+
#     return best_params, iter

# # def train_gradient(num_params, cost_function, max_iter, learning_rate, beta1, beta2):
# def train_gradient(optimizer_option: OptimizerOption):
#     # , requires_grad=False
#     params = 2*np.pi*np.random.uniform(0, 1, optimizer_option.num_params)
#     return adam_optimizer()
