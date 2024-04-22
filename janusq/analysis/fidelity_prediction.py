'''
Author: name/jxhhhh� 2071379252@qq.com
Date: 2024-04-17 03:33:02
LastEditors: name/jxhhhh� 2071379252@qq.com
LastEditTime: 2024-04-21 10:53:10
FilePath: /JanusQ/janusq/analysis/fidelity_prediction.py
Description: 

Copyright (c) 2024 by name/jxhhhh� 2071379252@qq.com, All Rights Reserved. 
'''
import random
import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import vmap
from sklearn.model_selection import train_test_split

from janusq.data_objects.circuit import Circuit
from janusq.tools.optimizer import OptimizingHistory
from janusq.tools.ray_func import batch, wait, map
from janusq.tools.saver import dump, load

from .vectorization import RandomwalkModel, extract_device
from tqdm import tqdm
import logging
PARAM_RESCALE = 10000  # helps eliminate loss of significance


class FidelityModel():
    def __init__(self, vec_model: RandomwalkModel):
        '''
        description: fidelity model base on random walk model to vectorized gate, and fidelity model train base on all vectors  
        param {RandomwalkModel} vec_model: random walk model
        '''
        self.vec_model = vec_model
        self.backend = vec_model.backend

        self.error_weights = None

        self.devices = list(self.vec_model.device_to_pathtable.keys())

    def train(self, train_dataset: tuple[list[Circuit], list[float]], validation_dataset: tuple[list[Circuit], list[float]] = None, max_epoch=1000, verbose=True, learning_rate: float = 0.01, multi_process = True):
        '''
        description: using MLP to train a fidelity preiction model
        param {tuple} train_dataset: train dataset
        param {tuple} validation_dataset: validation dataset
        param {int} max_epoch: maximum train epochs
        param {int} verbose: weather print log
        param {float} learning_rate: learning rate of optimizor
        param {bool} multi_process: weather to enable multi-process
        '''
        vec_model = self.vec_model
        if validation_dataset is None:
            train_cirucits, validation_circuits, train_fidelities, validation_fidelities = train_test_split(
                train_dataset[0],  train_dataset[1], test_size=.2)  # min(100, len(train_dataset[0])//5)
            train_dataset = (train_cirucits, train_fidelities)
            validation_dataset = (validation_circuits, validation_fidelities)

        if verbose:
            logging.warn(
                f'len(train dataset) = {len(train_dataset[0])}, len(validation dataset) = {len(validation_dataset[0])}')

        devices = self.devices
        if self.error_weights is not None:
            params = self.error_weights  # finetune
        else:
            params = {
                'gate_params': jnp.zeros(shape=(len(devices), vec_model.max_table_size)),
                'circuit_bias': jnp.zeros(shape=(1,)),
            }

        '''
            A dataset consists of X, Y
            X includes circuits that are transformed inth device_indecies (D) and gate_vecs (GV)
            Y includes the fidelities of the circuits (F)
        '''


        def format(dataset: tuple[list[Circuit], list[float]]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            GV, D, F = [], [], []

            for circuit, fidelity in zip(*dataset):
                device_indices = self._obtain_gate_devices(circuit)
                F.append(fidelity)
                D.append(jnp.array(device_indices, jnp.int32))
                
            def vectorize(circuit, vec_model: RandomwalkModel):
                return jnp.array(vec_model.vectorize(circuit), jnp.float32)
            
            GV = map(vectorize, dataset[0], multi_process, show_progress=False, vec_model = vec_model)
                
            return jnp.array(GV), jnp.array(D), jnp.array(F)



        def constrain(param, constraint):
            param = param.at[param > constraint[1]].set(
                constraint[1])
            param = param.at[param < 0].set(constraint[0])
            return param

        def get_n_gates2circuit(dataset):
            from collections import defaultdict
            n_gates2circuit = defaultdict(list)
            n_gates2fidelity = defaultdict(list)
            for circuit, fidelity in tqdm(zip(*dataset)):
                n_gates = circuit.n_gates
                n_gates2circuit[n_gates].append(circuit)
                n_gates2fidelity[n_gates].append(fidelity)

            for n_gate, same_n_gate_dataset in n_gates2circuit.items():
                n_gates2circuit[n_gate] = format((same_n_gate_dataset, n_gates2fidelity[n_gate]) )
            
            n_gates_list = list(n_gates2circuit.keys())
            n_gates_list.sort()
            # logging.info("n_gates_list:", n_gates_list)
            # logging.info("n_gates_count_list:", [len(n_gates2circuit[ele][0]) for ele in n_gates_list])
            return n_gates2circuit, n_gates_list
        
        
        n_gates2circuit, n_gates_list = get_n_gates2circuit(train_dataset)
        n_gates2circuit_valid, _ = get_n_gates2circuit(validation_dataset)
        
        
        optimizer = optax.adamw(learning_rate=learning_rate)
        opt_state = optimizer.init(params)

        best_params = None
        n_iter_unchange = 10
        unchange_tolerance = 1e-5
        
        

        for epoch in range(max_epoch):
            opt_history = OptimizingHistory(params, learning_rate, unchange_tolerance, n_iter_unchange, max_epoch, -1e10, False)
            
            batch_losses = []
            
            random_gates_list = list(n_gates_list)
            random.shuffle(random_gates_list)

            valid_loss = 0.
            for gate_num in n_gates_list + random_gates_list:  
                for GV, D, F in batch(*n_gates2circuit[gate_num], batch_size=100, should_shuffle=True):
                    loss_value, gradient = jax.value_and_grad(
                        batch_loss)(params, GV, D, F)
                    updates, opt_state = optimizer.update(
                        gradient, opt_state, params)
                    params = optax.apply_updates(params, updates)

                    # 假设一个特征对error贡献肯定小于0.1，大于0
                    params['gate_params'] = constrain(
                        params['gate_params'], [0, PARAM_RESCALE / 10])
                    params['circuit_bias'] = constrain(
                        params['circuit_bias'], [-PARAM_RESCALE / 5, PARAM_RESCALE / 5])

                    batch_losses.append(loss_value)

                if n_gates2circuit_valid.__contains__(gate_num): 
                    valid_loss += batch_loss(params, *n_gates2circuit_valid[gate_num]) / len(n_gates2circuit_valid[gate_num][2])


            opt_history.update(valid_loss, params)
            if opt_history.should_break:
                break

            if verbose and epoch %100 == 0:
                logging.warn(
                    f'epoch: {epoch}, \t epoch loss = {sum(batch_losses)}, \t validation loss = {valid_loss}')
            
            # jax.clear_backends()
            # jax.clear_caches()
            

        self.error_weights = opt_history.best_params
        if verbose:
            logging.warn(f'finish taining')

        return best_params

    def _obtain_gate_devices(self, circuit: Circuit) -> np.array:
        gate_devices = []
        for gate in circuit.gates:
            device = extract_device(gate)
            device_index = self.devices.index(device)
            gate_devices.append(device_index)
        return np.array(gate_devices)

    def predict_circuit_fidelity(self, circuit: Circuit):
        '''
        description: use random walk model to vectorize a circuit and predict its fidelity
        return {float}: circuit fidelity
        '''
        error_params = self.error_weights

        gate_devices = self._obtain_gate_devices(circuit)

        vecs = self.vec_model.vectorize(circuit)
        return predict_circuit_fidelity(
            error_params, vecs, gate_devices)


    def predict_gate_fidelities(self, circuit: Circuit):
        '''
        description: predict each gate fidelity
        param {Circuit} circuit: target circuit
        return {np.ndarray} gate fidelities
        '''
        error_params = self.error_weights

        gate_devices = self._obtain_gate_devices(circuit)

        vecs = self.vec_model.vectorize(circuit)

        gate_errors = np.array(vmap(predict_gate_error, in_axes=(
            None, 0, 0), out_axes=0)(error_params, vecs, gate_devices))

        gate_errors[gate_errors < 0] = 0
        gate_errors[gate_errors > 1] = 1

        return 1 - gate_errors


    def get_all_path_errors(self):
        all_path_errors = []

        for device, pathtable in self.vec_model.device_to_pathtable.items():
            for path, path_index in pathtable.items():
                error = self.error_weights['gate_params'][self.devices.index(
                    device)][path_index]
                all_path_errors.append((path, float(error)))
        all_path_errors.sort(key=lambda elm: -elm[1])
        return all_path_errors
    


    
    def plot_path_error(self, title='', top_k=20, save_path=None, plot=True):
        '''
            draw graph to analyze the path errors
        '''
        all_path_errors = self.get_all_path_errors()
        paths = [
            path
            for path, error in all_path_errors[:top_k]
        ]
        errors = [
            error
            for path, error in all_path_errors[:top_k]
        ]
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(top_k, 4))

        plt.bar(paths, errors, width=0.8)
        plt.xticks(rotation=45)

        plt.title(title, size=26)

        # 设置轴标签
        plt.xlabel('path', size=28)
        plt.ylabel("error", size=28)

        if save_path is not None:
            plt.savefig(save_path)

        # if plot:
        #     plt.show()

        return fig

    @staticmethod
    def load(name):
        return load(name)

    def save(self, name):
        dump(name, self)

def predict_gate_error(params, gate_vecs, device):
    '''predict the error of gates'''
    error = jnp.dot((params['gate_params'][device] / PARAM_RESCALE), gate_vecs)
    return error


@jax.jit
def predict_circuit_fidelity(params, gate_vecs, devices):
    '''预测电路的保真度'''
    errors = vmap(predict_gate_error, in_axes=(None, 0, 0),
                  out_axes=0)(params, gate_vecs, devices)
    return jnp.prod(1 - errors, axis=0) + params['circuit_bias'][0]/PARAM_RESCALE


@jax.jit
def loss_func(params, vecs, devices, true_fidelity):
    predict_fidelity = predict_circuit_fidelity(params, vecs, devices)
    return optax.l2_loss(true_fidelity - predict_fidelity) * 100


def batch_loss(params, GV, D, F):
    losses = vmap(loss_func, in_axes=(None, 0, 0, 0),
                  out_axes=0)(params, GV, D, F)
    return jnp.array(losses).sum()
