# importing torch
import torch
import numpy as np
from torchdiffeq import odeint

# importing multiprocessing
from torch.multiprocessing import Pool, set_start_method
from itertools import starmap

# importing numba
from numba import cuda, jit, float64, float32, int32, int8
import math


I = 1j


# For multiprocessing!
def main():
    pass

if __name__ == '__main__':
    main()


# Inputs must be torch.tensors, outputs are torch.tensors
class KuramotoModel:

    def __init__(self, adjacency_matrix, interaction_consts=torch.tensor([1e-3]), freqs=torch.tensor([0]), device=torch.device('cpu'), dtype=torch.float64):
        # adjacency_matrix - torch matrix of graph with shape [num_nodes, num_nodes]
        # interaction_consts - torch vector with shape [num_nodes] or torch number
        # freqs - torch vector with shape [num_nodes] or torch number
        self.device = device
        # in release version it is better to specify presicion of calculations, now all types are float64
        self.dtype = dtype
        self.num_nodes = adjacency_matrix.shape[0]
        self.adjacency_matrix = adjacency_matrix.to(device=device, dtype=dtype)
        self.interaction_consts = interaction_consts.to(device=device, dtype=dtype)
        self.freqs = freqs.to(device=device, dtype=dtype)

    def update_adjacency_matrix(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix.to(device=self.device, dtype=self.dtype)
        self.num_nodes = adjacency_matrix.shape[0]

    def update_interaction_consts(self, interaction_consts):
        self.interaction_consts = interaction_consts.to(device=self.device, dtype=self.dtype)
 
    def update_freqs(self, freqs):
        self.freqs = freqs.to(device=self.device, dtype=self.dtype)

    def to(self, device):
        self.device = device
        self.adjacency_matrix = self.adjacency_matrix.to(device)
        self.interaction_consts = self.interaction_consts.to(device)
        self.freqs = self.freqs.to(device)

    #####################################################################
    def evolution_func(self, t, phases):
        phases = phases.view(self.num_nodes, -1)
        interaction_part = self.adjacency_matrix * torch.sin(phases.T - phases)
        interaction_part = self.interaction_consts * torch.sum(interaction_part, dim=1)
        out_phases = self.freqs + interaction_part
        return out_phases

    def evaluate(self, timesteps, initial_phases):
        time = torch.arange(timesteps, dtype=self.dtype, device=self.device)
        initial_phases = initial_phases.to(device=self.device, dtype=self.dtype)
        return odeint(self.evolution_func, initial_phases, time)

    # another realization, which allows using batches: phases[batch_size, num_nodes]
    def evolution_func_(self, t, phases):
        res = torch.cos(phases) * torch.matmul(torch.sin(phases), self.adjacency_matrix.T) - torch.sin(phases) \
            * torch.matmul(torch.cos(phases), self.adjacency_matrix.T)
        res = self.freqs + self.interaction_consts * res
        return res
        
    def evaluate_(self, timesteps, initial_phases):
        time = torch.arange(timesteps, dtype=self.dtype, device=self.device)
        initial_phases = initial_phases.to(device=self.device, dtype=self.dtype)
        return odeint(self.evolution_func_, initial_phases, time)
    
    ########################################################################
    @staticmethod
    def evaluate_pool(adjacency_matrix, interaction_const, freqs, timesteps, initial_phases):
        time = torch.arange(timesteps, dtype=torch.float64).share_memory_()
        
        def evolution_func_pool(t, phases):
            phases = phases.view(adjacency_matrix.shape[0], -1)
            interaction_part = adjacency_matrix * torch.sin(phases.T - phases)
            interaction_part = interaction_const * torch.sum(interaction_part, dim=1)
            out_phases = freqs + interaction_part
            return out_phases
        
        return odeint(evolution_func_pool, initial_phases, time)

    # another realization, which allows using batches: phases[batch_size, num_nodes]  
    @staticmethod
    def evaluate_pool_(adjacency_matrix, interaction_const, freqs, timesteps, initial_phases):
        time = torch.arange(timesteps, dtype=torch.float64).share_memory_()
        
        def evolution_func_pool_(t, phases):
            res = torch.cos(phases) * torch.matmul(torch.sin(phases), adjacency_matrix.T) - torch.sin(phases) \
                * torch.matmul(torch.cos(phases), adjacency_matrix.T)
            res = freqs + interaction_const * res
            return res
        
        return odeint(evolution_func_pool_, initial_phases, time)

    ##########################################################################
    @staticmethod
    def order_parameter(phases):
        return torch.abs(torch.mean(torch.exp(I * phases), axis=1))

    def __call__(self, timesteps, initial_phases):
        return self.evaluate(self, timesteps, initial_phases)

    # main generating loop
    def gen_order_parameters_(self, interaction_consts, timesteps=1000, thermalization_time=0, initial_phases=None, method=True):
        if initial_phases is None:
            initial_phases = torch.zeros(self.num_nodes, dtype=self.dtype, device=self.device)
        else:
            initial_phases = initial_phases.to(device=self.device, dtype=self.dtype)
        
        prev_interaction_consts = self.interaction_consts
        order_parameters = []
        
        for interaction_const in interaction_consts:
            self.update_interaction_consts(interaction_const)
            if method is True:
                phases = self.evaluate(timesteps, initial_phases)
            else:
                phases = self.evaluate_(timesteps, initial_phases)
            order_parameter = torch.mean(self.order_parameter(phases[thermalization_time:]))
            order_parameters.append(order_parameter)
        
        self.update_interaction_consts(prev_interaction_consts)
        
        return torch.tensor(order_parameters, dtype=self.dtype, device=self.device)
        
    # function for graph perturbating
    def gen_order_parameters(self, adjacency_matrix, interaction_consts, freqs, timesteps=1000, thermalization_time=0, initial_phases=None):
        prev_adjacency_matrix = self.adjacency_matrix
        prev_interaction_consts = self.interaction_consts
        prev_freqs = self.freqs
        
        self.update_adjacency_matrix(adjacency_matrix)
        self.update_interaction_consts(interaction_consts)
        self.update_freqs(freqs)
        
        if self.device == torch.device('cpu'):
            order_parameters = self.gen_order_parameters_pool(interaction_consts, timesteps=timesteps, thermalization_time=thermalization_time, initial_phases=initial_phases, method=True)
        else:
            order_parameters = self.gen_order_parameters_(interaction_consts, timesteps=timesteps, thermalization_time=thermalization_time, initial_phases=initial_phases, method=True)
        
        self.update_adjacency_matrix(prev_adjacency_matrix)
        self.update_interaction_consts(prev_interaction_consts)
        self.update_freqs(prev_freqs)
        
        return order_parameters
    
    # CPU only!
    def gen_order_parameters_pool(self, interaction_consts, timesteps, thermalization_time=0, initial_phases=None, method=True):
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        self.to(torch.device('cpu'))
        if initial_phases is None:
            initial_phases = torch.zeros(self.num_nodes, dtype=self.dtype, device=self.device).share_memory_()
        else:
            initial_phases = initial_phases.to(device=self.device, dtype=self.dtype).share_memory_()
            
        run_func = KuramotoModel.evaluate_pool if method is True else KuramotoModel.evaluate_pool_
        adjacency_matrix, freqs = self.adjacency_matrix.share_memory_(), self.freqs.share_memory_()
        
        # debug condition
        if False:
            # use standart maping
            phases = list(starmap(run_func, [(adjacency_matrix, interaction_const, freqs, timesteps, initial_phases) for interaction_const in interaction_consts.share_memory_()]))
        else:
            # use multiprocessing maping
            pool = Pool(processes=8)
            phases = list(pool.starmap(run_func, [(adjacency_matrix, interaction_const, freqs, timesteps, initial_phases) for interaction_const in interaction_consts.share_memory_()]))
            pool.close()
            pool.join()
        
        order_parameters = list(map(lambda x: torch.mean(self.order_parameter(x[thermalization_time:])), phases))
        
        return torch.tensor(order_parameters, dtype=self.dtype, device=self.device)


# GPU only!
# Inputs must be np.arrays, outputs are np.arrays
class KuramotoModelNumba():

    def __init__(self, adjacency_matrix, interaction_consts, freqs):
        # adjacency_matrix - array-like matrix of graph with shape [num_nodes, num_nodes]
        # interaction_consts - array-like vector with shape [blocks]
        # freqs - array-like vector with shape [num_nodes]
        self.num_nodes = adjacency_matrix.shape[0]
        self.adjacency_matrix = np.array(adjacency_matrix, dtype=np.int8).flatten()
        self.blocks = interaction_consts.shape[0]
        self.interaction_consts = np.array(interaction_consts, dtype=np.float32)
        self.freqs = np.array(freqs, dtype=np.int8)
        self.kernel = KuramotoModelNumba.create_gpu_kernel(self.num_nodes)
    
    def update_adjacency_matrix(self, adjacency_matrix):
        if adjacency_matrix.size != self.num_nodes * self.num_nodes:
            print("Shape mismatch! You might build new copy of class for such purpose!")
            raise ValueError
        self.adjacency_matrix = np.array(adjacency_matrix, dtype=np.int8).flatten()

    def update_interaction_consts(self, interaction_consts):
        self.blocks = interaction_consts.shape[0]
        self.interaction_consts = np.array(interaction_consts, dtype=np.float32)
 
    def update_freqs(self, freqs):
        if freqs.shape[0] != self.num_nodes:
            print("Shape mismatch! You might build new copy of class for such purpose!")
            raise ValueError
        self.freqs = np.array(freqs, dtype=np.int8)
    
    def __call__(self, timesteps=1000, dt=0.1, thermalization_time=0, stream_num=1, initial_phases=None):
        return self.gen_order_parameters_(self, timesteps, dt, thermalization_time, stream_num, initial_phases)
    
    # main generating loop
    def gen_order_parameters_(self, timesteps=1000, dt=0.1, thermalization_time=0, stream_num=1, initial_phases=None):
        # initail_phases must has shape[stream_num, num_modes] or be None
        if initial_phases is None:
            initial_phases = 2 * np.pi * np.random.rand(stream_num, self.num_nodes).astype(np.float64) - np.pi
            
        # Passing constants to gpu and run

        streams = []
        for i in range(stream_num):
            streams.append(cuda.stream())

        order_parameters = np.zeros((stream_num, self.blocks), dtype=np.float64)

        for i, stream in enumerate(streams):
            d_adjacency_matrix = cuda.to_device(self.adjacency_matrix, stream)
            d_interaction_consts = cuda.to_device(self.interaction_consts, stream)
            d_freqs = cuda.to_device(self.freqs, stream)
            d_initial_phases = cuda.to_device(initial_phases[i], stream)

            # here we put a result of order parameters
            d_order_parameters = cuda.device_array(self.blocks, stream=stream)
               
            self.kernel[self.blocks, self.num_nodes, stream, 0] \
                (d_adjacency_matrix, d_interaction_consts, d_freqs, d_initial_phases, timesteps, dt, thermalization_time, d_order_parameters)
            order_parameters[i] = d_order_parameters.copy_to_host(stream=stream) / (timesteps - thermalization_time)
            stream.synchronize()

        cuda.synchronize()
        order_parameters = order_parameters.mean(axis=0)
        
        return order_parameters
        
    # function for graph perturbating
    def gen_order_parameters(self, adjacency_matrix, interaction_consts, freqs, timesteps=1000, thermalization_time=0, initial_phases=None):
        prev_adjacency_matrix = self.adjacency_matrix
        prev_interaction_consts = self.interaction_consts
        prev_freqs = self.freqs
        
        self.update_adjacency_matrix(adjacency_matrix)
        self.update_interaction_consts(interaction_consts)
        self.update_freqs(freqs)
        
        order_parameters = self.gen_order_parameters_(timesteps, dt=0.1, thermalization_time=thermalization_time, stream_num=1, initial_phases=initial_phases)
        
        self.update_adjacency_matrix(prev_adjacency_matrix)
        self.update_interaction_consts(prev_interaction_consts)
        self.update_freqs(prev_freqs)
        
        return order_parameters
    
    
    @staticmethod
    def create_gpu_kernel(num_nodes):
    
        matrix_size = num_nodes * num_nodes
        num_nodes_2 = 2 * num_nodes

        # Here you can put device function for other DE solving circuits or other models
        @cuda.jit("float64(int8, int8, int8[:], float32[:], int8[:], float64[:], float32)", device=True)
        def kuramoto_model_solver(tid, bid, ds_adjacency_matrix, d_interaction_consts, ds_freqs, ds_current_phases, dt):
            tmp = 0.
            for i in range(num_nodes):
                tmp += ds_adjacency_matrix[i + tid * num_nodes] * math.sin(ds_current_phases[i] - ds_current_phases[tid])
            tmp = (dt) * (ds_freqs[tid] + d_interaction_consts[bid] * tmp)
            return tmp
        
        @cuda.jit("void(int8[:], float32[:], int8[:], float64[:], int64, float64, int64, float64[:])")
        def kernel(d_adjacency_matrix, d_interaction_consts, d_freqs, d_initial_phases, timesteps, dt, thermalization_time, d_order_parameters):
            # d_adjacency_matrix - graph adjacency matrix, d_interaction_consts - interaction constants, d_freqs - natural frequencies
            # d_initial_phases - initial conditions, d_order_parameters - order parameter (return parameter), num_modes - number of nodes
            
            # getting thread information
            tid = cuda.threadIdx.x
            bid = cuda.blockIdx.x
            x = cuda.grid(1)
            
            # memmory allocation
            ds_adjacency_matrix = cuda.shared.array(shape=(matrix_size), dtype=int8)
            ds_freqs = cuda.shared.array(shape=(num_nodes), dtype=int8)
            ds_current_phases = cuda.shared.array(shape=(num_nodes), dtype=float64)
            ds_order_parameters = cuda.shared.array((num_nodes_2), dtype=float64)

            # coping data
            for i in range(num_nodes):
                ds_adjacency_matrix[tid + i * num_nodes] = d_adjacency_matrix[tid + i * num_nodes]
            ds_current_phases[tid] = d_initial_phases[tid]
            ds_freqs[tid] = d_freqs[tid]
            d_order_parameters[bid] = 0
            
            cuda.syncthreads()

            # running DE solving procedure
            for t in range(timesteps):
                # computing phase update
                # tmp = kuramoto_model_solver(tid, bid, ds_adjacency_matrix, d_interaction_consts, ds_freqs, ds_current_phases, dt)
                
                # direct implementation works a little bit faster :)
                tmp = 0.
                for i in range(num_nodes):
                    tmp += ds_adjacency_matrix[i + tid * num_nodes] * math.sin(ds_current_phases[i] - ds_current_phases[tid])
                tmp = (dt) * (ds_freqs[tid] + d_interaction_consts[bid] * tmp)
                
                cuda.syncthreads()
                
                ds_current_phases[tid] += tmp
                
                cuda.syncthreads()
                
                # correcting DE numerical solving
                ds_current_phases[tid] = ds_current_phases[tid] % math.pi
                
                cuda.syncthreads()
                
                # computing order parameter for each node
                ds_order_parameters[tid] = math.sin(ds_current_phases[tid])
                ds_order_parameters[tid + num_nodes] = math.cos(ds_current_phases[tid])
                
                cuda.syncthreads() 
                
                # compute order parameters
                if t >= thermalization_time:
                    if tid == 0:
                        tmp0 = 0.
                        tmp1 = 0.
                        for i in range(num_nodes):
                            tmp0 += ds_order_parameters[i]
                            tmp1 += ds_order_parameters[i + num_nodes]
                        d_order_parameters[bid] += math.sqrt(tmp0 * tmp0 + tmp1 * tmp1) / num_nodes
                    
                cuda.syncthreads()
                
        return kernel


# Inputs must be np.arrays, outputs are np.arrays, but if gen_order_parameters_func use torch.tensors set is_torch_gen == True
class ThermalAnnealing:

    def __init__(self, gen_order_parameters_func,
            interaction_consts,
            initial_adjacency_matrix, initail_freqs,
            timesteps=1000, thermalization_time=0, initial_phases=None, is_torch_gen = False,
            threshold=0.9,
            is_directed=False, save_valency=False, modify_freqs=True, gen_freqs_func=lambda x: x.sum(axis=1), is_weighted=False, is_integer=True, weight_interval=(0, 1),
            annealing_func=lambda x: np.exp(-x), initial_T=300, temperature_decrease_func=lambda x: np.sqrt(1 + x) if x >=0 else 1, temperature_change_steps=10, num_changes_per_num_connections=0.1):
        self.gen_order_parameters_func = gen_order_parameters_func
        self.interaction_consts = interaction_consts
        self.current_adjacency_matrix = initial_adjacency_matrix
        self.current_freqs = initail_freqs
        self.threshold = threshold
        self.timesteps = timesteps
        self.thermalization_time = thermalization_time
        self.initial_phases = initial_phases
        self.is_torch_gen = is_torch_gen
        self.is_directed = is_directed
        self.save_valency = save_valency
        self.modify_freqs = modify_freqs
        self.gen_freqs_func = gen_freqs_func
        self.is_weighted = is_weighted
        self.is_integer = is_integer
        self.weight_interval = weight_interval
        self.annealing_func = annealing_func
        self.initial_T = initial_T
        self.temperature_decrease_func = temperature_decrease_func
        self.temperature_change_steps = temperature_change_steps
        self.num_changes_per_num_connections = num_changes_per_num_connections
    
    def perturb_check(self):
        return self.perturb(self.current_adjacency_matrix, self.current_freqs, self.is_directed, self.save_valency,
                self.modify_freqs, self.gen_freqs_func, self.is_weighted, self.is_integer, self.weight_interval, self.num_changes_per_num_connections)
    
    def perturb(self, current_adjacency_matrix, current_freqs, is_directed, save_valency, modify_freqs, gen_freqs_func, is_weighted, is_integer, weight_interval, num_changes_per_num_connections):
        num_nodes = current_adjacency_matrix.shape[0]
        new_adjacency_matrix = current_adjacency_matrix.copy()
        if is_directed is False:
            to_change = round(0.5 * np.array(current_adjacency_matrix > 0).sum() * num_changes_per_num_connections)
            if save_valency is False:
                possible_pairs = np.array(np.meshgrid(np.arange(num_nodes), np.arange(num_nodes))).T.reshape(-1, 2)
                mask = np.where(possible_pairs[:, 0] > possible_pairs[:, 1], 1, 0)
                possible_pairs = possible_pairs[mask > 0]
                changing_pairs_num = np.random.choice(np.arange(possible_pairs.shape[0]), size=to_change, replace=False)
                changing_pairs = possible_pairs[changing_pairs_num]
                # changing_pairs_merged = np.concatenate((possible_pairs[changing_pairs_num], possible_pairs[changing_pairs_num][..., ::-1]))
                for pair in changing_pairs:
                    if new_adjacency_matrix[pair[0], pair[1]] > 0:
                        new_adjacency_matrix[pair[0], pair[1]] = 0
                        new_adjacency_matrix[pair[1], pair[0]] = 0
                    else:
                        if is_weighted is False:
                            new_adjacency_matrix[pair[0], pair[1]] = 1
                            new_adjacency_matrix[pair[1], pair[0]] = 1
                        else:
                            gen_rand = np.random.uniform(weight_interval[0], weight_interval[1], 1).item()
                            if is_integer is True:
                                new_adjacency_matrix[pair[0], pair[1]] = round(gen_rand)
                                new_adjacency_matrix[pair[1], pair[0]] = round(gen_rand)
                            else:
                                new_adjacency_matrix[pair[0], pair[1]] = gen_rand
                                new_adjacency_matrix[pair[1], pair[0]] = gen_rand
            else:
                raise NotImplementedError
        else:
            if save_valency is False:
                possible_pairs = np.array(np.meshgrid(np.arange(num_nodes), np.arange(num_nodes))).T.reshape(-1, 2)
                mask = np.where(possible_pairs[:, 0] != possible_pairs[:, 1], 1, 0)
                possible_pairs = possible_pairs[mask > 0]
                changing_pairs_num = np.random.choice(np.arange(possible_pairs.shape[0]), size=to_change, replace=False)
                changing_pairs = possible_pairs[changing_pairs_num]
                for pair in changing_pairs:
                    if new_adjacency_matrix[pair[0], pair[1]] > 0:
                        new_adjacency_matrix[pair[0], pair[1]] = 0
                    else:
                        if is_weighted is False:
                            new_adjacency_matrix[pair[0], pair[1]] = 1
                        else:
                            gen_rand = np.random.uniform(weight_interval[0], weight_interval[1], 1).item()
                            if is_integer is True:
                                new_adjacency_matrix[pair[0], pair[1]] = round(gen_rand)
                            else:
                                new_adjacency_matrix[pair[0], pair[1]] = gen_rand
            else:
                raise NotImplementedError
                    
                    
        if modify_freqs is True:
            new_freqs = gen_freqs_func(new_adjacency_matrix)
        else:
            new_freqs = current_freqs.copy()
            
        return new_adjacency_matrix.astype(current_adjacency_matrix.dtype), new_freqs.astype(current_freqs.dtype)
    
    def __call__(self):
        return self.anneal()
    
    def anneal(self):
        # here we save all order parameters for all steps
        order_parameters_history = []
        # here we save chosen interaction constants at each step
        converge_interaction_consts_history = []
        
        if self.is_torch_gen is True:
            order_parameters = self.gen_order_parameters_func(torch.tensor(self.current_adjacency_matrix, dtype=torch.float64), torch.tensor(self.interaction_consts, dtype=torch.float64),
                torch.tensor(self.current_freqs, dtype=torch.float64), self.timesteps, self.thermalization_time, torch.tensor(self.initial_phases, dtype=torch.float64)).cpu().numpy()
        else:
            order_parameters = self.gen_order_parameters_func(self.current_adjacency_matrix, self.interaction_consts,
                self.current_freqs, self.timesteps, self.thermalization_time, self.initial_phases)
        
        if np.allclose(np.zeros_like(order_parameters), np.array((order_parameters >= self.threshold), dtype=np.int32)) is True:
            best_converge_interaction_const = self.interaction_consts[np.argmax(order_parameters)]
        else:
            best_converge_interaction_const = self.interaction_consts[np.argmax(order_parameters >= self.threshold)]
        
        converge_interaction_consts_history.append(best_converge_interaction_const)
        order_parameters_history.append(order_parameters)
        
        for i in range(self.temperature_change_steps):
            
            new_adjacency_matrix, new_freqs = self.perturb(self.current_adjacency_matrix, self.current_freqs, \
                    self.is_directed, self.save_valency, self.modify_freqs, self.gen_freqs_func, self.is_weighted, self.is_integer, self.weight_interval, self.num_changes_per_num_connections)
            
            if self.is_torch_gen is True:
                order_parameters = self.gen_order_parameters_func(torch.tensor(self.current_adjacency_matrix, dtype=torch.float64), torch.tensor(self.interaction_consts, dtype=torch.float64),
                    torch.tensor(self.current_freqs, dtype=torch.float64), self.timesteps, self.thermalization_time, torch.tensor(self.initial_phases, dtype=torch.float64)).cpu().numpy()
            else:
                order_parameters = self.gen_order_parameters_func(self.current_adjacency_matrix, self.interaction_consts,
                    self.current_freqs, self.timesteps, self.thermalization_time, self.initial_phases)
            
            if np.allclose(np.zeros_like(order_parameters), np.array((order_parameters >= self.threshold), dtype=np.int32)) is True:
                current_converge_interaction_const = self.interaction_consts[np.argmax(order_parameters)]
            else:
                current_converge_interaction_const = self.interaction_consts[np.argmax(order_parameters >= self.threshold)]
            
            converge_interaction_consts_history.append(current_converge_interaction_const)
            order_parameters_history.append(order_parameters)
            
            prob = self.annealing_func(1 / self.initial_T * self.temperature_decrease_func(i) * (current_converge_interaction_const - best_converge_interaction_const))
            
            if prob >= 1 or prob >= np.random.rand():
                best_converge_interaction_const = current_converge_interaction_const
                self.current_adjacency_matrix = new_adjacency_matrix
                self.current_freqs = new_freqs
                
                
        return (self.current_adjacency_matrix, self.current_freqs, order_parameters_history, converge_interaction_consts_history)
        