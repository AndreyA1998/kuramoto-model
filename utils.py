# All generators in this section provide np.array output with shape [num_nodes, num_nodes]

import numpy as np

# generating full-connected non-oriented graph with equal weights
def gen_full_connected_graph(num_nodes, weight=1):
    return np.full((num_nodes, num_nodes), weight) - np.eye(num_nodes) * weight
 
 
# generating full-connected non-oriented graph with random uniform weights from the interval [a, b]
def gen_full_connected_graph_random_weights(num_nodes, weight_interval=(1, 2), integer=True):
    res = np.full((num_nodes, num_nodes), 1) - np.eye(num_nodes)
    gen_rand = np.random.uniform(weight_interval[0], weight_interval[1], (num_nodes, num_nodes))
    res = (res * gen_rand + res * gen_rand.T) / 2
    if integer is True:
        return np.around(res)
    else:
        return res


# generating linear non-oriented graph with equal weights
def gen_linear_graph(num_nodes, weight=1):
    res = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes - 1):
        res[i, i + 1] = weight
        res[i + 1, i] = weight
    return res


# generating linear non-oriented graph with random uniform weights from the interval [a, b]
def gen_linear_graph_random_weights(num_nodes, weight_interval=(1, 2), integer=True):
    res = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes - 1):
        gen_rand = (weight_interval[1] - weight_interval[0]) * np.random.rand() + weight_interval[0]
        if integer is True:
            res[i, i + 1] = round(gen_rand)
            res[i + 1, i] = round(gen_rand)
        else: 
            res[i, i + 1] = gen_rand
            res[i + 1, i] = gen_rand
    return res


# generating random non-directed graph with equal weights
def gen_poisson_graph(num_nodes, prob=0.5, weight=1):
    gen_rand = np.random.rand(num_nodes, num_nodes)
    random = (gen_rand + gen_rand.T) / 2
    res = random * (np.ones((num_nodes, num_nodes)) - np.eye(num_nodes))
    res = np.where(res >= prob, weight, 0)
    return res
    

# generating random non-directed graph with random uniform weights from the interval [a, b]
def gen_poisson_graph_random_weights(num_nodes, weight_interval=(1, 2), integer=True):
    gen_rand = np.random.uniform(weight_interval[0], weight_interval[1], (num_nodes, num_nodes))
    random = (gen_rand + gen_rand.T) / 2
    res = random * (np.ones((num_nodes, num_nodes)) - np.eye(num_nodes))
    if integer is True:
        return np.around(res)
    else:
        return res


# generating star non-oriented graph with odd or even decorated weights
def gen_decorated_star(rays_num, rays_length, decorated_rays_num=1, decorate_even=True):
    # numeration of num_nodes starts from center and go through each ray one by one
    # decorate_even - put decorated_rays_num on even (odd if False) connections instead of 1
    graph_size = rays_num * rays_length
    graph = np.zeros((graph_size + 1, graph_size + 1))
    
    graph[1::rays_length, 0] = 1 if decorate_even else decorated_rays_num

    step = 0 if decorate_even else 1
    
    for i in range(1, graph_size):
        if i % rays_length != 0:
            graph[i + 1, i] = 1 if (i + step) % 2 == 0 else decorated_rays_num

    return graph + graph.T


# generating direct (oriented) acyclic graph with random uniform weights from the interval [a, b]
# WARNING! in this method one could obtain not one-componentd graph
def gen_direct_acyclic_graph(num_nodes_in_layers, connections_per_node, weight=1):
    if len(num_nodes_in_layers) != len(connections_per_node) + 1:
        raise ValueError
    num_nodes = num_nodes_in_layers.sum()
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    num_previous_nodes = np.cumsum(num_nodes_in_layers) - num_nodes_in_layers
    for i, num_nodes in enumerate(num_previous_nodes[1:]):
        previous_nodes = np.arange(num_nodes)
        current_nodes = np.arange(num_nodes, num_nodes + num_nodes_in_layers[i + 1])
        possible_connections = np.array(np.meshgrid(previous_nodes, current_nodes)).T.reshape(-1, 2)

        idx = np.random.choice(
            np.arange(possible_connections.shape[0]),
            connections_per_node[i] * num_nodes_in_layers[i],  # [i + 1]?
            replace=False
        )
        for connection in possible_connections[idx]:
            adjacency_matrix[connection[0], connection[1]] = weight

    return adjacency_matrix


# generating direct (oriented) acyclic graph with equal weights
# WARNING! in this method one could obtain not one-componentd graph
def gen_direct_acyclic_graph_random_weights(num_nodes_in_layers, connections_per_node, weight_interval=(1, 2), integer=True):
    if len(num_nodes_in_layers) != len(connections_per_node) + 1:
        raise ValueError
    num_nodes = num_nodes_in_layers.sum()
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    num_previous_nodes = np.cumsum(num_nodes_in_layers) - num_nodes_in_layers
    for i, num_nodes in enumerate(num_previous_nodes[1:]):
        previous_nodes = np.arange(num_nodes)
        current_nodes = np.arange(num_nodes, num_nodes + num_nodes_in_layers[i + 1])
        possible_connections = np.array(np.meshgrid(previous_nodes, current_nodes)).T.reshape(-1, 2)

        idx = np.random.choice(
            np.arange(possible_connections.shape[0]),
            connections_per_node[i] * num_nodes_in_layers[i],  # [i + 1]?
            replace=False
        )
        for connection in possible_connections[idx]:
            res = (weight_interval[1] - weight_interval[0]) * np.random.rand() + weight_interval[0]
            if integer is True:
                res = round(res)
            adjacency_matrix[connection[0], connection[1]] = res

    return adjacency_matrix


# generating direct (oriented) acyclic graph with weak_connections
# WARNING! in this method one could obtain not one-componentd graph
def gen_direct_acyclic_graph_weak_connections(num_nodes_in_layers, connections_per_node, weight_interval=(1, 2), integer=True):
    if len(num_nodes_in_layers) != len(connections_per_node) + 1:
        raise ValueError
    num_nodes = num_nodes_in_layers.sum()
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    num_previous_nodes = np.cumsum(num_nodes_in_layers) - num_nodes_in_layers
    for i, num_nodes in enumerate(num_previous_nodes[1:]):
        previous_nodes = np.arange(num_nodes)
        current_nodes = np.arange(num_nodes, num_nodes + num_nodes_in_layers[i + 1])
        possible_connections = np.array(np.meshgrid(previous_nodes, current_nodes)).T.reshape(-1, 2)

        idx = np.random.choice(
            np.arange(possible_connections.shape[0]),
            connections_per_node[i] * num_nodes_in_layers[i],  # [i + 1]?
            replace=False
        )
        for connection in possible_connections[idx]:
            if connection[1] - connection[0] > num_nodes_in_layers[i + 1]:
                adjacency_matrix[connection[0], connection[1]] = weight_interval[0]
            else:
                adjacency_matrix[connection[0], connection[1]] = weight_interval[1]

    return adjacency_matrix 
