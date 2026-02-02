# Synchronization phenomenon of phase dynamics in the Kuramoto model on graphs

This project consists of four logical parts:
1. Fast and efficient [Kuramoto model](https://en.wikipedia.org/wiki/Kuramoto_model) simulation toolkit on arbitrary graphs with Numba CUDA-jitter.  
2. Set of auxiliary functions that could be used to generate graphs from different families with specified properties.  
3. Study of the [phase transition](https://en.wikipedia.org/wiki/Phase_transition), which could be measured by the rapid change of the order parameter, from a chaotic to a synchronized state depending on the coupling constant in the Kuramoto model for different types of graphs.    
4. Study of the possibility of iterative evolution of the graph adjacency matrix using the [thermal annealing](https://en.wikipedia.org/wiki/Simulated_annealing) method in the direction of the order parameter increase, which is responsible for [synchronization](https://en.wikipedia.org/wiki/Synchronization_of_chaos) enhancement.

Some function annotations and specifications could be found in source files.

**kuramoto_model.py** - main framework file with Kuramoto model implementation (the first part),  
**utils.py** - a bag with graph generation methods (the second part),  
**Synchronization.ipynb** - Jupyter notebooks with phase transition detection experiments (the third part),  
**Graph Evolution.ipynb** - Jupyter notebooks with graph adjacency matrix evolution experiments (the fourth part).

*Nota Bene 1*: The project is still in the development stage, so the code often contains many auxiliary constructs and comments on possible improvements.

*Nota Bene 2*: To-do lists and comments are partially written in Russian.
