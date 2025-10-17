# NeuralGraph
Graph network modeling neural activities

![NeuralGraph Overview](./assets/Fig1.png)
*The temporal activity of a simulated neural network (**a**) is converted into densely connected graph (**b**) processed by a message passing GNN (**c**). Each neuron (node i) receives activity signals x_j from connected neurons (node j), processed by a transfer function ψ\* and weighted by the matrix W_ij. The sum of these messages is updated with functions φ\* and Ω\* to obtain the predicted activity rate ẋ̂_i. In addition to the observed activity x_i, the GNN has access to learnable latent vectors a_i associated with each node i.*


### Setup
Run the following line from the terminal to create a new environment particle-graph:
```
conda env create -f environment.yaml
```

Activate the environment:
```
conda activate neural-graph
```

Install the package by executing the following command from the root of this directory:
```
pip install -e .
```

Download the pretrained FlyVis models by running:
```
flyvis download-pretrained
```


Then, you should be able to import all the modules from the package in python:
```python
from NeuralGraph import *
```
