# NeuralGraph
Graph network modeling neural activities

<p align="center">
  <img src="./assets/Fig1.png" alt="NeuralGraph Overview" width="600">
</p>
<p align="center"><b>Figure 1:</b> The temporal activity of a simulated neural network (a) is converted into densely connected graph (b) processed by a message passing GNN (c). Each neuron (node i) receives activity signals from connected neurons (node j), processed by a transfer function and weighted by the connection matrix. The sum of these messages is updated to obtain the predicted activity rate. In addition to the observed activity, the GNN has access to learnable latent vectors associated with each node.</p>



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
