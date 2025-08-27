# NeuralGraph
Graph network modeling neural activities 

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
