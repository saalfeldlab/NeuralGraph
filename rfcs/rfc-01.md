This is a repo of neural network models implemented in pytorch.
We would like to jointly develop models & apply them to diverse datasets.
For example, in this repo, we have graph neural network models, and these are
applied to simulations based on FlyVis. We would like these models to be applied
to future zebrafish data, or perhaps, other types. To the extent possible we
would like to share the basic model code.
However, we are not opposed to code duplication where it helps avoid
complicating the interface of general purpose modules.

- Keep the model PYTHONPATH "src/"

## General purpose modules

General model config definitions should live along with the code that utilizes them. For example,

```
class MLPConfig(BaseModel):
    num_input_dims: int
    ...

class MLP(torch.nn.Module):
    ...
```

This code is intended to be importable by other models. These library code files would live in src/models/...
And these models will only import other code in src/models/...

A model should never bake in a parameter that is specific to an application. This is an anti-pattern

```
from ZapBench import ZapModule

class GeneralModule(torch.nn.Module):
    def __init__(...):
        self.layer = ZapModule(...)
```

This will lead to circular imports.

Modules in models/ will have no functionality for training. These are meant to be imported.
The MLP configs for example do not specify the number of epochs.

## Models trained for specific applications

Structure:

```
src
|- models
   |- model1.py
   |- model2.py
   |- ...
|- flyvis
   |- simulation # this defines the data source here
      |- configs
         |- config_baseline.yaml * always points to canonical config to use
         |- config1_expt1.yaml
         |- config2_expt1.yaml
         |- ...
         |- config1_expt2.yaml
         |- ...
         |- test.py # ensure that configs are ok, unique names etc.
      |- method1.py
         |- defines Method1Config(BaseModel)...
         |- defines Method1Module(torch.nn.Module)...
         |- defines def train(...)...

   |- graph
      |- configs # specifies a data source which uniquely specifies a config name in ../simulation/configs
         |- config_baseline.yaml # (*) always points to canonical config to use
         |- config1_expt1.yaml
         |- config2_expt1.yaml
         |- ...
         |- config1_expt2.yaml
         |- ...
      |- method1.py
         |- defines Method1Config(BaseModel)...
         |- defines Method1Module(torch.nn.Module)...

   |- rnn
   |- mlp
   |- ...
|- zapbench
   |- graph
   |- siren
   |- instant_ngp
   |- ...
|- devbio
   |- simulation
   |- graph
   |- ...
```

We will use the components defined in the models/ module to build networks that will be trained.

`method1.py` is executable
