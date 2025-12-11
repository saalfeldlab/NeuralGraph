# NeuralGraph Project Structure

## Key Directories

- `src/NeuralGraph/generators/` - PDE models (PDE_N2.py through PDE_N11.py) and data generation
- `src/NeuralGraph/models/` - Training code (graph_trainer.py) and neural network models
- `config/signal/` - Signal experiment configs (YAML files)
- `config/fly/` - FlyVis experiment configs

## Important Files for Signal Processing

- `src/NeuralGraph/generators/graph_data_generator.py` - `data_generate_synaptic()` function for signal data generation
- `src/NeuralGraph/models/graph_trainer.py` - `data_train_signal()` function for training
- `src/NeuralGraph/config.py` - Config dataclasses including `input_type`, `oscillation_*`, `triggered_*` params

## FlyVis Integration

- `data_generate_fly_voltage()` in graph_data_generator.py - Uses `visual_input_type` for external inputs
- `data_train_flyvis()` in graph_trainer.py - Training for fly models

## Data Flow

1. Config YAML -> `config.py` dataclasses
2. `graph_data_generator.py` calls appropriate `data_generate_*()` based on config
3. PDE_N\* model simulates dynamics with external inputs
4. `graph_trainer.py` trains on generated data
