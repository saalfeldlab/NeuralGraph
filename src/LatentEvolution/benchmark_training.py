"""
minimal benchmark for training loop timing.
runs 1 epoch of training with detailed timing breakdown.
"""

import time
import random
import numpy as np
import torch
import yaml
from pathlib import Path

from LatentEvolution.load_flyvis import FlyVisSim
from LatentEvolution.latent import LossType, ModelParams, LatentModel, train_step, train_step_nocompile
from LatentEvolution.acquisition import compute_neuron_phases, sample_batch_indices
from LatentEvolution.load_flyvis import load_column_slice


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def benchmark_epoch(cfg: ModelParams, warmup_batches: int = 4, compile_mode: str = "default",
                    use_amp: bool = False, use_fused_adam: bool = False,
                    compile_backward: bool = False, compile_optimizer: bool = False):
    """
    run 1 epoch with detailed timing breakdown.

    compile_mode options:
        - "none": no compilation
        - "default": torch.compile with default settings
        - "reduce-overhead": torch.compile with reduce-overhead mode
    """
    seed_everything(cfg.training.seed)
    device = torch.device("cuda")
    print(f"using cuda: {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision("high")

    # model
    model = LatentModel(cfg).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.train()

    # optimizer
    if use_fused_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, fused=True)
        print("using fused adam")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # compile optimizer step if requested
    if compile_optimizer:
        optimizer.step = torch.compile(optimizer.step, mode="reduce-overhead")
        print("compiled optimizer.step")

    # select train_step function
    if compile_mode == "none":
        train_step_fn = train_step_nocompile
        print("using non-compiled train_step")
    elif compile_mode == "reduce-overhead":
        train_step_fn = torch.compile(train_step_nocompile, fullgraph=True, mode="reduce-overhead")
        print("using reduce-overhead compiled train_step")
    elif compile_mode == "max-autotune":
        train_step_fn = torch.compile(train_step_nocompile, fullgraph=True, mode="max-autotune")
        print("using max-autotune compiled train_step")
    else:
        train_step_fn = train_step
        print("using default compiled train_step")

    if compile_backward:
        print("using compiled autograd (backward)")

    if use_amp:
        print("using automatic mixed precision (amp)")

    # data - load single 16K chunk directly to GPU
    chunk_size = 65536
    data_path = f"/groups/saalfeld/home/kumarv4/repos/NeuralGraph/graphs_data/fly/{cfg.training.simulation_config}/x_list_0"
    column_idx = FlyVisSim[cfg.training.column_to_model].value

    train_start = cfg.training.data_split.train_start
    chunk_data = torch.from_numpy(
        load_column_slice(data_path, column_idx, train_start, train_start + chunk_size)
    ).to(device)
    chunk_stim = torch.from_numpy(
        load_column_slice(data_path, FlyVisSim.STIMULUS.value, train_start, train_start + chunk_size,
                          neuron_limit=cfg.stimulus_encoder_params.num_input_dims)
    ).to(device)
    print(f"loaded chunk: {chunk_data.shape}")

    # batches per epoch from single chunk
    batches_per_epoch = chunk_size // cfg.training.batch_size
    print(f"batches per epoch: {batches_per_epoch}")

    # acquisition mode
    total_steps = cfg.training.time_units * cfg.training.evolve_multiple_steps
    neuron_phases = compute_neuron_phases(
        num_neurons=cfg.num_neurons,
        time_units=cfg.training.time_units,
        acquisition_mode=cfg.training.acquisition_mode,
        device=device,
    )

    # pre-allocate empty tensors for augmentation (not used with default config)
    selected_neurons = torch.empty(0, dtype=torch.long, device=device)
    needed_indices = torch.empty(0, dtype=torch.long, device=device)

    # amp scaler
    scaler = torch.amp.GradScaler() if use_amp else None

    # enable compiled autograd if requested
    if compile_backward:
        torch._dynamo.config.compiled_autograd = True

    # warmup to trigger compilation
    print(f"warmup ({warmup_batches} batches to trigger torch.compile)...")
    for _ in range(warmup_batches):
        optimizer.zero_grad()
        observation_indices = sample_batch_indices(
            chunk_size=chunk_data.shape[0],
            total_steps=total_steps,
            time_units=cfg.training.time_units,
            batch_size=cfg.training.batch_size,
            num_neurons=cfg.num_neurons,
            neuron_phases=neuron_phases,
            device=device,
        )
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss_tuple = train_step_fn(
                    model, chunk_data, chunk_stim, observation_indices,
                    selected_neurons, needed_indices, cfg
                )
            scaler.scale(loss_tuple[0]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_tuple = train_step_fn(
                model, chunk_data, chunk_stim, observation_indices,
                selected_neurons, needed_indices, cfg
            )
            loss_tuple[LossType.TOTAL].backward()
            optimizer.step()

    torch.cuda.synchronize()
    print("warmup complete")

    # main epoch
    print("running 1 epoch...")
    epoch_start = time.perf_counter()

    for _ in range(batches_per_epoch):
        optimizer.zero_grad()

        observation_indices = sample_batch_indices(
            chunk_size=chunk_data.shape[0],
            total_steps=total_steps,
            time_units=cfg.training.time_units,
            batch_size=cfg.training.batch_size,
            num_neurons=cfg.num_neurons,
            neuron_phases=neuron_phases,
            device=device,
        )

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss_tuple = train_step_fn(
                    model, chunk_data, chunk_stim, observation_indices,
                    selected_neurons, needed_indices, cfg
                )
            scaler.scale(loss_tuple[0]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_tuple = train_step_fn(
                model, chunk_data, chunk_stim, observation_indices,
                selected_neurons, needed_indices, cfg
            )
            loss_tuple[LossType.TOTAL].backward()
            optimizer.step()

    torch.cuda.synchronize()
    epoch_duration = time.perf_counter() - epoch_start

    # results
    print("\n=== benchmark results ===")
    print(f"epoch duration: {epoch_duration:.2f}s")
    print(f"batches: {batches_per_epoch}")
    print(f"avg batch time: {epoch_duration/batches_per_epoch*1000:.2f}ms")

    return epoch_duration


if __name__ == "__main__":
    import sys

    config_path = Path(__file__).resolve().parent / "latent_20step.yaml"
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # set acquisition mode to time_aligned
    data["training"]["acquisition_mode"] = {"mode": "time_aligned"}

    cfg = ModelParams(**data)

    # parse args: mode [--amp] [--fused] [--compile-backward] [--compile-opt]
    args = sys.argv[1:]
    mode = "default"
    use_amp = False
    use_fused = False
    compile_backward = False
    compile_opt = False

    for arg in args:
        if arg == "--amp":
            use_amp = True
        elif arg == "--fused":
            use_fused = True
        elif arg == "--compile-backward":
            compile_backward = True
        elif arg == "--compile-opt":
            compile_opt = True
        elif not arg.startswith("--"):
            mode = arg

    desc = f"{mode}"
    if use_amp:
        desc += "+amp"
    if use_fused:
        desc += "+fused"
    if compile_backward:
        desc += "+bwd"
    if compile_opt:
        desc += "+opt"

    print(f"\n{'='*50}")
    print(f"test: {desc}")
    print(f"{'='*50}")
    t = benchmark_epoch(cfg, compile_mode=mode, use_amp=use_amp, use_fused_adam=use_fused,
                        compile_backward=compile_backward, compile_optimizer=compile_opt)
    print(f"\nRESULT: {desc} = {t:.2f}s ({t/64*1000:.2f}ms/batch)")
