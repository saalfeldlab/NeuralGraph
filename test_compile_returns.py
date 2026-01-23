"""Test torch.compile compatibility with different return types."""
import torch
import time
import numpy as np
from enum import Enum, auto
from typing import NamedTuple, Dict


class LossType(Enum):
    """Loss component types."""
    TOTAL = auto()
    RECON = auto()
    EVOLVE = auto()


# NamedTuple version
class LossDict(NamedTuple):
    total: torch.Tensor
    recon: torch.Tensor
    evolve: torch.Tensor


# Test functions - simulate realistic training step
def train_step_tuple(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> tuple:
    """Return regular tuple."""
    # Simulate encoder/decoder operations
    h1 = torch.relu(x @ w1)  # hidden layer
    out = h1 @ w2  # output

    # Multiple loss computations
    loss1 = ((out - x) ** 2).mean()  # reconstruction
    loss2 = torch.abs(h1).mean()  # l1 reg
    loss3 = ((out[1:] - out[:-1]) ** 2).mean()  # temporal smoothness
    total = loss1 + 0.1 * loss2 + 0.01 * loss3
    return (total, loss1, loss2, loss3)


def train_step_namedtuple(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> LossDict:
    """Return NamedTuple."""
    h1 = torch.relu(x @ w1)
    out = h1 @ w2

    loss1 = ((out - x) ** 2).mean()
    loss2 = torch.abs(h1).mean()
    loss3 = ((out[1:] - out[:-1]) ** 2).mean()
    total = loss1 + 0.1 * loss2 + 0.01 * loss3
    return LossDict(total=total, recon=loss1, evolve=loss3)


def train_step_dict_literal(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> dict:
    """Return dict with literal {} syntax."""
    h1 = torch.relu(x @ w1)
    out = h1 @ w2

    loss1 = ((out - x) ** 2).mean()
    loss2 = torch.abs(h1).mean()
    loss3 = ((out[1:] - out[:-1]) ** 2).mean()
    total = loss1 + 0.1 * loss2 + 0.01 * loss3
    return {"total": total, "recon": loss1, "evolve": loss3}


def train_step_dict_enum_keys(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> Dict[LossType, torch.Tensor]:
    """Return dict with enum keys."""
    h1 = torch.relu(x @ w1)
    out = h1 @ w2

    loss1 = ((out - x) ** 2).mean()
    loss2 = torch.abs(h1).mean()
    loss3 = ((out[1:] - out[:-1]) ** 2).mean()
    total = loss1 + 0.1 * loss2 + 0.01 * loss3
    return {LossType.TOTAL: total, LossType.RECON: loss1, LossType.EVOLVE: loss3}


def benchmark(fn, args, name, num_iters=1000, num_trials=10):
    """Benchmark a function with error bars."""
    # Warmup
    for _ in range(10):
        result = fn(*args)

    # Multiple trials
    trial_times = []
    for _ in range(num_trials):
        if args[0].is_cuda:
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            result = fn(*args)
        if args[0].is_cuda:
            torch.cuda.synchronize()
        elapsed = time.time() - start
        trial_times.append(elapsed)

    # Statistics
    mean_time = np.mean(trial_times)
    std_time = np.std(trial_times)
    mean_us = mean_time * 1000000 / num_iters
    std_us = std_time * 1000000 / num_iters

    print(f"{name:30s}: {mean_us:.2f} ± {std_us:.2f} µs/iter")
    return result, mean_us, std_us


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}\n")

    # Create test data - simulate batch of neural activity
    batch_size = 256
    num_neurons = 1000
    latent_dim = 256

    x = torch.randn(batch_size, num_neurons, device=device)
    w1 = torch.randn(num_neurons, latent_dim, device=device) / (num_neurons ** 0.5)
    w2 = torch.randn(latent_dim, num_neurons, device=device) / (latent_dim ** 0.5)
    args = (x, w1, w2)

    print("=" * 70)
    print("COMPILATION TEST (checking if each compiles without error)")
    print("=" * 70)

    # Test 1: Regular tuple
    try:
        compiled_tuple = torch.compile(train_step_tuple, fullgraph=True, mode="reduce-overhead")
        result = compiled_tuple(*args)
        print(f"✓ Regular tuple: SUCCESS - returns {type(result)}")
        print(f"  Values: {[f'{v.item():.4f}' for v in result]}")
    except Exception as e:
        print(f"✗ Regular tuple: FAILED - {e}")

    # Test 2: NamedTuple
    try:
        compiled_namedtuple = torch.compile(train_step_namedtuple, fullgraph=True, mode="reduce-overhead")
        result = compiled_namedtuple(*args)
        print(f"✓ NamedTuple: SUCCESS - returns {type(result)}")
        print(f"  Values: total={result.total.item():.4f}, recon={result.recon.item():.4f}, evolve={result.evolve.item():.4f}")
        print(f"  Can access by name: result.total = {result.total.item():.4f}")
    except Exception as e:
        print(f"✗ NamedTuple: FAILED - {e}")

    # Test 3: Dict with string keys (literal {})
    try:
        compiled_dict_literal = torch.compile(train_step_dict_literal, fullgraph=True, mode="reduce-overhead")
        result = compiled_dict_literal(*args)
        print(f"✓ Dict (string keys): SUCCESS - returns {type(result)}")
        print(f"  Values: {[(k, f'{v.item():.4f}') for k, v in result.items()]}")
    except Exception as e:
        print(f"✗ Dict (string keys): FAILED - {e}")

    # Test 4: Dict with enum keys
    try:
        compiled_dict_enum = torch.compile(train_step_dict_enum_keys, fullgraph=True, mode="reduce-overhead")
        result = compiled_dict_enum(*args)
        print(f"✓ Dict (enum keys): SUCCESS - returns {type(result)}")
        print(f"  Values: {[(k, f'{v.item():.4f}') for k, v in result.items()]}")
    except Exception as e:
        print(f"✗ Dict (enum keys): FAILED - {e}")

    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK (mean ± std over 10 trials)")
    print("=" * 70)

    # Benchmark each version
    results = {}

    print("\nUncompiled:")
    _, results['tuple_uncompiled'], _ = benchmark(train_step_tuple, args, "  Tuple")
    _, results['namedtuple_uncompiled'], _ = benchmark(train_step_namedtuple, args, "  NamedTuple")
    _, results['dict_str_uncompiled'], _ = benchmark(train_step_dict_literal, args, "  Dict (string keys)")
    _, results['dict_enum_uncompiled'], _ = benchmark(train_step_dict_enum_keys, args, "  Dict (enum keys)")

    print("\nCompiled (reduce-overhead):")
    _, results['tuple_compiled'], _ = benchmark(compiled_tuple, args, "  Tuple")
    _, results['namedtuple_compiled'], _ = benchmark(compiled_namedtuple, args, "  NamedTuple")
    _, results['dict_str_compiled'], _ = benchmark(compiled_dict_literal, args, "  Dict (string keys)")
    _, results['dict_enum_compiled'], _ = benchmark(compiled_dict_enum, args, "  Dict (enum keys)")

    print("\n" + "=" * 70)
    print("COMPILATION SPEEDUP (compiled vs uncompiled)")
    print("=" * 70)
    print(f"  Tuple:              {results['tuple_uncompiled']/results['tuple_compiled']:.2f}x faster")
    print(f"  NamedTuple:         {results['namedtuple_uncompiled']/results['namedtuple_compiled']:.2f}x faster")
    print(f"  Dict (string keys): {results['dict_str_uncompiled']/results['dict_str_compiled']:.2f}x faster")
    print(f"  Dict (enum keys):   {results['dict_enum_uncompiled']/results['dict_enum_compiled']:.2f}x faster")

    print("\n" + "=" * 70)
    print("SPEEDUP vs TUPLE (compiled)")
    print("=" * 70)
    baseline = results['tuple_compiled']
    print("  Tuple:              1.00x (baseline)")
    print(f"  NamedTuple:         {baseline/results['namedtuple_compiled']:.2f}x")
    print(f"  Dict (string keys): {baseline/results['dict_str_compiled']:.2f}x")
    print(f"  Dict (enum keys):   {baseline/results['dict_enum_compiled']:.2f}x")

    print("\n" + "=" * 70)
    print("OVERHEAD vs TUPLE (compiled)")
    print("=" * 70)
    print(f"  NamedTuple:         {(results['namedtuple_compiled']/baseline - 1)*100:+.1f}%")
    print(f"  Dict (string keys): {(results['dict_str_compiled']/baseline - 1)*100:+.1f}%")
    print(f"  Dict (enum keys):   {(results['dict_enum_compiled']/baseline - 1)*100:+.1f}%")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    overhead_namedtuple = (results['namedtuple_compiled']/baseline - 1)*100
    if overhead_namedtuple < 20:
        print(f"✓ NamedTuple has only {overhead_namedtuple:.1f}% overhead → RECOMMENDED")
        print("  Benefits: semantic access (result.recon), type safety, immutable")
    else:
        print(f"✗ NamedTuple has {overhead_namedtuple:.1f}% overhead → Use regular tuple")
    print("\nFor this codebase: NamedTuple with field names matching LossType enum")
