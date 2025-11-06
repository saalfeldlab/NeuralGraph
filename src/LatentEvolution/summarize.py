import glob
from pathlib import Path
import polars as pl
import yaml
from LatentEvolution.latent import ModelParams
from typing import Any, Dict, List, MutableMapping, Tuple

def flatten_dict(
    d: MutableMapping[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# parser = argparse.ArgumentParser()
# parser.add_argument("expt_code")
# args = parser.parse_args()
# expt_code = args.expt_code

expt_code = "latent_dim_sweep"

run_dir_base = Path("/groups/saalfeld/home/kumarv4/repos/NeuralGraph/runs") / expt_code
assert run_dir_base.exists()


run_dirs = [Path(p) for p in glob.glob(f"{run_dir_base}/*")]
configs = []
metrics = []
metric_keys = None
for run_dir in run_dirs:
    with open(run_dir / "config.yaml") as fin:
        raw = yaml.safe_load(fin)
    _config = ModelParams.model_validate(raw)
    raw_flat = flatten_dict(raw)
    configs.append(raw_flat)

    metrics_file = run_dir / "final_metrics.yaml"
    if not metrics_file.exists():
        metrics.append({})
        continue
    with open(metrics_file) as fin:
        raw = yaml.safe_load(fin)
        metrics.append(raw)
        if metric_keys is None:
            metric_keys = sorted(raw.keys())
        else:
            assert metric_keys == sorted(raw.keys())

assert metric_keys is not None, "All runs failed"

config_df = pl.DataFrame(configs)
config_cols = []
for col in config_df.columns:
    if config_df[col].unique().shape[0] > 1:
        config_cols.append(col)
config_cols.sort(key=lambda k: k.count("."))
config_df = config_df.sort(config_cols)

metrics_df = pl.DataFrame(metrics, schema=metric_keys)

df = pl.concat([config_df, metrics_df], how="horizontal")

commit_hash = metrics_df["commit_hash"][0]
print(f"{commit_hash=}")
with pl.Config(tbl_cols=30, tbl_rows=100, tbl_width_chars=3000, tbl_hide_column_data_types=True):
    print(df.sort(["final_val_loss"]).select(config_cols + metric_keys))
    print("Sorted by validation loss: ", df.sort("final_val_loss").select(config_cols + ["final_train_loss", "final_val_loss", "final_test_loss"]))