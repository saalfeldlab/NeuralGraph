import argparse
from pathlib import Path
import polars as pl
import yaml
from LatentEvolution.latent import ModelParams

parser = argparse.ArgumentParser()
parser.add_argument("expt_code")
args = parser.parse_args()
expt_code = args.expt_code

runs_base = Path("/groups/saalfeld/home/kumarv4/repos/NeuralGraph/runs")

# Find all experiment directories matching the expt_code
# New structure: runs/expt_code_YYYYMMDD_hash/param1/param2/.../uuid/
# Old structure: runs/expt_code/uuid/
expt_dirs = list(runs_base.glob(f"{expt_code}_*"))
if not expt_dirs:
    # Fallback to old flat structure
    expt_dirs = [runs_base / expt_code]

assert expt_dirs, f"No experiment directories found for {expt_code}"

# Find all run directories (directories containing config.yaml)
run_dirs = []
for expt_dir in expt_dirs:
    if not expt_dir.exists():
        continue
    # Recursively find all directories containing config.yaml
    for config_file in expt_dir.rglob("config.yaml"):
        run_dirs.append(config_file.parent)

assert run_dirs, f"No run directories found in {expt_dirs}"

configs = []
metrics = []
metric_keys = None
for run_dir in run_dirs:
    with open(run_dir / "config.yaml") as fin:
        raw = yaml.safe_load(fin)
    config = ModelParams.model_validate(raw)
    raw_flat = config.flatten()

    metrics_file = run_dir / "final_metrics.yaml"
    if not metrics_file.exists():
        metrics.append({})
        continue
    configs.append(raw_flat)
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

metrics_df = pl.DataFrame(metrics, schema=metric_keys)

df = pl.concat([config_df, metrics_df], how="horizontal")

commit_hash = metrics_df["commit_hash"][0]
print(f"{commit_hash=}")
with pl.Config(tbl_cols=30, tbl_rows=100, tbl_width_chars=3000, tbl_hide_column_data_types=True):
    print(df.sort(["final_val_loss"]).select(config_cols + metric_keys))
    print("Sorted by validation loss: ", df.sort("final_val_loss").select(config_cols + ["final_train_loss", "final_val_loss", "final_test_loss"]))