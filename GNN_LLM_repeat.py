import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import os
import shutil
import subprocess
import time
import yaml

# redirect PyTorch JIT cache to /scratch instead of /tmp (per IT request)
if os.path.isdir('/scratch'):
    os.environ['TMPDIR'] = '/scratch/allierc'
    os.makedirs('/scratch/allierc', exist_ok=True)

from NeuralGraph.config import NeuralGraphConfig
from NeuralGraph.generators.graph_data_generator import data_generate
from NeuralGraph.models.graph_trainer import data_train, data_test
from NeuralGraph.models.exploration_tree import compute_ucb_scores
from NeuralGraph.models.utils import save_exploration_artifacts
from NeuralGraph.utils import set_device, add_pre_folder
from GNN_PlotFigure import data_plot

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ──────────────────────────────────────────────────────────
N_REPEATS = 10
ITERATION = 1             # start from scratch
N_ITERATIONS = 2048       # total iterations (for prompt display)

llm_task_name = 'signal_landscape_Claude_repeat'
base_config_name = 'signal_landscape'
instruction_name = f'instruction_{base_config_name}'

root_dir = os.path.dirname(os.path.abspath(__file__))
config_root = root_dir + "/config"

# ── Load config ────────────────────────────────────────────────────────────
config_file_ = llm_task_name
config_file, pre_folder = add_pre_folder(config_file_)
config = NeuralGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
config.dataset = pre_folder + config.dataset
config.config_file = pre_folder + config_file_

device = set_device(config.training.device)

# Read claude parameters from config
config_path = f"{config_root}/{pre_folder}{config_file_}.yaml"
with open(config_path, 'r') as f:
    raw_config = yaml.safe_load(f)
claude_cfg = raw_config.get('claude', {})
n_iter_block = claude_cfg.get('n_iter_block', 24)
ucb_c = claude_cfg.get('ucb_c', 1.414)

# Compute block info for iteration
block_number = (ITERATION - 1) // n_iter_block + 1
iter_in_block = (ITERATION - 1) % n_iter_block + 1
is_block_end = iter_in_block == n_iter_block

# ── Paths ──────────────────────────────────────────────────────────────────
analysis_log_path = f"{root_dir}/{llm_task_name}_analysis.log"
instruction_path = f"{root_dir}/{instruction_name}_repeat.md"
analysis_path = f"{root_dir}/{llm_task_name}_analysis.md"
memory_path = f"{root_dir}/{llm_task_name}_memory.md"
ucb_path = f"{root_dir}/{llm_task_name}_ucb_scores.txt"

# Repeat output files
analysis_repeat_path = f"{root_dir}/{llm_task_name}_analysis_repeat.md"
memory_repeat_path = f"{root_dir}/{llm_task_name}_memory_repeat.md"
reasoning_repeat_path = f"{root_dir}/{llm_task_name}_reasoning_repeat.log"

# ── Phase 1: Run full pipeline — generate + train + test + plot (once) ────
print(f"\n\033[94m{'='*60}\033[0m")
print("\033[94mRepeat Reasoning Script — From Scratch\033[0m")
print(f"\033[94mConfig: {config_file_}\033[0m")
print(f"\033[94mIteration {ITERATION}, Block {block_number}, iter {iter_in_block}/{n_iter_block}\033[0m")
print(f"\033[94m{'='*60}\033[0m\n")

# Open analysis.log for metrics
log_file = open(analysis_log_path, 'w')

print("\033[93mRunning data_generate...\033[0m")
data_generate(
    config,
    device=device,
    visualize=False,
    run_vizualized=0,
    style="black color",
    alpha=1,
    erase=True,
    bSave=True,
    step=2,
    log_file=log_file
)

print("\033[93mRunning data_train...\033[0m")
data_train(
    config=config,
    erase=True,
    best_model=None,
    style='color',
    device=device,
    log_file=log_file
)

print("\033[93mRunning data_test...\033[0m")
config.simulation.noise_model_level = 0.0
data_test(
    config=config,
    visualize=False,
    style="black name continuous_slice",
    verbose=False,
    best_model='best',
    run=0,
    test_mode="",
    sample_embedding=False,
    step=10,
    n_rollout_frames=1000,
    device=device,
    particle_of_interest=0,
    new_params=None,
    log_file=log_file,
)

print("\033[93mRunning data_plot...\033[0m")
folder_name = './log/' + pre_folder + '/tmp_results/'
os.makedirs(folder_name, exist_ok=True)
data_plot(config=config, config_file=config_file, epoch_list=['best'],
          style='black color', extended='plots', device=device,
          apply_weight_correction=True, log_file=log_file)

log_file.close()
print(f"\033[92mMetrics written to: {analysis_log_path}\033[0m")

# Save exploration artifacts (activity image)
exploration_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}"
artifact_paths = save_exploration_artifacts(
    root_dir, exploration_dir, config, config_file_, pre_folder, ITERATION,
    iter_in_block=iter_in_block, block_number=block_number
)
activity_path = artifact_paths['activity_path']

# Initialize analysis.md and memory.md for this fresh run
with open(analysis_path, 'w') as f:
    f.write(f"# Experiment Log: {config_file_}\n\n")
print(f"\033[93mInitialized: {analysis_path}\033[0m")

with open(memory_path, 'w') as f:
    f.write(f"# Working Memory: {config_file_}\n\n")
    f.write("## Knowledge Base (accumulated across all blocks)\n\n")
    f.write("### Regime Comparison Table\n")
    f.write("| Block | Regime | E/I | n_frames | n_neurons | n_types | eff_rank | Best R² | Optimal lr_W | Optimal L1 | Key finding |\n")
    f.write("| ----- | ------ | --- | -------- | --------- | ------- | -------- | ------- | ------------ | ---------- | ----------- |\n\n")
    f.write("### Established Principles\n\n")
    f.write("### Open Questions\n\n")
    f.write("---\n\n")
    f.write("## Previous Block Summary\n\n")
    f.write("---\n\n")
    f.write("## Current Block (Block 1)\n\n")
    f.write("### Block Info\n\n")
    f.write("### Hypothesis\n\n")
    f.write("### Iterations This Block\n\n")
    f.write("### Emerging Observations\n\n")
print(f"\033[93mInitialized: {memory_path}\033[0m")

# Erase UCB file (fresh start)
if os.path.exists(ucb_path):
    os.remove(ucb_path)

# Compute UCB scores
compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                   current_log_path=analysis_log_path,
                   current_iteration=ITERATION,
                   block_size=n_iter_block)
print(f"\033[92mUCB scores computed (c={ucb_c}): {ucb_path}\033[0m")

# Check files are ready
time.sleep(2)
if not os.path.exists(activity_path):
    print(f"\033[91merror: activity.png not found at {activity_path}\033[0m")
    raise SystemExit(1)
if not os.path.exists(analysis_log_path):
    print(f"\033[91merror: analysis.log not found at {analysis_log_path}\033[0m")
    raise SystemExit(1)
print("\033[92mFiles ready: activity.png, analysis.log\033[0m")

# ── Phase 2: Initialize reasoning log (empty, append across all repeats) ──
open(reasoning_repeat_path, 'w').close()
print(f"\033[93mInitialized: {reasoning_repeat_path}\033[0m")

# ── Phase 3: Claude loop (N_REPEATS times) ────────────────────────────────
print(f"\n\033[94m{'='*60}\033[0m")
print(f"\033[94mStarting {N_REPEATS} Claude repeats\033[0m")
print(f"\033[94m{'='*60}\033[0m\n")

for k in range(1, N_REPEATS + 1):
    print(f"\n\033[96m{'─'*60}\033[0m")
    print(f"\033[96mRepeat {k}/{N_REPEATS}\033[0m")
    print(f"\033[96m{'─'*60}\033[0m\n")

    # Reset repeat files to empty BEFORE each Claude call (true independent repeat)
    with open(analysis_repeat_path, 'w') as f:
        f.write(f"# Repeat Experiment Log: {config_file_} — Repeat {k}\n\n")
    with open(memory_repeat_path, 'w') as f:
        f.write(f"# Repeat Working Memory: {config_file_} — Repeat {k}\n\n")

    claude_prompt = f"""Repeat {k}/{N_REPEATS} — Iteration {ITERATION}/{N_ITERATIONS}
Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block
{">>> BLOCK END <<<" if is_block_end else ""}

Instructions (follow all instructions): {instruction_path}
Working memory (READ-ONLY): {memory_path}
Full log (READ-ONLY): {analysis_path}
Repeat analysis (WRITE here): {analysis_repeat_path}
Repeat memory (WRITE here): {memory_repeat_path}
Activity image: {activity_path}
Metrics log: {analysis_log_path}
UCB scores: {ucb_path}
Current config (READ-ONLY): {config_path}"""

    claude_cmd = [
        'claude',
        '-p', claude_prompt,
        '--output-format', 'text',
        '--max-turns', '500',
        '--allowedTools',
        'Read', 'Edit', 'Write'
    ]

    # Run with real-time output streaming
    output_lines = []
    process = subprocess.Popen(
        claude_cmd,
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)

    process.wait()

    # Check for OAuth token expiration
    output_text = ''.join(output_lines)
    if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
        print(f"\n\033[91m{'='*60}\033[0m")
        print(f"\033[91mOAuth token expired at repeat {k}\033[0m")
        print("\033[93mTo resume: claude /login\033[0m")
        print(f"\033[91m{'='*60}\033[0m")
        raise SystemExit(1)

    # Append Claude's output to reasoning repeat log
    if output_text.strip():
        with open(reasoning_repeat_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"=== Repeat {k}/{N_REPEATS} ===\n")
            f.write(f"{'='*60}\n")
            f.write(output_text.strip())
            f.write("\n\n")

    # Save numbered copies of this repeat's output
    analysis_k = f"{root_dir}/{llm_task_name}_analysis_repeat_{k}.md"
    memory_k = f"{root_dir}/{llm_task_name}_memory_repeat_{k}.md"
    shutil.copy2(analysis_repeat_path, analysis_k)
    shutil.copy2(memory_repeat_path, memory_k)
    print(f"\033[92mSaved: {analysis_k}\033[0m")
    print(f"\033[92mSaved: {memory_k}\033[0m")

    print(f"\n\033[92mRepeat {k}/{N_REPEATS} completed\033[0m")

print(f"\n\033[94m{'='*60}\033[0m")
print(f"\033[94mAll {N_REPEATS} repeats completed\033[0m")
print("\033[94mResults:\033[0m")
print(f"\033[94m  Analysis: {analysis_repeat_path}\033[0m")
print(f"\033[94m  Memory:   {memory_repeat_path}\033[0m")
print(f"\033[94m  Reasoning: {reasoning_repeat_path}\033[0m")
print(f"\033[94m{'='*60}\033[0m")
