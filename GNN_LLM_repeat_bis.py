import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import os
import random
import shutil
import subprocess
import yaml

# redirect PyTorch JIT cache to /scratch instead of /tmp (per IT request)
if os.path.isdir('/scratch'):
    os.environ['TMPDIR'] = '/scratch/allierc'
    os.makedirs('/scratch/allierc', exist_ok=True)

from NeuralGraph.config import NeuralGraphConfig
from NeuralGraph.models.graph_trainer import data_test
from NeuralGraph.models.exploration_tree import compute_ucb_scores
from NeuralGraph.models.utils import save_exploration_artifacts
from NeuralGraph.utils import set_device, add_pre_folder
from GNN_PlotFigure import data_plot

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ──────────────────────────────────────────────────────────
N_REPEATS = 10
ITERATION = 117          # next iteration after last completed (116)
N_ITERATIONS = 2048      # total iterations (for prompt display)

llm_task_name = 'signal_landscape_Claude'
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

# Repeat output files (use _repeat_bis suffix)
analysis_repeat_path = f"{root_dir}/{llm_task_name}_analysis_repeat_bis.md"
memory_repeat_path = f"{root_dir}/{llm_task_name}_memory_repeat_bis.md"
reasoning_repeat_path = f"{root_dir}/{llm_task_name}_reasoning_repeat_bis.log"

# Shuffled config path (written before each Claude call)
shuffled_config_path = f"{config_root}/{pre_folder}{config_file_}_shuffled.yaml"

# ── Training keys to shuffle ──────────────────────────────────────────────
# These are the training section keys whose ORDER will be randomized
TRAINING_KEYS = [
    'n_epochs', 'n_runs', 'device', 'batch_size', 'small_init_batch_size',
    'seed', 'data_augmentation_loop', 'sparsity', 'sparsity_freq',
    'cluster_method', 'cluster_distance_threshold', 'fix_cluster_embedding',
    'learning_rate_W_start', 'learning_rate_start', 'learning_rate_embedding_start',
    'n_epochs_init', 'first_coeff_L1', 'coeff_W_L1', 'coeff_edge_diff',
    'low_rank_factorization', 'low_rank',
]


def write_shuffled_config(src_path, dst_path, seed):
    """Read config, shuffle training key order, write to dst."""
    with open(src_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Shuffle training section key order
    training = config_data.get('training', {})
    keys = list(training.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    shuffled_training = {k: training[k] for k in keys}
    config_data['training'] = shuffled_training

    with open(dst_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)


# ── Phase 1: Run data_test + data_plot (once) ─────────────────────────────
print(f"\n\033[94m{'='*60}\033[0m")
print(f"\033[94mRepeat-bis Reasoning Script — Iteration {ITERATION}\033[0m")
print(f"\033[94mBlock {block_number}, iter {iter_in_block}/{n_iter_block}\033[0m")
print(f"\033[94mTraining key order will be shuffled each repeat\033[0m")
print(f"\033[94m{'='*60}\033[0m\n")

# Open analysis.log for test/plot metrics
log_file = open(analysis_log_path, 'w')

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

# Compute UCB scores
compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                   current_log_path=analysis_log_path,
                   current_iteration=ITERATION,
                   block_size=n_iter_block)
print(f"\033[92mUCB scores computed (c={ucb_c}): {ucb_path}\033[0m")

# Check files are ready
import time
time.sleep(2)
if not os.path.exists(activity_path):
    print(f"\033[91merror: activity.png not found at {activity_path}\033[0m")
    raise SystemExit(1)
if not os.path.exists(analysis_log_path):
    print(f"\033[91merror: analysis.log not found at {analysis_log_path}\033[0m")
    raise SystemExit(1)
if not os.path.exists(ucb_path):
    print(f"\033[91merror: ucb_scores.txt not found at {ucb_path}\033[0m")
    raise SystemExit(1)
print("\033[92mFiles ready: activity.png, analysis.log, ucb_scores.txt\033[0m")

# ── Phase 2: Initialize reasoning log (empty, append across all repeats) ──
open(reasoning_repeat_path, 'w').close()
print(f"\033[93mInitialized: {reasoning_repeat_path}\033[0m")

# ── Phase 3: Claude loop (N_REPEATS times) ────────────────────────────────
print(f"\n\033[94m{'='*60}\033[0m")
print(f"\033[94mStarting {N_REPEATS} Claude repeats (shuffled training keys)\033[0m")
print(f"\033[94m{'='*60}\033[0m\n")

for k in range(1, N_REPEATS + 1):
    print(f"\n\033[96m{'─'*60}\033[0m")
    print(f"\033[96mRepeat {k}/{N_REPEATS}\033[0m")
    print(f"\033[96m{'─'*60}\033[0m\n")

    # Shuffle training key order and write to shuffled config
    write_shuffled_config(config_path, shuffled_config_path, seed=k)
    print(f"\033[93mShuffled training keys (seed={k}) -> {shuffled_config_path}\033[0m")

    # Reset repeat files to empty BEFORE each Claude call (true independent repeat)
    with open(analysis_repeat_path, 'w') as f:
        f.write(f"# Repeat-bis Experiment Log: {config_file_} — Repeat {k}\n\n")
    with open(memory_repeat_path, 'w') as f:
        f.write(f"# Repeat-bis Working Memory: {config_file_} — Repeat {k}\n\n")

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
Current config (READ-ONLY): {shuffled_config_path}"""

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
            f.write(f"=== Repeat {k}/{N_REPEATS} (seed={k}) ===\n")
            f.write(f"{'='*60}\n")
            f.write(output_text.strip())
            f.write("\n\n")

    # Save numbered copies of this repeat's output
    analysis_k = f"{root_dir}/{llm_task_name}_analysis_repeat_bis_{k}.md"
    memory_k = f"{root_dir}/{llm_task_name}_memory_repeat_bis_{k}.md"
    shutil.copy2(analysis_repeat_path, analysis_k)
    shutil.copy2(memory_repeat_path, memory_k)
    print(f"\033[92mSaved: {analysis_k}\033[0m")
    print(f"\033[92mSaved: {memory_k}\033[0m")

    print(f"\n\033[92mRepeat {k}/{N_REPEATS} completed\033[0m")

# Clean up shuffled config
if os.path.exists(shuffled_config_path):
    os.remove(shuffled_config_path)
    print(f"\033[93mCleaned up: {shuffled_config_path}\033[0m")

print(f"\n\033[94m{'='*60}\033[0m")
print(f"\033[94mAll {N_REPEATS} repeats completed (shuffled training keys)\033[0m")
print(f"\033[94mResults:\033[0m")
print(f"\033[94m  Analysis: {analysis_repeat_path}\033[0m")
print(f"\033[94m  Memory:   {memory_repeat_path}\033[0m")
print(f"\033[94m  Reasoning: {reasoning_repeat_path}\033[0m")
print(f"\033[94m{'='*60}\033[0m")
