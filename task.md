# Autonomous Optimization Task

## Objective
Optimize Dale's Law regularization to reduce mixed neurons while preventing weight clustering at ±0.5

## Current Problem
- Dale's law regularization (coeff_W_sign: 0.01, W_sign_temperature: 10.0) reduces mixed neurons from 60% to ~10%
- BUT: weights close to zero are repulsed away from zero, creating symmetric clusters at ±0.5
- Root cause: sigmoid(10*w) transitions too sharply, forcing small weights to "pick a side"

## Solution Hypotheses Being Tested

### Hypothesis 1: Temperature Adjustment
Lower W_sign_temperature → softer sigmoid → small weights contribute less to violation
- Test values: 5.0, 3.0, 2.0

### Hypothesis 2: L2 Regularization
Add coeff_W_L2 → quadratic penalty toward zero → counteracts Dale's law repulsion
- Test values: 1.0E-4, 5.0E-5, 1.0E-5

## Baseline
Config: fly_N9_62_22_3
- coeff_W_sign: 0.01
- W_sign_temperature: 10.0
- coeff_W_L2: 0 (none)
- Mixed neurons: ~56% (from previous run)

## Current Test Config
fly_N9_62_22_10 (working config that will be modified each iteration)

## Key Metrics to Track
1. **Dale's Law violation rate**: % Mixed neurons (target: <10% without clustering)
2. **Weight distribution**: Check for ±0.5 clustering in histogram
3. **Model performance**: FEVE scores, R² metrics
4. **Regularization balance**: W_L1, W_L2, W_sign magnitudes in loss plots

## Optimization Loop - DETAILED EXECUTION STEPS

### Step 1: Modify config based on previous results
- Read status.md to get current iteration state
- Determine next parameter values to test
- Edit config/fly/fly_N9_62_22_10.yaml with new parameters
- Update status.md with new config parameters

### Step 2: Run training
- Execute: `source /opt/conda/etc/profile.d/conda.sh && conda activate neural-graph-linux && python GNN_Main.py -o training fly_N9_62_22_10`
- Run in background, save bash_id
- Update status.md with bash_id and "TRAINING IN PROGRESS"
- **Verify training started**: Wait 20-30 seconds, check BashOutput for "start training" message

### Step 3: Monitor training completion
**CRITICAL: Implement AUTONOMOUS polling loop to detect completion**
- Check BashOutput status every 15 minutes (sleep 900)
- Use BashOutput tool with bash_id to check for "training completed." message
- Look for the literal string "training completed." in stdout
- When "training completed." is found, immediately proceed to Step 4
- Training takes ~1 hour, so 4-5 checks total
- Loop continues AUTONOMOUSLY without user intervention
- Can use filter parameter: `BashOutput(bash_id, filter="training completed")`

### Step 4: Analyze results
- Execute: `source /opt/conda/etc/profile.d/conda.sh && conda activate neural-graph-linux && python GNN_PlotFigure.py`
- Run in background, save bash_id
- Monitor for "training completed." message in BashOutput (indicates analysis finished)
- When "training completed." found, read log/fly/fly_N9_62_22_10/results.log for metrics

### Step 5: Evaluate improvements
- Extract key metrics from results.log:
  - Dale's Law violation %
  - Model performance (R², FEVE)
  - Loss values
- Compare to previous iterations
- Determine if improvement occurred

### Step 6: Choose best modification
- Based on evaluation, decide next parameter to test
- If hypothesis validated: refine in that direction
- If hypothesis failed: try alternative approach
- Document decision reasoning

### Step 7: Update status.md
- Add completed iteration results
- Document decision for next iteration
- Increment iteration counter
- Write next planned modification

### Step 8: Return to Step 1
- Go back to Step 1 with new parameters

## Monitoring Best Practices
- Use BashOutput to check for "training completed." message every 15 minutes
- Training takes ~1 hour, check 4-5 times total
- Search for literal string "training completed." in stdout
- Both GNN_Main.py and GNN_PlotFigure.py print this message when done
- Use filter parameter for efficiency: `filter="training completed"`
- If message not found, continue polling loop AUTONOMOUSLY
- Only proceed to next step when "training completed." is found
- Log progress periodically to status.md
- **Loop runs completely autonomously without user intervention**

## Success Criteria
- Mixed neurons: <10%
- No distinct ±0.5 clustering
- Model performance maintained or improved
- Smooth weight distribution
