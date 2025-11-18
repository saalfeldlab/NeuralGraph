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
- Execute: `source /opt/conda/etc/profile.d/conda.sh && conda activate neural-graph-linux && python GNN_Main.py`
- Run in background, save bash_id
- Update status.md with bash_id and "TRAINING IN PROGRESS"

### Step 3: Monitor training completion
**CRITICAL: Implement polling loop to detect completion**
- Check BashOutput status every 30 minutes (sleep 1800)
- Use BashOutput tool with bash_id to check status
- Look for `<status>completed</status>` in the response
- Check exit_code: 0 means success, non-zero means failure
- When status="completed", immediately proceed to Step 4
- Training takes ~1 hour, so 2-3 checks total
- MUST verify status is actually "completed" not "running"

### Step 4: Analyze results
- Execute: `source /opt/conda/etc/profile.d/conda.sh && conda activate neural-graph-linux && python GNN_PlotFigure.py`
- Wait for completion
- Read log/fly/fly_N9_62_22_10/results.log for metrics

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
- Use BashOutput to check training status every 30 minutes
- Training takes ~1 hour, check 2-3 times total
- Parse the XML response for `<status>completed</status>` tag
- Verify exit_code=0 for successful completion
- If status is still "running", continue polling loop
- Only proceed to analysis when status is definitively "completed"
- Log progress periodically to status.md

## Success Criteria
- Mixed neurons: <10%
- No distinct ±0.5 clustering
- Model performance maintained or improved
- Smooth weight distribution
