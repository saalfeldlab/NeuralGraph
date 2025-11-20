# Optimization Status - FRESH START

## Iteration 0: Baseline Training ✅ COMPLETED
**Date**: 2025-11-18 (new session)
**Config**: fly_N9_62_22_10
**Status**: COMPLETED
**Training bash_id**: 6c71e2

**Parameters**:
- coeff_W_sign: 0.01
- W_sign_temperature: 10.0 (baseline)
- coeff_W_L2: 0.0
- coeff_W_L1: 5.0E-5
- data_augmentation_loop: 1
- n_epochs: 3

**Rationale**: Establish baseline performance with standard Dale's Law regularization parameters.

**Results**:
- **Dale's Law**: Excitatory: 995, Inhibitory: 2367, Mixed: 9917 → **74.7% mixed neurons**
- **Weights R²**: 0.945 (2nd fit), slope: 0.993 → Excellent
- **GMM Accuracy**: 0.898 → Excellent
- **Tau R²**: 0.945, slope: 0.92
- **RMSE**: Weights: 0.042±0.071

**Analysis**: High temperature (10.0) causes severe Dale's Law violations (74.7% mixed). Model quality is excellent otherwise. Need to reduce temperature to soften sigmoid.

---

## Iteration 1: Lower Temperature (5.0)
**Date**: 2025-11-18
**Config**: fly_N9_62_22_10
**Status**: TRAINING IN PROGRESS
**Training bash_id**: 553d4c

**Parameters**:
- coeff_W_sign: 0.01
- W_sign_temperature: 5.0 → **Reduced from 10.0**
- coeff_W_L2: 0.0
- coeff_W_L1: 5.0E-5
- data_augmentation_loop: 1
- n_epochs: 3

**Rationale**: Test Hypothesis 1 - Lower temperature should soften sigmoid(temp*w), allowing small weights near zero to contribute less to violation penalty. Expected to reduce mixed neurons to ~40-50%.

**Prediction**: Mixed neurons will decrease from 74.7% to ~45-55%, but may not reach <10% target yet.
