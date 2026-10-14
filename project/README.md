# AI-Assisted Air Traffic Conflict Prediction & Resolution Evaluation System

> **Research / Academic Prototype**  
> **Simulation & Demo Data Notice:** All aircraft trajectories, radar measurements, sensor models, and atmospheric disturbances in this project are simulated for educational and research prototyping. This is **NOT** a certified aviation safety system and must not be used for actual air navigation or operational air traffic control.

---

## 1. System Overview

This project upgrades an autonomous air traffic control (ATC) simulation into an **AI-Assisted Air Traffic Conflict Prediction and Resolution Evaluation System**. The system integrates:
- **12-State Nonlinear Aircraft Flight Dynamics** (NASA wind-axis equations of motion, RK4 integration, ISA atmospheric density).
- **Extended Kalman Filter (EKF)** fusing IMU (gyro/accel), GPS (pos/vel), Barometer, Magnetometer, and Airspeed with Joseph-form covariance updates and innovation gating.
- **Surveillance Fusion Layer** combining ADS-B broadcasts and 3D primary radar tracks with alpha-beta tracking.
- **Dual-Monitor Conflict Prediction Layer**:
  - **Baseline:** 4D Geometric Trajectory Extrapolator (`conflict_detector_4d`).
  - **Machine Learning Engine:** Causal, uncertainty-aware Logistic Regression and Random Forest classifiers predicting $P(\text{conflict})$ and early warning lead time.
- **Tactical Conflict Resolver** evaluating candidate heading, altitude, and speed maneuvers using a multi-attribute cost function.
- **Guidance & Inner-Loop Autopilot** combining a Linear Time-Varying Model Predictive Controller (LTV-MPC) with attitude/rate feedback dampers.
- **Comprehensive Performance Dashboard & Automated Experiment Runner** providing publication-ready plots, CSV exports, and statistical verification.

---

## 2. Directory Structure & Key Modules

```
d:/a0/amrita/s5/robotics/project/
│
├── UnifiedFlightRadar3D.m        # Main entrypoint, 3D radar command center, physics & ATC simulation
├── performance_dashboard.m       # Interactive KPI & metric evaluation dashboard GUI
├── ml_conflict_predictor.m       # Real-time ML conflict inference engine (class)
├── ml_feature_extractor.m        # Causal, uncertainty-aware feature extraction
├── ml_dataset_generator.m        # Leakage-free training dataset generator
├── ml_train_models.m             # ML training, cross-validation, and benchmarking pipeline
├── experiment_runner.m           # Parametric Monte Carlo experiment runner & CSV/plot exporter
├── run_regression_tests.m        # Automated 15-scenario regression test harness
├── ml_conflict_model.mat         # Saved trained model artifacts, scalers, and evaluation metrics
├── docs_architecture_plan.md     # Architectural map and technical specifications
│
└── exports/                      # Exported CSVs and publication-quality figures
    ├── experiment_results_trials.csv     # Trial-by-trial detailed metrics
    ├── experiment_results_summary.csv    # Aggregated metrics grouped by Scenario & Weather
    ├── ml_benchmark_metrics.csv          # Classification metrics for Rule vs Logistic vs RF
    ├── fig_detector_vs_ml_metrics.png    # Precision, Recall, F1, and FAR bar comparison
    ├── fig_environmental_robustness.png  # Weather uncertainty & noise stress response
    ├── fig_roc_curves.png                # Receiver Operating Characteristic curves
    └── fig_lead_time_distribution.png    # Early warning lead-time distribution
```

---

## 3. Requirements & Environment

- **MATLAB Version:** MATLAB R2021a or newer (developed and verified on MATLAB R2025b Update 7).
- **Required Toolboxes:**
  - *Statistics and Machine Learning Toolbox* (for `TreeBagger`, `fitglm`, and classification metrics).
  - *Optimization Toolbox* (optional; built-in box QP solver `qp_box_solve` automatically serves as active-set fallback).
- **Operating System:** Windows, macOS, or Linux.

---

## 4. Execution & Quick Start Guide

Open MATLAB, navigate to the project directory, and run any of the following commands:

### A. Run 3D ATC Simulation with Interactive Radar
```matlab
% Run standard head-on encounter with live 3D visualization and dual ML monitoring
UnifiedFlightRadar3D('head_on')

% Run any of the 15 supported scenarios:
UnifiedFlightRadar3D('crossing')
UnifiedFlightRadar3D('overtaking')
UnifiedFlightRadar3D('multi_conflict')
UnifiedFlightRadar3D('dense_airspace')
UnifiedFlightRadar3D('converging_hub')
```
*Live Figure Shortcuts:*
- `1`..`9`: Focus camera on specific aircraft
- `c`: Cycle camera view (`fixed`, `follow`, `orbit`, `top`, `auto`)
- `r`: Toggle primary radar display
- `t`: Toggle telemetry HUD
- `p`: Toggle 4D predicted trajectories
- `m`: Toggle performance / diagnostic panel
- `g`: Toggle Primary Flight Display (PFD) / artificial horizon
- `Space`: Pause/resume simulation
- `x`: Stop simulation

---

### B. Launch Comprehensive Performance Dashboard
```matlab
% Launch dedicated multi-panel performance evaluation dashboard
performance_dashboard()

% Or via the main router:
UnifiedFlightRadar3D('dashboard')
```

---

### C. Run Full Regression Test Suite (All 15 Scenarios)
```matlab
% Verifies numerical stability, physics invariants, EKF, and MPC
summary = run_regression_tests()

% Or via the main router:
UnifiedFlightRadar3D('test')
```

---

### D. Run Automated Multi-Condition Experiments & Export CSVs
```matlab
% Runs multi-seed experiments across Calm, Moderate, and Severe weather conditions
% Exports structured CSV files and publication-quality PNG plots into /exports/
exp_results = experiment_runner()

% Custom parametric experiment:
exp_results = experiment_runner(struct('n_trials', 5, 'scenarios', {{'head_on', 'crossing', 'multi_conflict'}}))

% Or via the main router:
UnifiedFlightRadar3D('experiment')
```

---

### E. Retrain & Calibrate Machine Learning Models
```matlab
% Regenerates leakage-free dataset, trains Logistic Regression & Random Forest,
% benchmarks against Rule-Based detector, and saves ml_conflict_model.mat
pkg = ml_train_models()

% Or via the main router:
UnifiedFlightRadar3D('train')
```

---

## 5. Metric Definitions & Aviation Units

| Metric | Symbol / Formula | Unit | Definition |
| :--- | :--- | :--- | :--- |
| **Horizontal Separation** | $d_h = \sqrt{\Delta x^2 + \Delta y^2}$ | meters [m] | Current distance between two aircraft in the local horizontal plane. |
| **Vertical Separation** | $d_v = \|\Delta z\|$ | meters [m] | Current altitude difference between two aircraft. |
| **Loss of Separation (LoS)** | $d_h < 1500\text{ m} \land d_v < 300\text{ m}$ | Boolean / Count | Breach of minimum separation standard cylinder ($H_{\text{sep}} = 1500$ m, $V_{\text{sep}} = 300$ m). |
| **Conflict Recall** | $\frac{\text{TP}}{\text{TP} + \text{FN}} \times 100\%$ | percentage [%] | Fraction of true impending separation losses successfully flagged by predictor. |
| **Conflict Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}} \times 100\%$ | percentage [%] | Proportion of issued alerts that correspond to genuine impending separation losses. |
| **False Alarm Rate (FAR)**| $\frac{\text{FP}}{\text{FP} + \text{TN}} \times 100\%$ | percentage [%] | Proportion of non-conflicting encounters falsely flagged as conflicts. |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | dimensionless | Harmonic mean of precision and recall. |
| **Time to CPA (TCPA)** | $\tau_{\text{cpa}} = -\frac{\Delta\mathbf{r}\cdot\Delta\mathbf{v}}{\|\Delta\mathbf{v}\|^2}$ | seconds [s] | Time remaining until the two aircraft reach closest point of approach. |
| **Early Warning Lead Time** | $t_{\text{loss}} - t_{\text{detect}}$ | seconds [s] | Time difference between initial conflict alert and actual separation loss. |
| **Inference Latency** | $t_{\text{infer}}$ | microseconds [$\mu$s]| Wall-clock duration required to extract features and evaluate ML probability. |
| **EKF Position RMSE** | $\sqrt{\frac{1}{N}\sum (\mathbf{r} - \hat{\mathbf{r}})^2}$ | meters [m] | State estimation error between true aircraft position and EKF estimate. |

---

## 6. Safety & Dual-Monitoring Architecture

The machine learning module is strictly designed as an **advisory decision-support layer**:
1. **Rule Safeguard Backstop:** The rule-based geometric detector acts as a safety invariant. The ML model cannot override or suppress a safety-critical maneuver commanded by the rule-based ATC resolver.
2. **Early Advisory:** The ML model triggers an early warning if $P(\text{conflict}) \ge 0.50$, even when the geometric detector has not yet reached its time threshold, granting extra tactical planning lead time.
3. **Disagreement Tracking:** The system logs every disagreement between the AI predictor and rule detector, enabling comprehensive post-run safety audits.
4. **Uncertainty Propagation:** State uncertainty from the EKF position covariance $\mathbf{P}$ is incorporated into the feature vector and visualized in the tactical displays.

---

## 7. Experimental Results Summary

Evaluated on an independent test split of 1,067 simulated encounter samples:

| Model | Accuracy | Recall (Sensitivity) | Precision (PPV) | F1-Score | False Alarm Rate (FAR) | Brier Score | AUC | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based (Baseline)** | 91.6% | 96.5% | 62.1% | 0.755 | 9.2% | 0.0843 | 0.9004 | **4.0 $\mu$s** |
| **Logistic Regression** | **98.9%** | **100.0%** | **92.3%** | **0.960** | **1.3%** | **0.0113** | **0.9961** | **291.5 $\mu$s** |
| **Random Forest** | **100.0%** | **100.0%** | **100.0%** | **1.000** | **0.0%** | **0.0034** | **1.0000** | **16,238 $\mu$s** |

### Key Findings:
- **False Alarm Reduction:** Logistic Regression reduced false alarms by **85.9%** compared to the rule-based baseline (12 false alarms vs 85 false alarms) while achieving **100% recall**.
- **Early Warning:** AI prediction provided an average of **25.3 seconds** of warning lead time before CPA, compared to 24.2 seconds for the rule-based detector, enabling earlier advisory replanning.
- **Latency Trade-off:** Logistic Regression evaluated in **0.29 ms** (well within the 20 ms / 50 Hz real-time flight control budget), whereas Random Forest required **16.2 ms**.

---

## 8. License & Citation

Developed for academic and university research demonstration. All algorithms, dynamical models, and evaluation pipelines are fully reproducible with fixed random seeds.
