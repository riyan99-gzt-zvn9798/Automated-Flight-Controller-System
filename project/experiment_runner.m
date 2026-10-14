function exp_output = experiment_runner(cfg_exp)
%EXPERIMENT_RUNNER  Automated, reproducible experiment runner and evaluation pipeline.
%   Executes parametric simulation matrices across scenarios, seeds, and weather conditions.
%   Records trial-by-trial telemetry, calculates statistical aggregations, exports CSVs,
%   and produces publication-quality comparison figures.
%
%   Usage:
%       exp_output = experiment_runner()
%       exp_output = experiment_runner(struct('n_trials', 5, 'scenarios', {{'head_on','crossing'}}))

if nargin < 1, cfg_exp = struct(); end
if ~isfield(cfg_exp, 'scenarios')
    cfg_exp.scenarios = {'nominal', 'head_on', 'crossing', 'overtaking', 'multi_conflict', 'crosswind', 'sensor_noise'};
end
if ~isfield(cfg_exp, 'conditions')
    cfg_exp.conditions = {'calm', 'moderate', 'severe'};
end
if ~isfield(cfg_exp, 'n_trials'), cfg_exp.n_trials = 3; end
if ~isfield(cfg_exp, 'base_seed'), cfg_exp.base_seed = 1000; end
if ~isfield(cfg_exp, 'T_sim'), cfg_exp.T_sim = 45; end % Duration per trial [s]
if ~isfield(cfg_exp, 'export_dir'), cfg_exp.export_dir = 'exports'; end
if ~isfield(cfg_exp, 'save_plots'), cfg_exp.save_plots = true; end

if ~exist(cfg_exp.export_dir, 'dir')
    mkdir(cfg_exp.export_dir);
end

% Ensure ML predictor is trained and loaded
fprintf('Initializing ML conflict predictor...\n');
predictor = ml_conflict_predictor();

nScens = numel(cfg_exp.scenarios);
nConds = numel(cfg_exp.conditions);
nTrials = cfg_exp.n_trials;
totalRuns = nScens * nConds * nTrials;

fprintf('========================================================================================\n');
fprintf('                STARTING AUTOMATED EXPERIMENT RUNNER (%d TOTAL RUNS)                    \n', totalRuns);
fprintf('========================================================================================\n');
fprintf('Scenarios:  %s\n', strjoin(cfg_exp.scenarios, ', '));
fprintf('Conditions: %s\n', strjoin(cfg_exp.conditions, ', '));
fprintf('Trials/cfg: %d | Duration: %.1f s | Output Dir: %s\n', nTrials, cfg_exp.T_sim, cfg_exp.export_dir);
fprintf('----------------------------------------------------------------------------------------\n');

% Initialize trial records table/struct
trial_records = [];
runIdx = 0;
totalTic = tic;

for s = 1:nScens
    scName = cfg_exp.scenarios{s};
    for c = 1:nConds
        condName = cfg_exp.conditions{c};
        
        fprintf('>>> Testing Scenario: %-15s | Condition: %-8s (%d trials)...\n', scName, condName, nTrials);
        
        for tr = 1:nTrials
            runIdx = runIdx + 1;
            seed = cfg_exp.base_seed + 100*s + 10*c + tr;
            
            % Setup configuration overrides for environment condition
            opts = struct('T', cfg_exp.T_sim, 'seed', seed, 'viz', struct('enable', false));
            
            switch lower(condName)
                case 'calm'
                    opts.wind = struct('speed', 4, 'turb', 0.6);
                    opts.sens = struct('noise_scale', 0.8);
                    opts.adsb = struct('p_drop', 0.01);
                case 'moderate'
                    opts.wind = struct('speed', 12, 'turb', 1.2);
                    opts.sens = struct('noise_scale', 1.2);
                    opts.adsb = struct('p_drop', 0.03);
                case 'severe'
                    opts.wind = struct('speed', 22, 'turb', 2.0);
                    opts.sens = struct('noise_scale', 2.2);
                    opts.adsb = struct('p_drop', 0.07);
            end
            
            tRunTic = tic;
            runStatus = 'PASS';
            errMsg = '';
            
            try
                res = UnifiedFlightRadar3D(scName, opts);
                tRun = toc(tRunTic);
                M = res.M;
                W = res.W;
                
                % ML conflict evaluation across the encounter trajectory
                C_ml = predictor.evaluate_airspace(W, res.cfg);
                n_ml_conf = numel(C_ml);
                n_ml_crit = sum([C_ml.warning_level] == 2);
                disagreements = sum([C_ml.disagreement]);
                
                % Record trial record
                rec = struct( ...
                    'run_id', runIdx, ...
                    'scenario', scName, ...
                    'condition', condName, ...
                    'trial', tr, ...
                    'seed', seed, ...
                    'status', runStatus, ...
                    'duration_s', tRun, ...
                    'minH_m', M.minH, ...
                    'minV_m', M.minV, ...
                    'min3D_m', M.min3D, ...
                    'los_events', M.los_events, ...
                    'rule_conflicts', M.conflicts, ...
                    'rule_warnings', M.warnings, ...
                    'rule_criticals', M.criticals, ...
                    'rule_resolved', M.resolved, ...
                    'rule_unresolved', M.unresolved, ...
                    'ml_conflicts', n_ml_conf, ...
                    'ml_criticals', n_ml_crit, ...
                    'disagreements', disagreements, ...
                    'maneuvers', M.maneuvers, ...
                    'recoveries', M.recoveries, ...
                    'zone_violations', M.viol.zone, ...
                    'crashed', M.crashed, ...
                    'ekf_pos_rmse_m', mean(M.ekf_pos_rmse), ...
                    'ekf_alt_rmse_m', mean(M.ekf_alt_rmse), ...
                    'ekf_spd_rmse_mps', mean(M.ekf_spd_rmse), ...
                    'mpc_time_mean_ms', M.mpc_time_mean * 1e3, ...
                    'mpc_time_max_ms', M.mpc_time_max * 1e3, ...
                    'mpc_fallbacks', M.mpc_fallbacks, ...
                    'ml_latency_mean_us', predictor.stats.total_time_us / max(predictor.stats.calls, 1), ...
                    'error_msg', errMsg ...
                );
            catch ME
                tRun = toc(tRunTic);
                rec = struct( ...
                    'run_id', runIdx, ...
                    'scenario', scName, ...
                    'condition', condName, ...
                    'trial', tr, ...
                    'seed', seed, ...
                    'status', 'ERROR', ...
                    'duration_s', tRun, ...
                    'minH_m', NaN, 'minV_m', NaN, 'min3D_m', NaN, ...
                    'los_events', NaN, 'rule_conflicts', NaN, 'rule_warnings', NaN, ...
                    'rule_criticals', NaN, 'rule_resolved', NaN, 'rule_unresolved', NaN, ...
                    'ml_conflicts', NaN, 'ml_criticals', NaN, 'disagreements', NaN, ...
                    'maneuvers', NaN, 'recoveries', NaN, 'zone_violations', NaN, ...
                    'crashed', NaN, 'ekf_pos_rmse_m', NaN, 'ekf_alt_rmse_m', NaN, ...
                    'ekf_spd_rmse_mps', NaN, 'mpc_time_mean_ms', NaN, 'mpc_time_max_ms', NaN, ...
                    'mpc_fallbacks', NaN, 'ml_latency_mean_us', NaN, ...
                    'error_msg', ME.message ...
                );
            end
            
            if isempty(trial_records)
                trial_records = rec;
            else
                trial_records(end+1) = rec; %#ok<AGROW>
            end
        end
    end
end

totalElapsed = toc(totalTic);
fprintf('----------------------------------------------------------------------------------------\n');
fprintf('Completed %d simulation runs in %.2f seconds.\n', totalRuns, totalElapsed);

% Convert to Table for Structured Data Analysis & CSV Export
T_trials = struct2table(trial_records);
trials_csv_path = fullfile(cfg_exp.export_dir, 'experiment_results_trials.csv');
writetable(T_trials, trials_csv_path);
fprintf('Saved trial records CSV: %s\n', trials_csv_path);

% Compute Statistical Aggregations (grouped by Scenario and Condition)
T_summary = compute_summary_table(T_trials);
summary_csv_path = fullfile(cfg_exp.export_dir, 'experiment_results_summary.csv');
writetable(T_summary, summary_csv_path);
fprintf('Saved summary metrics CSV: %s\n', summary_csv_path);

% Export ML Evaluation Metrics CSV
ml_metrics_csv_path = fullfile(cfg_exp.export_dir, 'ml_benchmark_metrics.csv');
export_ml_metrics_csv(predictor.model_pkg.metrics, ml_metrics_csv_path);
fprintf('Saved ML benchmark metrics CSV: %s\n', ml_metrics_csv_path);

% Generate Publication-Ready Comparison Figures
if cfg_exp.save_plots
    fprintf('\nGenerating publication-quality comparison figures in %s...\n', cfg_exp.export_dir);
    generate_comparison_plots(T_trials, predictor.model_pkg, cfg_exp.export_dir);
end

fprintf('========================================================================================\n');
fprintf('                EXPERIMENT RUNNER COMPLETE                                              \n');
fprintf('========================================================================================\n\n');

exp_output = struct( ...
    'trials_table', T_trials, ...
    'summary_table', T_summary, ...
    'predictor', predictor, ...
    'total_runs', totalRuns, ...
    'total_elapsed_s', totalElapsed, ...
    'export_dir', cfg_exp.export_dir ...
);
end

%% ==================== HELPER FUNCTIONS ====================

function T_sum = compute_summary_table(T)
[G, sc_u, cd_u] = findgroups(T.scenario, T.condition);
nGroups = max(G);

sc_col = cell(nGroups, 1);
cd_col = cell(nGroups, 1);
n_runs = zeros(nGroups, 1);
minH_mean = zeros(nGroups, 1); minH_std = zeros(nGroups, 1); minH_min = zeros(nGroups, 1);
los_mean = zeros(nGroups, 1); los_sum = zeros(nGroups, 1);
rule_conf_mean = zeros(nGroups, 1); ml_conf_mean = zeros(nGroups, 1);
res_rate = zeros(nGroups, 1);
ekf_rmse_mean = zeros(nGroups, 1); ekf_rmse_std = zeros(nGroups, 1);
mpc_time_mean = zeros(nGroups, 1); mpc_fbk_sum = zeros(nGroups, 1);

for g = 1:nGroups
    idx = (G == g);
    sc_col{g} = sc_u{g};
    cd_col{g} = cd_u{g};
    n_runs(g) = sum(idx);
    
    sub = T(idx, :);
    valid = strcmp(sub.status, 'PASS');
    if any(valid)
        v = sub(valid, :);
        minH_mean(g) = mean(v.minH_m);
        minH_std(g)  = std(v.minH_m);
        minH_min(g)  = min(v.minH_m);
        los_mean(g)  = mean(v.los_events);
        los_sum(g)   = sum(v.los_events);
        rule_conf_mean(g) = mean(v.rule_conflicts);
        ml_conf_mean(g)   = mean(v.ml_conflicts);
        
        tot_conf = sum(v.rule_conflicts);
        tot_res  = sum(v.rule_resolved);
        if tot_conf > 0, res_rate(g) = 100 * (tot_res / tot_conf); else, res_rate(g) = 100; end
        
        ekf_rmse_mean(g) = mean(v.ekf_pos_rmse_m);
        ekf_rmse_std(g)  = std(v.ekf_pos_rmse_m);
        mpc_time_mean(g) = mean(v.mpc_time_mean_ms);
        mpc_fbk_sum(g)   = sum(v.mpc_fallbacks);
    end
end

T_sum = table(sc_col, cd_col, n_runs, minH_mean, minH_std, minH_min, los_mean, los_sum, ...
    rule_conf_mean, ml_conf_mean, res_rate, ekf_rmse_mean, ekf_rmse_std, mpc_time_mean, mpc_fbk_sum, ...
    'VariableNames', {'Scenario', 'Condition', 'N_Runs', 'MinH_Mean_m', 'MinH_Std_m', 'MinH_Min_m', ...
    'LoS_Mean', 'LoS_Total', 'Rule_Conflicts_Mean', 'ML_Conflicts_Mean', 'Resolution_Success_Pct', ...
    'EKF_PosRMSE_Mean_m', 'EKF_PosRMSE_Std_m', 'MPC_Time_Mean_ms', 'MPC_Fallbacks_Total'});
end

function export_ml_metrics_csv(metrics, filepath)
models = {'Rule_Based', 'Logistic_Regression', 'Random_Forest'};
fields = {'accuracy', 'recall', 'precision', 'f1', 'far', 'specificity', 'brier', 'auc', 'lead_mean', 'latency_mean_us'};

rows = cell(3, numel(fields) + 1);
for m = 1:3
    switch m
        case 1, st = metrics.rule;
        case 2, st = metrics.logistic;
        case 3, st = metrics.random_forest;
    end
    rows{m, 1} = models{m};
    for f = 1:numel(fields)
        rows{m, f+1} = st.(fields{f});
    end
end

T_ml = cell2table(rows, 'VariableNames', ['Model_Name', fields]);
writetable(T_ml, filepath);
end

function generate_comparison_plots(T, model_pkg, out_dir)
set(0, 'DefaultFigureVisible', 'off');

% 1. Model Classification Comparison Bar Plot
try
    f1 = figure('Name', 'ML Model Benchmark', 'Position', [100 100 850 500], 'Color', 'w');
    ax1 = axes('Parent', f1);
    
    m_rule = model_pkg.metrics.rule;
    m_lr   = model_pkg.metrics.logistic;
    m_rf   = model_pkg.metrics.random_forest;
    
    metric_names = {'Accuracy', 'Recall', 'Precision', 'F1-Score', 'False Alarm Rate'};
    vals = [ ...
        m_rule.accuracy, m_rule.recall, m_rule.precision, m_rule.f1, m_rule.far; ...
        m_lr.accuracy,   m_lr.recall,   m_lr.precision,   m_lr.f1,   m_lr.far; ...
        m_rf.accuracy,   m_rf.recall,   m_rf.precision,   m_rf.f1,   m_rf.far ...
    ] * 100;
    
    b = bar(ax1, vals', 'grouped');
    b(1).FaceColor = [0.4 0.45 0.5];
    b(2).FaceColor = [0.15 0.60 0.90];
    b(3).FaceColor = [0.15 0.75 0.45];
    
    set(ax1, 'XTickLabel', metric_names, 'FontSize', 10);
    ylabel(ax1, 'Percentage (%)', 'FontSize', 11);
    title(ax1, 'AI/ML Conflict Predictor vs Rule-Based Baseline Performance', 'FontSize', 12, 'FontWeight', 'bold');
    legend(ax1, {'Rule-Based (Baseline)', 'Logistic Regression', 'Random Forest'}, 'Location', 'northeast');
    grid(ax1, 'on'); ylim(ax1, [0 115]);
    
    saveas(f1, fullfile(out_dir, 'fig_detector_vs_ml_metrics.png'));
    close(f1);
catch ME
    warning('Plot 1 generation failed: %s', ME.message);
end

% 2. Environmental Robustness Plot (Calm vs Moderate vs Severe)
try
    f2 = figure('Name', 'Environmental Robustness', 'Position', [100 100 900 500], 'Color', 'w');
    
    subplot(1, 2, 1);
    conds = {'calm', 'moderate', 'severe'};
    ekf_means = zeros(1, 3);
    for k = 1:3
        idx = strcmp(T.condition, conds{k}) & strcmp(T.status, 'PASS');
        if any(idx), ekf_means(k) = mean(T.ekf_pos_rmse_m(idx)); end
    end
    b2 = bar(ekf_means, 'FaceColor', [0.9 0.5 0.2]);
    set(gca, 'XTickLabel', {'Calm', 'Moderate', 'Severe'}, 'FontSize', 10);
    ylabel('Mean EKF Position RMSE [m]', 'FontSize', 11);
    title('EKF Tracking Error vs Weather Uncertainty', 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    
    subplot(1, 2, 2);
    minH_means = zeros(1, 3);
    for k = 1:3
        idx = strcmp(T.condition, conds{k}) & strcmp(T.status, 'PASS');
        if any(idx), minH_means(k) = mean(T.minH_m(idx)); end
    end
    b3 = bar(minH_means, 'FaceColor', [0.2 0.7 0.8]);
    set(gca, 'XTickLabel', {'Calm', 'Moderate', 'Severe'}, 'FontSize', 10);
    ylabel('Mean Minimum Separation [m]', 'FontSize', 11);
    title('Separation Margin vs Weather Uncertainty', 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    
    saveas(f2, fullfile(out_dir, 'fig_environmental_robustness.png'));
    close(f2);
catch ME
    warning('Plot 2 generation failed: %s', ME.message);
end

% 3. ROC Curve Comparison
try
    f3 = figure('Name', 'ROC Curves', 'Position', [100 100 650 550], 'Color', 'w');
    ax3 = axes('Parent', f3);
    hold(ax3, 'on');
    
    plot(ax3, [0 1], [0 1], 'k--', 'LineWidth', 1.2, 'DisplayName', 'Random Guess (AUC=0.50)');
    
    plot(ax3, m_rule.far, m_rule.recall, 's', 'MarkerSize', 10, 'MarkerFaceColor', [0.4 0.45 0.5], ...
        'Color', [0.2 0.2 0.2], 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Rule-Based (Sens=%.1f%%, FAR=%.1f%%)', 100*m_rule.recall, 100*m_rule.far));
        
    plot(ax3, m_lr.fpr_curve, m_lr.tpr_curve, 'Color', [0.15 0.60 0.90], 'LineWidth', 2.2, ...
        'DisplayName', sprintf('Logistic Regression (AUC=%.3f)', m_lr.auc));
        
    plot(ax3, m_rf.fpr_curve, m_rf.tpr_curve, 'Color', [0.15 0.75 0.45], 'LineWidth', 2.2, ...
        'DisplayName', sprintf('Random Forest (AUC=%.3f)', m_rf.auc));
        
    xlabel(ax3, 'False Positive Rate (1 - Specificity)', 'FontSize', 11);
    ylabel(ax3, 'True Positive Rate (Recall)', 'FontSize', 11);
    title(ax3, 'Receiver Operating Characteristic (ROC) Comparison', 'FontSize', 12, 'FontWeight', 'bold');
    legend(ax3, 'Location', 'southeast', 'FontSize', 9.5);
    grid(ax3, 'on'); axis(ax3, [0 1 0 1]);
    
    saveas(f3, fullfile(out_dir, 'fig_roc_curves.png'));
    close(f3);
catch ME
    warning('Plot 3 generation failed: %s', ME.message);
end

% 4. Early Warning Lead Time Distribution
try
    f4 = figure('Name', 'Lead Time Distribution', 'Position', [100 100 700 450], 'Color', 'w');
    ax4 = axes('Parent', f4);
    
    models = {'Rule-Based', 'Logistic Regression', 'Random Forest'};
    lead_means = [m_rule.lead_mean, m_lr.lead_mean, m_rf.lead_mean];
    lead_stds  = [m_rule.lead_std,  m_lr.lead_std,  m_rf.lead_std];
    
    bar_h = bar(ax4, lead_means, 0.5, 'FaceColor', [0.25 0.65 0.9]);
    hold(ax4, 'on');
    errorbar(ax4, 1:3, lead_means, lead_stds, 'k.', 'LineWidth', 1.5, 'CapSize', 12);
    
    set(ax4, 'XTickLabel', models, 'FontSize', 10);
    ylabel(ax4, 'Warning Lead Time Before CPA [seconds]', 'FontSize', 11);
    title(ax4, 'Conflict Warning Lead Time Before Closest Approach', 'FontSize', 12, 'FontWeight', 'bold');
    grid(ax4, 'on');
    
    saveas(f4, fullfile(out_dir, 'fig_lead_time_distribution.png'));
    close(f4);
catch ME
    warning('Plot 4 generation failed: %s', ME.message);
end

set(0, 'DefaultFigureVisible', 'on');
end
