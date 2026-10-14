function model_pkg = ml_train_models(dataset, opts)
%ML_TRAIN_MODELS  Train, calibrate, evaluate, and benchmark ML conflict prediction models.
%
%   Trains:
%       1. Logistic Regression (Baseline with sigmoid link)
%       2. Random Forest / Ensemble (TreeBagger)
%       3. Rule-Based Geometric Detector (Baseline comparison)
%
%   Guarantees:
%       - Leakage prevention: Encounter-stratified train/test split
%       - Preprocessing fit strictly on training set only
%       - Full metric computation: TP, FP, TN, FN, Recall, Precision, F1, FPR, Brier score, AUC
%       - Early warning lead-time distribution and latency benchmarks
%
%   Usage:
%       model_pkg = ml_train_models()
%       model_pkg = ml_train_models(dataset, struct('train_ratio', 0.75, 'n_trees', 50))

if nargin < 1 || isempty(dataset)
    dataset = ml_dataset_generator(struct('n_encounters', 60, 'seed', 42));
end
if nargin < 2, opts = struct(); end
if ~isfield(opts, 'train_ratio'), opts.train_ratio = 0.75; end
if ~isfield(opts, 'n_trees'), opts.n_trees = 50; end
if ~isfield(opts, 'seed'), opts.seed = 2026; end
if ~isfield(opts, 'save_file'), opts.save_file = 'ml_conflict_model.mat'; end

rng(opts.seed);
fprintf('========================================================================================\n');
fprintf('                TRAINING AI/ML CONFLICT PREDICTION MODELS                               \n');
fprintf('========================================================================================\n');

% 1. Encounter-Stratified Train / Test Split
% Ensures contiguous time steps from the same encounter do not leak into both sets
unique_scens = unique(dataset.scenarios);
nScens = numel(unique_scens);
perm_scens = unique_scens(randperm(nScens));
nTrainScens = max(1, round(opts.train_ratio * nScens));

train_scen_names = perm_scens(1:nTrainScens);
test_scen_names  = perm_scens(nTrainScens+1:end);

is_train = ismember(dataset.scenarios, train_scen_names);
is_test  = ~is_train;

X_tr = dataset.X(is_train, :);
y_tr = dataset.y(is_train, :);
t_lead_tr = dataset.t_lead(is_train, :);

X_te = dataset.X(is_test, :);
y_te = dataset.y(is_test, :);
t_lead_te = dataset.t_lead(is_test, :);

fprintf('Train samples: %d (Conflicts: %d, %.1f%%) across %d encounters\n', ...
    size(X_tr,1), sum(y_tr), 100*mean(y_tr), nTrainScens);
fprintf('Test samples:  %d (Conflicts: %d, %.1f%%) across %d encounters\n', ...
    size(X_te,1), sum(y_te), 100*mean(y_te), nScens - nTrainScens);

% 2. Fit Preprocessor (Z-score standardizer) ONLY on Training Split
mu = mean(X_tr, 1);
sigma = std(X_tr, 0, 1);
sigma(sigma < 1e-6) = 1.0; % Prevent zero-division on constant features

X_tr_norm = (X_tr - mu) ./ sigma;
X_te_norm = (X_te - mu) ./ sigma;

scaler = struct('mu', mu, 'sigma', sigma, 'feat_names', {dataset.feat_names});

% 3. Train Model 1: Logistic Regression Baseline
fprintf('\n[1/3] Fitting Logistic Regression baseline...\n');
tLogTic = tic;
try
    % Use fitglm with penalized likelihood (Firth's method) to prevent separation
    mdl_lr = fitglm(X_tr_norm, y_tr, 'Distribution', 'binomial', 'Link', 'logit', ...
        'LikelihoodPenalty', 'jeffreys-prior');
    lr_predict_fn = @(x) predict(mdl_lr, (x - mu) ./ sigma);
catch ME
    warning('fitglm unavailable or failed (%s). Using robust regularized ridge logistic regression.', ME.message);
    % Robust built-in Newton-Raphson L2-regularized logistic regression fallback
    w = train_logistic_l2(X_tr_norm, y_tr, 1e-3);
    mdl_lr = struct('w', w, 'is_fallback', true);
    lr_predict_fn = @(x) 1 ./ (1 + exp(-[ones(size(x,1),1), (x - mu)./sigma] * w));
end
t_train_lr = toc(tLogTic);
fprintf('      Logistic Regression trained in %.3f s\n', t_train_lr);

% 4. Train Model 2: Random Forest / Ensemble
fprintf('[2/3] Fitting Random Forest classifier (%d trees)...\n', opts.n_trees);
tRfTic = tic;
try
    mdl_rf = TreeBagger(opts.n_trees, X_tr, y_tr, ...
        'Method', 'classification', ...
        'MinLeafSize', 5, ...
        'PredictorNames', dataset.feat_names);
    rf_predict_fn = @(x) predict_rf_prob(mdl_rf, x);
catch ME
    warning('TreeBagger unavailable (%s). Using gradient-boosted / ensemble fallback.', ME.message);
    mdl_rf = fitcensemble(X_tr, y_tr, 'Method', 'AdaBoostM1', 'NumLearningCycles', opts.n_trees);
    rf_predict_fn = @(x) predict_ensemble_prob(mdl_rf, x);
end
t_train_rf = toc(tRfTic);
fprintf('      Random Forest trained in %.3f s\n', t_train_rf);

% 5. Model 3: Rule-Based Geometric Detector Baseline
% Evaluates standard geometric rule on same test features:
% Conflict if normalized separation at CPA < 1.1 and TCPA <= 60s and closing speed > 0
fprintf('[3/3] Evaluating Rule-Based Geometric detector baseline...\n');
rule_predict_fn = @(x) eval_rule_baseline(x);

% 6. Comprehensive Test Evaluation
fprintf('\n========================================================================================\n');
fprintf('                MODEL EVALUATION & BENCHMARK ON TEST SET                                \n');
fprintf('========================================================================================\n');

% Benchmarking Inference Latency
nTest = size(X_te, 1);
lat_lr = zeros(min(nTest, 500), 1);
lat_rf = zeros(min(nTest, 500), 1);
lat_rule = zeros(min(nTest, 500), 1);

for i = 1:min(nTest, 500)
    row = X_te(i,:);
    t0 = tic; lr_predict_fn(row); lat_lr(i) = toc(t0)*1e6; % microseconds
    t0 = tic; rf_predict_fn(row); lat_rf(i) = toc(t0)*1e6;
    t0 = tic; rule_predict_fn(row); lat_rule(i) = toc(t0)*1e6;
end

p_lr_te = lr_predict_fn(X_te);
p_rf_te = rf_predict_fn(X_te);
p_rule_te = rule_predict_fn(X_te);

metrics_lr   = compute_classification_metrics(y_te, p_lr_te, 0.50, t_lead_te, lat_lr);
metrics_rf   = compute_classification_metrics(y_te, p_rf_te, 0.50, t_lead_te, lat_rf);
metrics_rule = compute_classification_metrics(y_te, p_rule_te, 0.50, t_lead_te, lat_rule);

% Display Detailed Performance Comparison Table
fprintf('%-22s | %-6s | %-7s | %-6s | %-6s | %-6s | %-6s | %-8s | %-10s\n', ...
    'Model', 'Acc(%)', 'Sens(%)', 'Prec(%)', 'F1', 'FAR(%)', 'Brier', 'AUC', 'Latency[us]');
fprintf('----------------------------------------------------------------------------------------\n');
print_metric_row('Rule-Based (Baseline)', metrics_rule);
print_metric_row('Logistic Regression', metrics_lr);
print_metric_row('Random Forest', metrics_rf);
fprintf('----------------------------------------------------------------------------------------\n\n');

% Confusion Matrices
fprintf('CONFUSION MATRICES (Threshold = 0.50):\n');
fprintf('  Rule-Based:          TP=%4d | FP=%4d | TN=%4d | FN=%4d\n', ...
    metrics_rule.tp, metrics_rule.fp, metrics_rule.tn, metrics_rule.fn);
fprintf('  Logistic Regression: TP=%4d | FP=%4d | TN=%4d | FN=%4d\n', ...
    metrics_lr.tp, metrics_lr.fp, metrics_lr.tn, metrics_lr.fn);
fprintf('  Random Forest:       TP=%4d | FP=%4d | TN=%4d | FN=%4d\n\n', ...
    metrics_rf.tp, metrics_rf.fp, metrics_rf.tn, metrics_rf.fn);

fprintf('EARLY WARNING LEAD TIME (True Positives, Mean +/- Std):\n');
fprintf('  Rule-Based:          %.1f +/- %.1f s\n', metrics_rule.lead_mean, metrics_rule.lead_std);
fprintf('  Logistic Regression: %.1f +/- %.1f s\n', metrics_lr.lead_mean, metrics_lr.lead_std);
fprintf('  Random Forest:       %.1f +/- %.1f s\n\n', metrics_rf.lead_mean, metrics_rf.lead_std);

% Assemble Model Package
model_pkg = struct( ...
    'models', struct('logistic', mdl_lr, 'random_forest', mdl_rf), ...
    'predict_fns', struct('logistic', lr_predict_fn, 'random_forest', rf_predict_fn, 'rule', rule_predict_fn), ...
    'scaler', scaler, ...
    'metrics', struct('logistic', metrics_lr, 'random_forest', metrics_rf, 'rule', metrics_rule), ...
    'test_data', struct('X', X_te, 'y', y_te, 't_lead', t_lead_te, 'p_lr', p_lr_te, 'p_rf', p_rf_te, 'p_rule', p_rule_te), ...
    'opts', opts, ...
    'trained_at', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')) ...
);

% Save artifact
save(opts.save_file, 'model_pkg', '-v7.3');
fprintf('Model package and preprocessor successfully saved to: %s\n', opts.save_file);
fprintf('========================================================================================\n\n');
end

%% ==================== HELPER FUNCTIONS ====================

function p = predict_rf_prob(mdl, X)
[~, scores] = predict(mdl, X);
p = scores(:, 2);
end

function p = predict_ensemble_prob(mdl, X)
[~, scores] = predict(mdl, X);
p = scores(:, 2);
end

function p = eval_rule_baseline(X)
% Feature indices from ml_feature_extractor:
% 5: closing_speed_h_mps, 7: tcpa_s, 12: sep_norm_cpa
sep_norm_cpa = X(:, 12);
tcpa = X(:, 7);
vclose_h = X(:, 5);

% Rule: conflict predicted if sep_norm_cpa <= 1.1 AND tcpa <= 60s AND vclose_h > 0
is_rule = (sep_norm_cpa <= 1.1) & (tcpa <= 60) & (vclose_h > 0);
p = double(is_rule);
end

function m = compute_classification_metrics(y_true, y_prob, thresh, t_lead, lat_vec)
y_pred = double(y_prob >= thresh);

tp = sum((y_pred == 1) & (y_true == 1));
fp = sum((y_pred == 1) & (y_true == 0));
tn = sum((y_pred == 0) & (y_true == 0));
fn = sum((y_pred == 0) & (y_true == 1));

nTotal = numel(y_true);
acc = (tp + tn) / max(nTotal, 1);
sens = tp / max(tp + fn, 1);       % Recall
prec = tp / max(tp + fp, 1);       % Precision
if (prec + sens) > 0
    f1 = 2 * (prec * sens) / (prec + sens);
else
    f1 = 0;
end
far = fp / max(fp + tn, 1);        % False Alarm Rate (FPR)
spec = tn / max(tn + fp, 1);       % Specificity
brier = mean((y_prob - y_true).^2); % Brier calibration score

% Approximate ROC AUC
[~, ord] = sort(y_prob, 'descend');
y_sorted = y_true(ord);
cum_tp = cumsum(y_sorted == 1);
cum_fp = cumsum(y_sorted == 0);
tpr_curve = [0; cum_tp / max(sum(y_true == 1), 1)];
fpr_curve = [0; cum_fp / max(sum(y_true == 0), 1)];
auc = trapz(fpr_curve, tpr_curve);

% Lead-time statistics for true positives
tp_mask = (y_pred == 1) & (y_true == 1);
if any(tp_mask)
    leads = t_lead(tp_mask);
    leads = leads(isfinite(leads) & leads > 0);
    if ~isempty(leads)
        lead_mean = mean(leads);
        lead_std  = std(leads);
    else
        lead_mean = 0; lead_std = 0;
    end
else
    lead_mean = 0; lead_std = 0;
end

m = struct( ...
    'tp', tp, 'fp', fp, 'tn', tn, 'fn', fn, ...
    'accuracy', acc, ...
    'recall', sens, ...
    'precision', prec, ...
    'f1', f1, ...
    'far', far, ...
    'specificity', spec, ...
    'brier', brier, ...
    'auc', auc, ...
    'lead_mean', lead_mean, ...
    'lead_std', lead_std, ...
    'latency_mean_us', mean(lat_vec), ...
    'latency_max_us', max(lat_vec), ...
    'threshold', thresh, ...
    'fpr_curve', fpr_curve, ...
    'tpr_curve', tpr_curve ...
);
end

function print_metric_row(name, m)
fprintf('%-22s | %6.1f | %7.1f | %6.1f | %6.3f | %6.1f | %6.4f | %8.4f | %7.1f us\n', ...
    name, 100*m.accuracy, 100*m.recall, 100*m.precision, m.f1, 100*m.far, ...
    m.brier, m.auc, m.latency_mean_us);
end

function w = train_logistic_l2(X, y, lambda)
% Robust Newton-Raphson L2-regularized logistic regression
N = size(X, 1);
X_ext = [ones(N, 1), X];
D = size(X_ext, 2);
w = zeros(D, 1);
max_iter = 30;
tol = 1e-5;

for iter = 1:max_iter
    p = 1 ./ (1 + exp(-X_ext * w));
    p = min(max(p, 1e-12), 1 - 1e-12);
    W_diag = p .* (1 - p);
    
    grad = (X_ext' * (p - y)) / N + lambda * [0; w(2:end)];
    H = (X_ext' * (W_diag .* X_ext)) / N + lambda * diag([0; ones(D-1, 1)]);
    
    step = H \ grad;
    w = w - step;
    if norm(step) < tol
        break;
    end
end
end
