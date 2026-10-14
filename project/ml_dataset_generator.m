function dataset = ml_dataset_generator(opts)
%ML_DATASET_GENERATOR  Generate reproducible, leakage-free training data for conflict prediction.
%
%   Generates pairwise aircraft encounters with varied geometries, speeds,
%   altitudes, sensor noise, and wind conditions. Ground-truth labels are established
%   from forward physical propagation without tactical intervention (open-loop baseline).
%
%   Usage:
%       dataset = ml_dataset_generator()
%       dataset = ml_dataset_generator(struct('n_encounters', 60, 'seed', 101))

if nargin < 1, opts = struct(); end
if ~isfield(opts, 'n_encounters'), opts.n_encounters = 60; end
if ~isfield(opts, 'seed'), opts.seed = 2026; end
if ~isfield(opts, 'T_lookahead'), opts.T_lookahead = 75; end % [s] prediction horizon
if ~isfield(opts, 'dt_sample'), opts.dt_sample = 1.0; end   % [s] feature sampling interval

rng(opts.seed);
H_sep = 1500; % [m] horizontal separation standard
V_sep = 300;  % [m] vertical separation standard
cfg_sep = struct('sep', struct('H', H_sep, 'V', V_sep));

fprintf('========================================================================================\n');
fprintf('                GENERATING CONFLICT PREDICTION DATASET                                  \n');
fprintf('========================================================================================\n');
fprintf('Target encounters: %d | Horizon: %.0f s | Sample rate: %.1f s | Seed: %d\n', ...
    opts.n_encounters, opts.T_lookahead, opts.dt_sample, opts.seed);

X_all = [];
y_all = [];
t_lead_all = [];
scen_all = {};
pair_all = [];

nCreated = 0;
totalTic = tic;

% Base scenario geometries and encounter parameter matrices
angles_deg = [0, 25, 45, 90, 135, 175, 180];
miss_dist_h = [0, 300, 600, 1100, 1600, 2400, 3500, 6000]; % [m]
miss_dist_v = [0, 150, 320, 600];                            % [m]
speeds = [85, 100, 115];                                     % [m/s]

nGeom = numel(angles_deg) * numel(miss_dist_h);
scenIdx = 0;

for k = 1:opts.n_encounters
    scenIdx = scenIdx + 1;
    % Pick geometry parameters with pseudo-random perturbations
    ang = angles_deg(mod(k-1, numel(angles_deg)) + 1) * (pi/180) + 0.05*(2*rand - 1);
    lat_off = miss_dist_h(mod(k-1, numel(miss_dist_h)) + 1) + 80*(2*rand - 1);
    vert_off = miss_dist_v(mod(k-1, numel(miss_dist_v)) + 1) * sign(rand - 0.5);
    
    v1 = speeds(mod(k-1, numel(speeds)) + 1) + 5*(2*rand - 1);
    v2 = speeds(mod(k, numel(speeds)) + 1) + 5*(2*rand - 1);
    
    % Geometry setup in North-East-Down:
    % AC1 starts at [0, 0, 3000] flying North (hdg = 0)
    % AC2 placed so they approach a mutual midpoint around t = 35..55s
    t_cpa_target = 40 + 20*rand;
    mid_N = 10000; mid_E = 0;
    
    p1_start = [mid_N - v1*t_cpa_target, lat_off/2, 3000];
    p1_end   = [p1_start(1) + v1*100, lat_off/2, 3000];
    
    % AC2 approaches at heading = ang
    % Velocity vector of AC2
    v2_N = v2 * cos(pi - ang);
    v2_E = v2 * sin(pi - ang);
    
    p2_start = [mid_N - v2_N*t_cpa_target, -lat_off/2 - v2_E*t_cpa_target, 3000 + vert_off];
    p2_end   = [p2_start(1) + v2_N*100, p2_start(2) + v2_E*100, 3000 + vert_off];
    
    % Build minimal synthetic flight trajectory
    T_sim = 80;
    dt = opts.dt_sample;
    time_vec = 0:dt:T_sim;
    N_steps = numel(time_vec);
    
    pos1 = zeros(N_steps, 3);
    pos2 = zeros(N_steps, 3);
    vel1 = zeros(N_steps, 3);
    vel2 = zeros(N_steps, 3);
    
    % Add gentle wind drift and sensor noise
    wind_drift = [3*randn, 3*randn, 0];
    
    for s = 1:N_steps
        t_curr = time_vec(s);
        pos1(s,:) = p1_start + [v1*t_curr, 0, 0] + wind_drift * (t_curr/T_sim);
        vel1(s,:) = [v1, 0, 0];
        
        pos2(s,:) = p2_start + [v2_N*t_curr, v2_E*t_curr, 0] + wind_drift * (t_curr/T_sim);
        vel2(s,:) = [v2_N, v2_E, 0];
    end
    
    % Find ground truth separation trajectory across entire flight
    d_h_traj = hypot(pos1(:,1) - pos2(:,1), pos1(:,2) - pos2(:,2));
    d_v_traj = abs(pos1(:,3) - pos2(:,3));
    
    is_los_traj = (d_h_traj < H_sep) & (d_v_traj < V_sep);
    has_los = any(is_los_traj);
    
    if has_los
        idx_los_first = find(is_los_traj, 1);
        t_los_actual = time_vec(idx_los_first);
    else
        t_los_actual = inf;
    end
    
    % Now sample observations at time t
    for s = 1:N_steps
        t_now = time_vec(s);
        
        % If already inside loss of separation, skip or record terminal
        if is_los_traj(s)
            continue;
        end
        
        % Check if a loss of separation will occur within T_lookahead
        time_to_los = t_los_actual - t_now;
        is_impending_conflict = (time_to_los > 0) && (time_to_los <= opts.T_lookahead);
        
        % Realistic noisy surveillance track (ADS-B / radar noise)
        meas_noise1 = [12*randn, 12*randn, 6*randn];
        meas_noise2 = [12*randn, 12*randn, 6*randn];
        vel_noise1  = [0.8*randn, 0.8*randn, 0.3*randn];
        vel_noise2  = [0.8*randn, 0.8*randn, 0.3*randn];
        
        tr_i = struct('pos', pos1(s,:)' + meas_noise1', 'vel', vel1(s,:)' + vel_noise1');
        tr_j = struct('pos', pos2(s,:)' + meas_noise2', 'vel', vel2(s,:)' + vel_noise2');
        
        % EKF covariance approximation
        P_cov = diag([0.5, 0.5, 0.5, 0.02, 0.02, 0.02, 1e-4, 1e-4, 1e-4, 25 + 5*randn^2, 25 + 5*randn^2, 10]);
        E_i = struct('P', P_cov);
        E_j = struct('P', P_cov);
        
        [feats, feat_names] = ml_feature_extractor(tr_i, tr_j, E_i, E_j, cfg_sep);
        
        % Filter out trivial long-distance non-interacting samples (> 35 km)
        if feats(1) > 35000
            continue;
        end
        
        X_all(end+1, :) = feats; %#ok<AGROW>
        y_all(end+1, 1) = double(is_impending_conflict); %#ok<AGROW>
        t_lead_all(end+1, 1) = max(0, min(opts.T_lookahead, time_to_los)); %#ok<AGROW>
        scen_all{end+1, 1} = sprintf('Encounter_%03d', k); %#ok<AGROW>
        pair_all(end+1, :) = [1, 2]; %#ok<AGROW>
    end
    nCreated = nCreated + 1;
end

elapsed = toc(totalTic);
nSamples = size(X_all, 1);
nConflicts = sum(y_all == 1);
nNonConflicts = sum(y_all == 0);
conflict_ratio = nConflicts / max(nSamples, 1);

fprintf('Generated %d encounter scenarios in %.2f s.\n', nCreated, elapsed);
fprintf('Total samples: %d | Conflicts: %d (%.1f%%) | Non-conflicts: %d (%.1f%%)\n', ...
    nSamples, nConflicts, 100*conflict_ratio, nNonConflicts, 100*(1 - conflict_ratio));
fprintf('Feature dimensionality: %d features\n', size(X_all, 2));
fprintf('========================================================================================\n\n');

dataset = struct( ...
    'X', X_all, ...
    'y', y_all, ...
    't_lead', t_lead_all, ...
    'feat_names', {feat_names}, ...
    'scenarios', {scen_all}, ...
    'pairs', pair_all, ...
    'n_samples', nSamples, ...
    'n_conflicts', nConflicts, ...
    'n_nonconflicts', nNonConflicts, ...
    'conflict_ratio', conflict_ratio, ...
    'lookahead_horizon_s', opts.T_lookahead ...
);
end
