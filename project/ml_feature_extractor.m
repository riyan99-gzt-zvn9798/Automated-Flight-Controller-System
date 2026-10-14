function [feats, feat_names] = ml_feature_extractor(tr_i, tr_j, E_i, E_j, cfg)
%ML_FEATURE_EXTRACTOR  Extract causal, uncertainty-aware features for conflict prediction.
%
%   INPUTS:
%       tr_i, tr_j : Surveillance track structures with .pos [N;E;alt] and .vel [vN;vE;vU]
%       E_i, E_j   : (Optional) EKF structures with .x (12 states) and .P (12x12 covariance)
%       cfg        : Configuration structure (containing cfg.sep.H, cfg.sep.V)
%
%   OUTPUTS:
%       feats      : [1 x 14] double vector of normalized/physical features
%       feat_names : {1 x 14} cell array of feature names
%
%   All features are strictly CAUSAL: computed solely from observations available at time t.

feat_names = { ...
    'dist_horiz_m', ...          % 1. Current horizontal distance [m]
    'dist_vert_m', ...           % 2. Current vertical distance [m]
    'dist_3d_m', ...             % 3. Current 3D Euclidean distance [m]
    'speed_rel_mps', ...         % 4. Relative speed ||v_j - v_i|| [m/s]
    'closing_speed_h_mps', ...   % 5. Horizontal closing speed [m/s] (+ closing, - diverging)
    'closing_speed_3d_mps', ...  % 6. 3D closing speed [m/s]
    'tcpa_s', ...                % 7. Time to Closest Point of Approach [s]
    'dcpa_h_m', ...              % 8. Estimated horizontal distance at CPA [m]
    'dcpa_v_m', ...              % 9. Estimated vertical distance at CPA [m]
    'heading_diff_rad', ...      % 10. Heading difference |wrapToPi(psi_i - psi_j)| [rad]
    'bearing_rel_rad', ...       % 11. Relative bearing of j from i [rad]
    'sep_norm_cpa', ...          % 12. Combined normalized separation at CPA max(dcpa_h/H, dcpa_v/V)
    'pos_uncertainty_m', ...     % 13. Combined EKF position uncertainty (trace of joint cov) [m]
    'speed_ratio' ...            % 14. Speed ratio Vi / (Vj + eps)
};

if nargin < 5 || isempty(cfg)
    H_sep = 1500; V_sep = 300;
else
    H_sep = cfg.sep.H; V_sep = cfg.sep.V;
end

% 1. Relative Position & Velocity
pos_i = tr_i.pos(:); pos_j = tr_j.pos(:);
vel_i = tr_i.vel(:); vel_j = tr_j.vel(:);

dr = pos_j - pos_i;       % [dN; dE; dAlt]
dv = vel_j - vel_i;       % [dvN; dvE; dvAlt]

dr_h = dr(1:2);
dv_h = dv(1:2);

d_h  = hypot(dr(1), dr(2));
d_v  = abs(dr(3));
d_3d = norm(dr);

v_rel = norm(dv);
v_rel_h = norm(dv_h);

% 2. Closing speeds (+ means converging, - means diverging)
if d_h > 1e-3
    vclose_h = -(dot(dr_h, dv_h)) / d_h;
else
    vclose_h = 0;
end

if d_3d > 1e-3
    vclose_3d = -(dot(dr, dv)) / d_3d;
else
    vclose_3d = 0;
end

% 3. Time to Closest Point of Approach (TCPA)
% Under constant relative velocity assumption:
% r(t) = dr + dv*t => d/dt ||r(t)||^2 = 2 (dr + dv*t) . dv = 0 => t_cpa = - (dr . dv) / ||dv||^2
dv_sq = dot(dv, dv);
if dv_sq > 1e-3
    tcpa_raw = - dot(dr, dv) / dv_sq;
    tcpa = max(0, min(180, tcpa_raw)); % Clamp to [0, 180s]
else
    tcpa = 0;
end

% 4. Separation at CPA
rcpa = dr + dv * tcpa;
dcpa_h = hypot(rcpa(1), rcpa(2));
dcpa_v = abs(rcpa(3));

% 5. Headings and Relative Bearing
hdg_i = atan2(vel_i(2), vel_i(1));
hdg_j = atan2(vel_j(2), vel_j(1));

hdg_diff = abs(wrapToPi_local(hdg_i - hdg_j));
bearing_ij = atan2(dr(2), dr(1));
bearing_rel = abs(wrapToPi_local(bearing_ij - hdg_i));

% 6. Normalized separation at CPA
sep_norm_cpa = max(dcpa_h / H_sep, dcpa_v / V_sep);

% 7. Uncertainty Metric (from EKF position covariance if available)
pos_uncert = 0;
if nargin >= 4 && ~isempty(E_i) && ~isempty(E_j) && isfield(E_i, 'P') && isfield(E_j, 'P')
    % EKF states: x=[u v w p q r phi th psi x y z] -> pos indices are 10,11,12
    if size(E_i.P, 1) >= 12 && size(E_j.P, 1) >= 12
        var_pos_i = trace(E_i.P(10:11, 10:11));
        var_pos_j = trace(E_j.P(10:11, 10:11));
        pos_uncert = sqrt(max(0, var_pos_i + var_pos_j));
    end
end
if pos_uncert == 0
    pos_uncert = 15.0; % Nominal baseline ADS-B/radar position uncertainty [m]
end

% 8. Speed ratio
vi_mag = norm(vel_i); vj_mag = norm(vel_j);
speed_ratio = vi_mag / max(vj_mag, 1.0);

% Assemble output feature vector
feats = [ ...
    d_h, ...
    d_v, ...
    d_3d, ...
    v_rel, ...
    vclose_h, ...
    vclose_3d, ...
    tcpa, ...
    dcpa_h, ...
    dcpa_v, ...
    hdg_diff, ...
    bearing_rel, ...
    sep_norm_cpa, ...
    pos_uncert, ...
    speed_ratio ...
];

end

function a = wrapToPi_local(a)
a = mod(a + pi, 2*pi) - pi;
end
