function results = UnifiedFlightRadar3D(mode, arg2, arg3)
%UNIFIEDFLIGHTRADAR3D  Single-file package: autonomous ATC / flight-control simulation
%  (live 3-D airspace) + 3-D aircraft radar command-center dashboard.
%  ALL DATA IS SIMULATED. Research / demo prototype - NOT a certified aviation system.
%
%  USAGE (save this file as UnifiedFlightRadar3D.m, then):
%    >> UnifiedFlightRadar3D                    % ATC simulation, head_on scenario, live 3-D figure
%    >> UnifiedFlightRadar3D('crossing')        % any scenario name (see list below)
%    >> UnifiedFlightRadar3D('radar')           % 3-D radar command-center dashboard only
%    >> UnifiedFlightRadar3D('both')            % radar dashboard + ATC simulation together
%    >> UnifiedFlightRadar3D('both','crossing') % both, with a chosen ATC scenario
%    >> UnifiedFlightRadar3D('all')             % every scenario, no graphics, text reports
%    >> UnifiedFlightRadar3D('montecarlo')      % Monte-Carlo batch, no graphics
%    >> UnifiedFlightRadar3D('head_on', struct('T',120))   % override any default_config field
%  Scenarios: nominal, head_on, crossing, overtaking, crosswind, gps_failure, engine_failure,
%    multi_conflict, restricted_zone, sensor_noise, actuator_degradation, comm_loss,
%    altitude_bank_deviation, dense_airspace (8 aircraft), converging_hub (6 aircraft)
%
%  Requires MATLAB R2021a or newer (uifigure/uiaxes for the radar dashboard). No toolboxes needed.
%  Live ATC figure keys: 1..9 focus aircraft | c camera (fixed/follow/orbit/top/auto) | r radar | t telemetry | p predictions |
%    m performance | g attitude indicator (PFD) | a animation | space pause | x stop.   Mouse rotates/zooms the 3-D axes.
if nargin < 1 || isempty(mode), mode = 'head_on'; end
results = [];
switch lower(mode)
    case 'radar'
        AircraftRadar3D();
    case 'dashboard'
        if nargin >= 2 && ~isempty(arg2), res_arg = arg2; else, res_arg = []; end
        results = performance_dashboard(res_arg);
    case 'experiment'
        exp_opts = struct();
        if nargin >= 2 && ~isempty(arg2), exp_opts = arg2; end
        results = experiment_runner(exp_opts);
    case 'train'
        results = ml_train_models();
    case 'test'
        results = run_regression_tests();
    case 'both'
        sc = 'head_on'; op = struct();
        if nargin >= 2 && ~isempty(arg2), sc = arg2; end
        if nargin >= 3 && ~isempty(arg3), op = arg3; end
        AircraftRadar3D();
        drawnow;
        results = unified_atc_system(sc, op);
    otherwise
        op = struct();
        if nargin >= 2 && ~isempty(arg2), op = arg2; end
        results = unified_atc_system(mode, op);
end
end

%% ######################################################################
%  PART 1 - UNIFIED FLIGHT CONTROL & AUTONOMOUS ATC SIMULATION
%% ######################################################################
function results = unified_atc_system(scenario, opts)
%UNIFIED_ATC_SYSTEM  Complete Unified Aircraft Flight Control & Autonomous ATC System
%  RESEARCH / SIMULATION PROTOTYPE - NOT a certified aviation safety system.
%
%  USAGE
%    unified_atc_system                       % head-on scenario, live 3D interface
%    unified_atc_system('crossing')           % any scenario from scenario_list()
%    unified_atc_system('all')                % every scenario, no graphics, test reports
%    unified_atc_system('montecarlo', struct('mc', struct('trials',20)))
%    unified_atc_system('head_on', struct('T',120, 'viz', struct('enable',false)))
%  Any field of default_config() can be overridden through OPTS (nested structs merge).
%
%  SCENARIOS: nominal, head_on, crossing, overtaking, crosswind, gps_failure,
%    engine_failure, multi_conflict, restricted_zone, sensor_noise,
%    actuator_degradation, comm_loss, altitude_bank_deviation, dense_airspace (8 aircraft),
%    converging_hub (6 aircraft)
%
%  LIVE FIGURE KEYS: 1..9 focus aircraft | c camera mode (fixed/follow/orbit/top/auto)
%    r radar | t telemetry | p predictions | m performance panels | a animation on/off
%    space pause | x stop simulation.  Mouse rotate/zoom works in the 3D axes.
%
%  ARCHITECTURE (every stage is a callable local function)
%    initialize_simulation/initialize_aircraft/initialize_sensors/initialize_airspace
%    -> per step: wind_model, sensor_model, estimator_step (ekf_predict + ekf_update_*),
%       emergency_manager, waypoint_manager, trajectory_predictor, mpc_controller
%       (build_mpc_prediction_matrices, quadprog or toolbox-free QP, guidance fallback),
%       inner-loop autopilot, actuator_model, aircraft_dynamics (nasa_wind_dynamics,
%       nasa_12state_dynamics, rk4_step)
%    -> every dt_atc: adsb_model/radar_model tracks, conflict_detector_4d,
%       conflict_resolver, trajectory_replanner, restricted_zone_checker
%    -> visualization: update_3d_visualization, update_radar_display, update_telemetry,
%       update_perf_panels; metrics: calculate_performance_metrics, generate_test_report,
%       run_monte_carlo.
%
%  COORDINATES, STATES, UNITS, SIGN CONVENTIONS
%    State x = [u v w p q r phi theta psi x y h]'   (12 states, SI units, radians)
%     u,v,w  inertial velocity components in the BODY frame [m/s]
%            (body x forward, y right, z down)
%     p,q,r  body angular rates [rad/s]; phi,theta,psi = roll, pitch, yaw (heading from North,
%            clockwise positive) [rad], 3-2-1 sequence; body->NED DCM = Rz(psi)*Ry(theta)*Rx(phi)
%     x = North [m], y = East [m], h = altitude above ground, positive UP [m]
%    Controls u = [throttle(0..1); elevator; aileron; rudder] [rad]. Positive elevator = trailing
%     edge down (nose down), positive aileron = positive roll moment, positive rudder = nose-left.
%    Wind is a NED vector [N;E;D] (m/s) of the air mass over ground. Airspeed = body velocity -
%     R_ned2body*wind. Plot frame: X=East, Y=North, Z=altitude (km); mesh rotation uses
%     Rplot = P*Rz*Ry*Rx with P = [0 1 0;1 0 0;0 0 -1] (NED->ENU, proper rotation).
%
%  ASSUMPTIONS / SIMPLIFICATIONS
%    * Medium twin-engine transport, linear aero coefficients with alpha clamp and CLmax cap,
%      ISA density, flat non-rotating earth, constant gravity, wind treated as locally constant
%      (no wind-gradient acceleration), first-order Dryden-like turbulence.
%    * A cascaded design is used: MPC (LTV, 7-state guidance model z=[x y h psi V phi gamma],
%      inputs [phi_cmd gamma_cmd V_cmd]) tracks the conflict-free reference; an inner-loop autopilot
%      (attitude/rate + throttle PI) converts the commands to surface deflections which pass through
%      the actuator model. The 12-state nonlinear model is the plant and the EKF process model.
%    * EKF: IMU(gyro direct, accelerometer through model specific force), GPS pos+vel, barometer,
%      magnetometer, airspeed (wind from the known mean-wind model). Chi-square gating, Joseph form,
%      symmetrization, positive-definiteness repair. GPS-denied: ATC radar position uplink at low weight.
%    * ATC uses ADS-B + radar tracks (alpha-beta), ground-speed constant 4D prediction along the
%      flight plan with a bank-limited direct-to navigation model, cylinder separation criteria.
%    * Conflict classes (cfg.sep): loss predicted <= T_crit -> CRITICAL, <= T_warn -> WARNING, otherwise
%      NORMAL. Separation = horizontal H and vertical V simultaneously (loss when both are violated).
%    * Toolboxes: none required. quadprog (Optimization Toolbox) is used when available, otherwise a
%      built-in active-set box QP. Needs MATLAB R2016b+ (implicit expansion, local functions in scripts).
%    * Limitations: no terrain, no weather cells, simplified ATC procedures, no pilot/ATC latency model,
%      MPC model is reduced-order, results are not validated against real aircraft or procedures.

if nargin < 1 || isempty(scenario), scenario = 'head_on'; end
if nargin < 2 || isempty(opts), opts = struct(); end
cfg0 = merge_struct(default_config(), opts);
userViz = isfield(opts,'viz') && isstruct(opts.viz) && isfield(opts.viz,'enable');
switch lower(scenario)
    case 'all'
        if ~userViz, cfg0.viz.enable = false; end
        names = scenario_list(); results = cell(1, numel(names));
        for k = 1:numel(names)
            cfg = apply_scenario(default_config(), names{k});
            cfg = merge_struct(cfg, opts);
            if ~userViz, cfg.viz.enable = false; end
            results{k} = run_simulation(cfg);
            fprintf('%s\n', results{k}.report);
        end
        print_batch_summary(results);
    case 'montecarlo'
        if ~userViz, cfg0.viz.enable = false; end
        results = run_monte_carlo(cfg0);
    otherwise
        cfg = apply_scenario(default_config(), scenario);
        cfg = merge_struct(cfg, opts);
        R = run_simulation(cfg);
        fprintf('%s\n', R.report);
        print_final_states(R.W);
        results = R;
end
end

%% ======================================================================
%  CONFIGURATION
%% ======================================================================
function cfg = default_config()
d2r = pi/180;
cfg.scenario = 'head_on'; cfg.seed = 1;
cfg.dt = 0.02; cfg.T = 170; cfg.dt_mpc = 0.5; cfg.dt_atc = 1.0; cfg.dt_em = 0.1; cfg.dt_log = 0.1;
cfg.g = 9.80665; cfg.report_file = '';
cfg.modes = {'NORMAL','WARNING','CONFLICT_AVOIDANCE','EMERGENCY','ENGINE_FAILURE', ...
             'GPS_FAILURE','COMMUNICATION_LOSS','ALTITUDE_DEVIATION','EXCESSIVE_BANK'};
% ---- aircraft (medium twin-engine transport) ----
p.m = 8000; p.S = 30; p.b = 14; p.c = 2.3;
p.Ixx = 15000; p.Iyy = 30000; p.Izz = 40000; p.Ixz = 1200;
p.CL0 = 0.30; p.CLa = 5.0; p.CLq = 7.0; p.CLde = 0.40; p.CLmax = 1.6;
p.CD0 = 0.025; p.k = 0.045;
p.CYb = -0.9; p.CYp = 0; p.CYr = 0.3; p.CYdr = 0.2;
p.Clb = -0.10; p.Clp = -0.45; p.Clr = 0.10; p.Clda = 0.20; p.Cldr = 0.03;
p.Cm0 = 0.05; p.Cma = -1.2; p.Cmq = -18; p.Cmde = -1.3;
p.Cnb = 0.15; p.Cnp = -0.05; p.Cnr = -0.20; p.Cnda = -0.01; p.Cndr = -0.10;
p.Tmax = 40000; p.thrust_lapse = 0.8; p.yeng = 3.0; p.eh = [1 1];
cfg.ac = p;
% ---- limits ----
cfg.lim = struct('phi_cmd',25*d2r,'phi_hard',60*d2r,'gam',0.10,'Vcmd_min',68,'Vcmd_max',130, ...
    'Vstall_warn',58,'alpha_max',0.26,'h_min',300,'h_max',9000, ...
    'rate_phi',0.35,'rate_gam',0.12,'rate_V',3.0);
% ---- navigation / prediction model ----
cfg.nav = struct('Kpsi',0.5,'phi_nav',22*d2r,'Kh',0.08,'climb_max',8,'acc_max',2,'capture',1200);
% ---- separation / ATC ----
cfg.sep = struct('H',1500,'V',300,'horizon',90,'dtp',1,'T_crit',30,'T_warn',60, ...
    'detect_factor',1.1,'margin',1.25,'zone_buf',500,'T_zone',90,'T_clear',8,'hold',12);
cfg.res = struct('dpsi',[-60 -45 -30 -20 -10 10 20 30 45 60]*d2r,'dalt',[-600 -300 0 300 600], ...
    'dv',[-15 0 15],'L_min',3500,'L_time',40,'w_sep',1000,'w_route',2,'w_alt',1,'w_spd',0.5, ...
    'w_man',0.8,'w_zone',50,'max_iter',6);
% ---- sensors ----
cfg.sens = struct('noise_scale',1,'gps_rate',5,'gps_sig_pos',3,'gps_sig_alt',5,'gps_sig_vel',0.3, ...
    'gps_bias_rw',0.02,'gyro_sig',0.003,'gyro_bias0',0.002,'gyro_bias_rw',2e-5, ...
    'accel_sig',0.08,'accel_bias0',0.05,'baro_rate',10,'baro_sig',1.5,'baro_bias_rw',0.05, ...
    'mag_rate',10,'mag_sig',0.02,'mag_bias0',0.01,'asp_rate',10,'asp_sig',0.6,'asp_bias0',0.3);
cfg.adsb  = struct('rate',1,'sig_pos',12,'sig_alt',8,'sig_gs',1.0,'sig_hdg',1*d2r,'sig_vs',0.5,'p_drop',0.02);
cfg.radar = struct('pos',[12000 0],'range',40000,'rate',0.25,'sig_r',40,'sig_az',0.4*d2r, ...
    'sig_alt',30,'p_detect',0.97,'sweep_rate',2*pi/4);
% ---- EKF ----
cfg.ekf = struct('Qc',diag([0.5 0.5 0.5 0.02 0.02 0.02 2e-4 2e-4 2e-4 0.5 0.5 0.5]), ...
    'P0',diag([1 1 1 0.02 0.02 0.02 0.01 0.01 0.02 25 25 25]), ...
    'init_err',[0.5 0.5 0.5 0.01 0.01 0.01 0.01 0.01 0.02 8 8 8]', ...
    'R_inflate',1.5,'gyro_sig',0.004,'accel_sig',1.5,'accel_div',5,'gps_denied_T',3,'uplink_sig',80);
% ---- wind (blows FROM dir_from, radians clockwise from North) ----
cfg.wind = struct('speed',5,'dir_from',270*d2r,'href',1000,'shear',0.14,'turb',1.0, ...
    'sig_u',1.0,'sig_v',1.0,'sig_w',0.7,'L',500);
% ---- MPC (guidance-level LTV model) ----
cfg.mpc = struct('N',16,'tau_phi',0.5,'tau_gam',1.2,'tau_V',4.0, ...
    'Qy',[1/60^2 1/60^2 1/25^2 1/(6*d2r)^2 1/3^2],'Ru',[2 5 0.02],'Su',[8 20 0.1], ...
    'term_factor',3,'max_lag',1500,'use_quadprog',true);
% ---- inner-loop autopilot ----
cfg.ic = struct('kth',2.0,'kq',0.6,'kphi',1.2,'kp',0.4,'kdr',1.0,'Kv',0.04,'Ki',0.01,'int_lim',0.3);
% ---- actuators: [throttle; elevator; aileron; rudder] ----
cfg.act = struct('lo',[0;-0.35;-0.35;-0.45],'hi',[1;0.35;0.35;0.45], ...
    'tau',[0.5;0.08;0.08;0.10],'rate',[0.4;1.2;1.2;1.0]);
% ---- emergency thresholds ----
cfg.em = struct('dwell',2,'alt_dev',150,'alt_t',3,'bank_on',40*d2r,'bank_off',20*d2r,'crit_t',10);
% ---- visualization ----
cfg.viz = struct('enable',true,'animate',true,'update_every',3,'trail_len',600,'show_pred',true, ...
    'show_radar',true,'show_hud',true,'show_perf',true,'show_pfd',true,'show_sep',true,'show_wind',true, ...
    'camera','fixed','ac_scale',180,'z_exag',2,'ext',[-14 14 -2 26 6],'vel_vec_sec',15, ...
    'radar_range_km',30,'max_fail',3, ...
    'trail_segs',6,'nav_lights',true,'radar_sweep3d',true,'altitude_planes',true,'alert_banner',true);
cfg.mc = struct('trials',10,'seed0',100,'scenarios',{{'head_on','crossing','overtaking','multi_conflict','crosswind'}}, ...
    'T',120);
cfg.ml = struct('enable', true, 'model', 'logistic', 'threshold', 0.50, 'show_ai', true);
cfg.fail = mkfail(); cfg.zones = []; cfg.aircraft = [];
end

function names = scenario_list()
names = {'nominal','head_on','crossing','overtaking','crosswind','gps_failure','engine_failure', ...
    'multi_conflict','restricted_zone','sensor_noise','actuator_degradation','comm_loss', ...
    'altitude_bank_deviation','dense_airspace','converging_hub'};
end

function s = mkac(name, route, V, col)
s = struct('name',name,'route',route,'V',V,'color',col,'p',[]);
end

function f = mkfail(type, ac, t_on, t_off, val)
if nargin == 0
    f = struct('type',{},'ac',{},'t_on',{},'t_off',{},'val',{});
else
    f = struct('type',type,'ac',ac,'t_on',t_on,'t_off',t_off,'val',val);
end
end

function z = mkzone(name, type, center, radius, poly, hmin, hmax)
z = struct('name',name,'type',type,'center',center,'radius',radius,'poly',poly, ...
           'hmin',hmin,'hmax',hmax,'bc',[0 0],'br',0);
end

function cfg = apply_scenario(cfg, name)
d2r = pi/180;
cfg.scenario = name; cfg.fail = mkfail(); cfg.zones = [];
col = [0.2 0.8 1; 1 0.65 0.2; 0.75 0.55 1; 0.45 1 0.5; 1 0.45 0.75; ...
       1 1 0.4; 0.3 0.9 0.8; 0.45 0.6 1; 0.75 1 0.3; 1 0.6 0.55];
A = @(k) col(mod(k-1,size(col,1))+1,:);
switch lower(name)
    case 'nominal'
        cfg.aircraft = [mkac('AC1',[0 0 3000; 24000 0 3000],100,A(1)); ...
                        mkac('AC2',[0 7000 3500; 24000 7000 3500],100,A(2))];
        cfg.T = 150;
    case 'head_on'
        cfg.aircraft = [mkac('AC1',[0 0 3000; 24000 0 3000],100,A(1)); ...
                        mkac('AC2',[24000 400 3000; 0 400 3000],100,A(2))];
    case 'crossing'
        cfg.aircraft = [mkac('AC1',[0 0 3000; 22000 0 3000],100,A(1)); ...
                        mkac('AC2',[11000 -11000 3000; 11000 11000 3000],100,A(2))];
    case 'overtaking'
        cfg.aircraft = [mkac('AC1',[3500 0 3000; 24000 0 3000],80,A(1)); ...
                        mkac('AC2',[0 100 3000; 24000 100 3000],115,A(2))];
    case 'crosswind'
        cfg.aircraft = [mkac('AC1',[0 0 3000; 10000 0 3000; 10000 10000 3000; 20000 10000 3000],100,A(1)); ...
                        mkac('AC2',[0 -8000 3600; 20000 -8000 3600],100,A(2))];
        cfg.wind.speed = 18; cfg.wind.dir_from = 90*d2r; cfg.wind.turb = 1.8;
    case 'gps_failure'
        cfg.aircraft = [mkac('AC1',[0 0 3000; 24000 3000 3000],100,A(1)); ...
                        mkac('AC2',[0 8000 3500; 24000 8000 3500],100,A(2))];
        cfg.fail(end+1) = mkfail('gps',1,40,inf,[]); cfg.T = 150;
    case 'engine_failure'
        cfg.aircraft = [mkac('AC1',[0 0 3000; 24000 0 3000],100,A(1)); ...
                        mkac('AC2',[0 6000 3000; 24000 6000 3000],100,A(2))];
        cfg.fail(end+1) = mkfail('engine',1,35,inf,[1 0.35]); cfg.T = 150;
    case 'multi_conflict'
        cfg.aircraft = [mkac('AC1',[0 0 3000; 24000 0 3000],100,A(1)); ...
                        mkac('AC2',[24000 0 3000; 0 0 3000],100,A(2)); ...
                        mkac('AC3',[12000 -12000 3000; 12000 12000 3000],100,A(3)); ...
                        mkac('AC4',[12000 12000 3000; 12000 -12000 3000],100,A(4))];
        cfg.T = 190;
    case 'restricted_zone'
        cfg.aircraft = [mkac('AC1',[0 0 3000; 24000 0 3000],100,A(1)); ...
                        mkac('AC2',[0 -5000 3000; 24000 6000 3000],100,A(2))];
        cfg.zones = [mkzone('ZONE-A','cylinder',[12000 300],2800,[],0,6000); ...
                     mkzone('ZONE-B','polygon',[0 0],0,[7000 -6500; 10000 -6500; 10000 -3500; 7000 -3500],0,7000); ...
                     mkzone('ZONE-C','circle',[18000 5000],1500,[],-inf,inf)];
        cfg.T = 190;
    case 'sensor_noise'
        cfg.aircraft = [mkac('AC1',[0 0 3000; 22000 0 3000],100,A(1)); ...
                        mkac('AC2',[11000 -11000 3000; 11000 11000 3000],100,A(2))];
        cfg.fail(end+1) = mkfail('noise',0,0,inf,4);
    case 'actuator_degradation'
        cfg.aircraft = [mkac('AC1',[0 0 3000; 10000 0 3000; 10000 8000 3000; 20000 8000 3000],100,A(1)); ...
                        mkac('AC2',[0 -7000 3500; 20000 -7000 3500],100,A(2))];
        cfg.fail(end+1) = mkfail('actuator',1,20,inf,[0.5 0.6 0.7 0.5]);
    case 'comm_loss'
        cfg.aircraft = [mkac('AC1',[0 0 3000; 24000 0 3000],100,A(1)); ...
                        mkac('AC2',[24000 400 3000; 0 400 3000],100,A(2))];
        cfg.fail(end+1) = mkfail('comm',2,20,inf,[]);
    case 'altitude_bank_deviation'
        cfg.aircraft = [mkac('AC1',[0 0 3000; 24000 0 3000],100,A(1)); ...
                        mkac('AC2',[0 7000 3500; 24000 7000 3500],100,A(2))];
        cfg.fail(end+1) = mkfail('bank_upset',1,30,inf,[0.30 1.5]);
        cfg.fail(end+1) = mkfail('alt_upset',1,70,inf,[0.25 2.5]); cfg.T = 150;
    case 'dense_airspace'
        % 8 aircraft: head-on pair, crossing, two opposing diagonals, overtaking pair, extra southbound
        cfg.aircraft = [mkac('AC1',[0 0 3000; 24000 0 3000],100,A(1)); ...
                        mkac('AC2',[24000 300 3000; 0 300 3000],100,A(2)); ...
                        mkac('AC3',[12000 -12000 3000; 12000 12000 3000],100,A(3)); ...
                        mkac('AC4',[0 -9000 3000; 24000 9000 3000],100,A(4)); ...
                        mkac('AC5',[0 9000 3000; 24000 -9000 3000],100,A(5)); ...
                        mkac('AC6',[3000 -5000 3000; 24000 -5000 3000],80,A(6)); ...
                        mkac('AC7',[0 -5000 3000; 24000 -5000 3000],115,A(7)); ...
                        mkac('AC8',[24000 5000 3000; 0 5000 3000],100,A(8))];
        cfg.T = 200;
    case 'converging_hub'
        % 6 aircraft on a star pattern, all passing through one hub point at two flight levels
        hub = [12000 0]; Rh = 11000; nHub = 6; spdHub = [100 96 104 98 102 100]; acl = [];
        for kh = 1:nHub
            thh = (kh-1)*2*pi/nHub; alth = 3000 + 600*mod(kh-1,2);
            p0 = hub + Rh*[cos(thh) sin(thh)]; p1 = hub - Rh*[cos(thh) sin(thh)];
            acl = [acl; mkac(sprintf('AC%d',kh),[p0 alth; p1 alth],spdHub(kh),A(kh))]; %#ok<AGROW>
        end
        cfg.aircraft = acl; cfg.T = 170;
    otherwise
        error('Unknown scenario "%s". Valid: %s', name, strjoin(scenario_list(), ', '));
end
end

%% ======================================================================
%  SIMULATION DRIVER
%% ======================================================================
function R = run_simulation(cfg)
rng(cfg.seed);
W = initialize_simulation(cfg);
G = [];
if cfg.viz.enable
    try
        G = init_visualization(W, cfg);
    catch ME
        warning('Visualization disabled (init failed): %s', ME.message);
        G = [];
    end
end
N = round(cfg.T/cfg.dt); vfail = 0; tWall = tic;
for k = 1:N
    t = (k-1)*cfg.dt; W.t = t; W.k = k;
    W = step_world(W, t);
    if ~isempty(G) && mod(k-1, cfg.viz.update_every) == 0
        try
            [G, stopReq] = live_update(G, W, cfg, t);
            vfail = 0;
            if stopReq, W = log_event(W, t, 'Simulation stopped by user/figure closed'); break; end
        catch ME
            vfail = vfail + 1;
            warning('Visualization update failed (%d): %s', vfail, ME.message);
            if vfail >= cfg.viz.max_fail
                warning('Visualization disabled after repeated failures; simulation continues.');
                G = [];
            end
        end
    end
end
W.wall_time = toc(tWall); W.t_end = W.t;
if ~isempty(G)
    try, [G, ~] = live_update(G, W, cfg, W.t); drawnow; catch, end %#ok<CTCH>
end
M = calculate_performance_metrics(W);
R.W = W; R.M = M; R.cfg = cfg; R.G = G;
R.report = generate_test_report(M, cfg, W);
if ~isempty(cfg.report_file)
    fid = fopen(cfg.report_file, 'a');
    if fid > 0, fprintf(fid, '%s\n', R.report); fclose(fid); end
end
end

function W = initialize_simulation(cfg)
W.cfg = cfg; W.t = 0; W.k = 0; W.wall_time = 0; W.t_end = 0;
W.zones = initialize_airspace(cfg);
na = numel(cfg.aircraft);
for i = 1:na
    ac = initialize_aircraft(cfg, i);
    if i == 1, W.ac = ac; else, W.ac(i) = ac; end
end
tr = struct('pos',[0;0;0],'vel',[0;0;0],'t',0,'t_adsb',0,'src','init');
for i = 1:na
    x = W.ac(i).x; Rn = dcm_b2n(x(7),x(8),x(9)); v = Rn*x(1:3);
    tr.pos = [x(10);x(11);x(12)]; tr.vel = [v(1);v(2);-v(3)];
    if i == 1, W.atc.tr = tr; else, W.atc.tr(i) = tr; end
end
W.atc.preds = cell(1,na); W.atc.conf = struct('i',{},'j',{},'tloss',{},'tcpa',{},'smin',{},'dh',{}, ...
    'dv',{},'sev',{},'sevname',{},'pos',{}); W.atc.pred_ver = 0; W.atc.t_radar = 0;
W.atc.act = false(na); W.atc.peak = zeros(na); W.atc.los = false(na); W.atc.clear_t = zeros(1,na);
W.atc.stats = struct('detected',0,'warnings',0,'criticals',0,'resolved',0,'unresolved',0, ...
    'maneuvers',0,'recoveries',0,'zone_replans',0,'zone_warn',0);
W.atc.ml_conf = struct('i',{},'j',{},'p_conflict',{},'is_conflict',{},'warning_level',{}, ...
    'warning_name',{},'tcpa',{},'dcpa_h',{},'dcpa_v',{},'confidence',{}, ...
    'pos_uncert',{},'disagreement',{},'pos',{});
if isfield(cfg, 'ml') && cfg.ml.enable
    try
        W.predictor = ml_conflict_predictor('ml_conflict_model.mat', cfg.ml.model);
    catch
        W.predictor = [];
    end
else
    W.predictor = [];
end
W.zone_alert = zeros(1, max(numel(W.zones),1));
W.sep = struct('minH',inf,'minV',inf,'min3D',inf,'in_los',false(na),'los_events',0,'los_samples',0);
W.viol = struct('los',0,'zone',0,'bank',0,'alpha',0,'speed',0,'ground',0,'clamp',0,'nan',0);
W.fail_on = false(1, numel(cfg.fail));
W.events = cell(0,2); W.em_events = 0;
Nl = floor(round(cfg.T/cfg.dt)/round(cfg.dt_log/cfg.dt)) + 2;
W.L.t = nan(1,Nl); W.L.x = nan(12,Nl,na); W.L.xh = nan(12,Nl,na); W.L.u = nan(4,Nl,na);
W.L.ref = nan(6,Nl,na); W.L.mode = nan(Nl,na); W.L.minH = nan(1,Nl); W.L.sat = nan(4,Nl,na);
W.nlog = 0;
end

function zones = initialize_airspace(cfg)
zones = cfg.zones;
for z = 1:numel(zones)
    switch zones(z).type
        case {'circle','cylinder'}
            zones(z).bc = zones(z).center(:)'; zones(z).br = zones(z).radius;
        case 'polygon'
            P = zones(z).poly; c = mean(P,1);
            zones(z).bc = c; zones(z).br = max(hypot(P(:,1)-c(1), P(:,2)-c(2)));
        otherwise
            error('Unknown zone type %s', zones(z).type);
    end
end
end

function ss = initialize_sensors(cfg)
sc = cfg.sens;
ss.gyro_b = sc.gyro_bias0*randn(3,1); ss.accel_b = sc.accel_bias0*randn(3,1);
ss.gps_b = [1;1;2].*randn(3,1); ss.baro_b = 2*randn; ss.mag_b = sc.mag_bias0*randn;
ss.asp_b = sc.asp_bias0*randn;
ss.t_gps = 0; ss.t_baro = 0; ss.t_mag = 0; ss.t_asp = 0; ss.last_gps = 0;
end

function f = default_flags()
f = struct('gps_out',false,'comm_out',false,'eh',[1 1],'eff',[1 1 1 1],'rate_scale',1, ...
    'noise_scale',1,'dist',zeros(4,1),'baro_out',false,'mag_out',false,'asp_out',false);
end

function ac = initialize_aircraft(cfg, i)
d = cfg.aircraft(i); p = cfg.ac; if ~isempty(d.p), p = d.p; end
p.eh = [1 1];
route = d.route; V0 = d.V; h0 = route(1,3);
psi0 = atan2(route(2,2)-route(1,2), route(2,1)-route(1,1));
[al, de, thr] = compute_trim(p, V0, h0, 1, 0);
wn0 = wind_mean_ned(h0, cfg);
Rn = dcm_b2n(0, al, psi0); vb = [V0*cos(al); 0; V0*sin(al)] + Rn'*wn0;
x0 = [vb; 0;0;0; 0; al; psi0; route(1,1); route(1,2); h0];
ac.id = i; ac.name = d.name; ac.color = d.color; ac.p = p; ac.x = x0;
ac.act = [thr; de; 0; 0]; ac.ueff = ac.act; ac.cmd = ac.act; ac.sat = false(4,1);
ac.f = default_flags(); ac.wn = wn0; ac.wst = struct('g', zeros(3,1));
ac.sens = initialize_sensors(cfg);
xe = x0 + cfg.ekf.init_err.*randn(12,1); xe(9) = wrapToPi(xe(9));
ac.E = struct('x',xe,'P',cfg.ekf.P0,'rej',zeros(1,7),'acc',zeros(1,7),'consec',zeros(1,7), ...
    'resets',0,'imu_k',0);
ac.xh_last = xe;
ac.fp = waypoint_manager([], 'init', route, V0);
ac.ref = struct('init',false,'x',0,'y',0,'h',0,'psi',0,'V',V0);
ac.refcur = [route(1,1:3) psi0 V0 al];
ac.ctl = struct('u3',[0;0;V0],'int_V',0);
ac.beh = struct('phi_max',cfg.lim.phi_cmd,'gam_max',cfg.lim.gam,'Vover',NaN,'qh',1,'level',false,'atc',true);
ac.mode = 1; ac.mode_t = 0; ac.pending_ev = '';
ac.atc_sev = 0; ac.crit_t = 0; ac.alt_t = 0; ac.bank_flag = false; ac.in_los = false;
ac.t_next_mpc = 0; ac.t_adsb = 0; ac.radar_new = false; ac.radar_pos = [0 0 0];
ac.alive = true;
ac.mpc = struct('t',0,'n',0,'nfb',0,'last_t',0,'solver','none','last_reason','ok');
ac.t_ekf = 0; ac.n_ekf = 0; ac.sat_cnt = zeros(4,1); ac.n_steps = 0; ac.use_sum = zeros(4,1);
ac.vf = struct('bank',false,'alpha',false,'speed',false,'ground',false,'zone',false);
ac.adsb_ok = 0; ac.adsb_lost = 0;
end

%% ======================================================================
%  PHYSICS: ATMOSPHERE, WIND, DYNAMICS, ACTUATORS, TRIM
%% ======================================================================
function rho = isa_density(h)
h = min(max(h,0),11000);
rho = 1.225*(1 - 2.25577e-5*h).^4.2561;
end

function R = dcm_b2n(phi, th, psi)
cph = cos(phi); sph = sin(phi); cth = cos(th); sth = sin(th); cps = cos(psi); sps = sin(psi);
R = [cth*cps, sph*sth*cps-cph*sps, cph*sth*cps+sph*sps; ...
     cth*sps, sph*sth*sps+cph*cps, cph*sth*sps-sph*cps; ...
     -sth,    sph*cth,             cph*cth];
end

function a = wrapToPi(a)
a = mod(a + pi, 2*pi) - pi;
end

function y = clp(x, lo, hi)
y = min(max(x, lo), hi);
end

function wm = wind_mean_ned(h, cfg)
w = cfg.wind;
ws = w.speed*(max(h,100)/w.href)^w.shear;
wm = [-ws*cos(w.dir_from); -ws*sin(w.dir_from); 0];
end

function [wn, wst] = wind_model(wst, h, V, dt, cfg)
w = cfg.wind;
a = exp(-max(V,30)*dt/w.L);
sig = w.turb*[w.sig_u; w.sig_v; w.sig_w];
wst.g = a*wst.g + sqrt(1-a^2)*sig.*randn(3,1);
wn = wind_mean_ned(h, cfg) + wst.g;
end

function [alpha, de, thr, CL] = compute_trim(p, V, h, n, gam)
g = 9.80665; rho = isa_density(h); V = max(V,30); q = 0.5*rho*V^2;
CL = clp(n*p.m*g*cos(gam)/(q*p.S), 0, 1.8);
A = [p.CLa p.CLde; p.Cma p.Cmde];
sol = A\[CL - p.CL0; -p.Cm0]; alpha = sol(1); de = sol(2);
D = (p.CD0 + p.k*CL^2)*q*p.S;
T = (D + p.m*g*sin(gam))/max(cos(alpha),0.5);
lapse = (rho/1.225)^p.thrust_lapse;
thr = T/max(p.Tmax*lapse*max(mean(p.eh),0.05), 1);
end

function [xd, fb] = nasa_wind_dynamics(x, u, p, wn)
% 12-state nonlinear 6-DOF dynamics with wind-relative aerodynamics.
% x=[u v w p q r phi theta psi x y h], u=[thr de da dr], wn=[N;E;D] wind. fb = body specific force.
U = x(1); V = x(2); Wv = x(3); P = x(4); Q = x(5); R = x(6);
phi = x(7); th = x(8); psi = x(9); h = x(12);
Rn = dcm_b2n(phi, th, psi);
wb = Rn'*wn(:);
ua = U - wb(1); va = V - wb(2); wa = Wv - wb(3);
Va = max(sqrt(ua^2+va^2+wa^2), 20);
alpha = atan2(wa, max(ua,1)); alc = clp(alpha, -0.35, 0.35);
beta = asin(clp(va/Va, -0.9, 0.9));
rho = isa_density(h); qb = 0.5*rho*Va^2;
pb = P*p.b/(2*Va); qh = Q*p.c/(2*Va); rb = R*p.b/(2*Va);
thr = u(1); de = u(2); da = u(3); dr = u(4);
CL = min(p.CL0 + p.CLa*alc + p.CLq*qh + p.CLde*de, p.CLmax);
CD = p.CD0 + p.k*CL^2;
CY = p.CYb*beta + p.CYp*pb + p.CYr*rb + p.CYdr*dr;
Cl = p.Clb*beta + p.Clp*pb + p.Clr*rb + p.Clda*da + p.Cldr*dr;
Cm = p.Cm0 + p.Cma*alc + p.Cmq*qh + p.Cmde*de;
Cn = p.Cnb*beta + p.Cnp*pb + p.Cnr*rb + p.Cnda*da + p.Cndr*dr;
Xa = qb*p.S*(-CD*cos(alc) + CL*sin(alc));
Za = qb*p.S*(-CD*sin(alc) - CL*cos(alc));
Ya = qb*p.S*CY;
lapse = (rho/1.225)^p.thrust_lapse;
Tl = 0.5*thr*p.Tmax*lapse*p.eh(1); Tr = 0.5*thr*p.Tmax*lapse*p.eh(2);
Fb = [Xa + Tl + Tr; Ya; Za];
fb = Fb/p.m;
Mb = [qb*p.S*p.b*Cl; qb*p.S*p.c*Cm; qb*p.S*p.b*Cn + p.yeng*(Tl - Tr)];
om = [P; Q; R]; vbod = [U; V; Wv];
vdot = fb + Rn'*[0;0;9.80665] - cross(om, vbod);
I = [p.Ixx 0 -p.Ixz; 0 p.Iyy 0; -p.Ixz 0 p.Izz];
omd = I\(Mb - cross(om, I*om));
cth = cos(th); if abs(cth) < 0.05, cth = 0.05*(1 - 2*(cth < 0)); end
phid = P + tan(th)*(Q*sin(phi) + R*cos(phi));
thd = Q*cos(phi) - R*sin(phi);
psid = (Q*sin(phi) + R*cos(phi))/cth;
pos = Rn*vbod;
xd = [vdot; omd; phid; thd; psid; pos(1); pos(2); -pos(3)];
end

function xd = nasa_12state_dynamics(x, u, p)
xd = nasa_wind_dynamics(x, u, p, [0;0;0]);
end

function xd = aircraft_dynamics(x, u, p, wn)
% Plant dynamics with input limiting and NaN protection.
u = u(:); u(1) = clp(u(1),0,1); u(2:4) = clp(u(2:4), -0.6, 0.6);
xd = nasa_wind_dynamics(x, u, p, wn);
if any(~isfinite(xd)), xd = zeros(12,1); end
end

function xn = rk4_step(f, x, dt)
k1 = f(x); k2 = f(x + 0.5*dt*k1); k3 = f(x + 0.5*dt*k2); k4 = f(x + dt*k3);
xn = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);
end

function [x, bad] = sanitize_state(x, xprev)
bad = false;
if any(~isfinite(x)), x = xprev; bad = true; end
x0 = x;
x(1) = clp(x(1),20,250); x(2) = clp(x(2),-80,80); x(3) = clp(x(3),-80,80);
x(4:6) = clp(x(4:6),-3,3); x(7) = wrapToPi(x(7)); x(8) = clp(x(8),-1.35,1.35);
x(9) = wrapToPi(x(9)); x(12) = max(x(12), 0);
if any(abs(x([1:6 8 12]) - x0([1:6 8 12])) > 1e-9), bad = true; end
end

function [a, sat] = actuator_model(a, cmd, act, dt, f)
% First-order lag + rate limit + saturation; failures: effectiveness scaling (applied by caller),
% rate degradation, additive command disturbance.
cmd = cmd(:) + f.dist(:);
cs = clp(cmd, act.lo, act.hi);
rl = act.rate*f.rate_scale;
rt = clp((cs - a)./act.tau, -rl, rl);
sat = (abs(cmd - cs) > 1e-9) | (abs(abs(rt) - rl) < 1e-9);
a = clp(a + rt*dt, act.lo, act.hi);
end

%% ======================================================================
%  FAILURE INJECTION
%% ======================================================================
function W = inject_failures(W, t)
F = W.cfg.fail;
for k = 1:numel(F)
    idx = F(k).ac; if idx == 0, idx = 1:numel(W.ac); end
    switch F(k).type
        case {'bank_upset','alt_upset'}
            act = t >= F(k).t_on && t < F(k).t_on + F(k).val(2);
            for i = idx
                if strcmp(F(k).type,'bank_upset'), W.ac(i).f.dist(3) = F(k).val(1)*act;
                else, W.ac(i).f.dist(2) = F(k).val(1)*act; end
            end
            if act ~= W.fail_on(k)
                W.fail_on(k) = act;
                if act, W = log_event(W, t, sprintf('FAILURE INJECTED: %s on aircraft %s', F(k).type, mat2str(idx))); end
            end
        otherwise
            on = t >= F(k).t_on && t < F(k).t_off;
            if on ~= W.fail_on(k)
                for i = idx, W.ac(i).f = apply_fail(W.ac(i).f, F(k), on); end
                W.fail_on(k) = on;
                W = log_event(W, t, sprintf('FAILURE %s: %s on aircraft %s', tern(on,'INJECTED','CLEARED'), F(k).type, mat2str(idx)));
            end
    end
end
end

function f = apply_fail(f, F, on)
switch F.type
    case 'gps', f.gps_out = on;
    case 'comm', f.comm_out = on;
    case 'baro', f.baro_out = on;
    case 'mag', f.mag_out = on;
    case 'airspeed', f.asp_out = on;
    case 'engine'
        if on, f.eh = F.val(:)'; else, f.eh = [1 1]; end
    case 'noise'
        if on, f.noise_scale = F.val; else, f.noise_scale = 1; end
    case 'actuator'
        if on, f.eff = [1 F.val(1:3)]; f.rate_scale = F.val(4); else, f.eff = [1 1 1 1]; f.rate_scale = 1; end
end
end

function s = tern(c, a, b)
if c, s = a; else, s = b; end
end

function W = log_event(W, t, msg)
if size(W.events,1) < 3000, W.events(end+1,:) = {t, msg}; end
end

%% ======================================================================
%  SENSORS
%% ======================================================================
function [S, ss] = sensor_model(ss, x, fb, wn, t, cfg, f)
sc = cfg.sens; ns = f.noise_scale*sc.noise_scale; dt = cfg.dt;
ss.gyro_b = ss.gyro_b + sc.gyro_bias_rw*sqrt(dt)*randn(3,1);
ss.baro_b = ss.baro_b + sc.baro_bias_rw*sqrt(dt)*randn;
ss.gps_b = ss.gps_b + sc.gps_bias_rw*sqrt(dt)*randn(3,1);
Rn = dcm_b2n(x(7),x(8),x(9));
S.gyro = x(4:6) + ss.gyro_b + ns*sc.gyro_sig*randn(3,1);
S.accel = fb + ss.accel_b + ns*sc.accel_sig*randn(3,1);
S.gps_new = false; S.baro_new = false; S.mag_new = false; S.asp_new = false;
S.gps_pos = zeros(3,1); S.gps_vel = zeros(3,1); S.baro = 0; S.mag = 0; S.asp = 0;
if t >= ss.t_gps - 1e-9
    ss.t_gps = ss.t_gps + 1/sc.gps_rate;
    if ~f.gps_out
        S.gps_new = true;
        S.gps_pos = [x(10);x(11);x(12)] + ss.gps_b + ns*[sc.gps_sig_pos*randn(2,1); sc.gps_sig_alt*randn];
        S.gps_vel = Rn*x(1:3) + ns*sc.gps_sig_vel*randn(3,1);
        ss.last_gps = t;
    end
end
if t >= ss.t_baro - 1e-9
    ss.t_baro = ss.t_baro + 1/sc.baro_rate;
    if ~f.baro_out, S.baro_new = true; S.baro = x(12) + ss.baro_b + ns*sc.baro_sig*randn; end
end
if t >= ss.t_mag - 1e-9
    ss.t_mag = ss.t_mag + 1/sc.mag_rate;
    if ~f.mag_out, S.mag_new = true; S.mag = wrapToPi(x(9) + ss.mag_b + ns*sc.mag_sig*randn); end
end
if t >= ss.t_asp - 1e-9
    ss.t_asp = ss.t_asp + 1/sc.asp_rate;
    if ~f.asp_out
        S.asp_new = true;
        S.asp = norm(x(1:3) - Rn'*wn) + ss.asp_b + ns*sc.asp_sig*randn;
    end
end
end

function msg = adsb_model(ac, t, cfg)
x = ac.x; sc = cfg.adsb; Rn = dcm_b2n(x(7),x(8),x(9)); v = Rn*x(1:3);
msg.valid = true; msg.id = ac.id; msg.t = t;
msg.pos = [x(10) + sc.sig_pos*randn; x(11) + sc.sig_pos*randn; x(12) + sc.sig_alt*randn];
msg.gs = max(hypot(v(1),v(2)) + sc.sig_gs*randn, 1);
msg.hdg = atan2(v(2), v(1)) + sc.sig_hdg*randn;
msg.vs = -v(3) + sc.sig_vs*randn;
end

function plots = radar_model(W, ~)
r = W.cfg.radar; plots = struct('id',{},'pos',{});
for i = 1:numel(W.ac)
    ac = W.ac(i); if ~ac.alive, continue; end
    dx = ac.x(10) - r.pos(1); dy = ac.x(11) - r.pos(2); rg = hypot(dx, dy);
    if rg > r.range || rand > r.p_detect, continue; end
    az = atan2(dy, dx); rm = rg + r.sig_r*randn; am = az + r.sig_az*randn;
    plots(end+1) = struct('id', i, 'pos', [r.pos(1) + rm*cos(am); r.pos(2) + rm*sin(am); ac.x(12) + r.sig_alt*randn]); %#ok<AGROW>
end
end

function tr = atc_track_update(tr, m, t, src)
dt = t - tr.t; pp = tr.pos + tr.vel*dt;
switch src
    case 'adsb'
        tr.pos = pp + 0.8*(m.pos - pp);
        vm = [m.gs*cos(m.hdg); m.gs*sin(m.hdg); m.vs];
        tr.vel = tr.vel + 0.7*(vm - tr.vel);
        tr.t_adsb = t;
    case 'radar'
        inn = m.pos - pp; tr.pos = pp + 0.5*inn;
        if (t - tr.t_adsb) > 3 && dt > 0.1, tr.vel = tr.vel + 0.25*inn/dt; end
end
tr.t = t; tr.src = src;
end

function tr = atc_track_predict(tr, t)
tr.pos = tr.pos + tr.vel*(t - tr.t); tr.t = t;
end

function st = track_state(tr)
st.pos = tr.pos(:)'; st.psi = atan2(tr.vel(2), tr.vel(1)); st.V = max(hypot(tr.vel(1), tr.vel(2)), 30);
end

function W = surveillance_step(W, t)
cfg = W.cfg; na = numel(W.ac);
for i = 1:na
    ac = W.ac(i);
    if ~ac.alive, continue; end
    if t >= ac.t_adsb - 1e-9
        ac.t_adsb = ac.t_adsb + 1/cfg.adsb.rate;
        if ~ac.f.comm_out && rand > cfg.adsb.p_drop
            msg = adsb_model(ac, t, cfg);
            W.atc.tr(i) = atc_track_update(W.atc.tr(i), msg, t, 'adsb'); ac.adsb_ok = ac.adsb_ok + 1;
        else
            ac.adsb_lost = ac.adsb_lost + 1;
        end
        W.ac(i) = ac;
    end
end
if t >= W.atc.t_radar - 1e-9
    W.atc.t_radar = W.atc.t_radar + 1/cfg.radar.rate;
    plots = radar_model(W, t);
    for q = 1:numel(plots)
        i = plots(q).id;
        W.atc.tr(i) = atc_track_update(W.atc.tr(i), struct('pos',plots(q).pos), t, 'radar');
        W.ac(i).radar_pos = plots(q).pos'; W.ac(i).radar_new = true;
    end
end
end

%% ======================================================================
%  EKF (12-state) AND SENSOR-SPECIFIC UPDATES
%% ======================================================================
function J = numjac(f, x, f0)
n = numel(x); J = zeros(numel(f0), n);
for k = 1:n
    d = 1e-6*max(1, abs(x(k))); xp = x; xp(k) = xp(k) + d;
    J(:,k) = (f(xp) - f0)/d;
end
end

function P = pd_protect(P)
P = (P + P')/2;
if any(~isfinite(P(:))), return; end
[~, fl] = chol(P);
if fl ~= 0
    [V, D] = eig(P); D = diag(max(diag(D), 1e-9)); P = V*D*V'; P = (P + P')/2;
end
P(1:size(P,1)+1:end) = min(diag(P), 1e8);
end

function E = ekf_predict(E, u, p, wm, dt, Qc)
f = @(xx) nasa_wind_dynamics(xx, u, p, wm);
x = E.x; xd = f(x);
F = eye(12) + dt*numjac(f, x, xd);
xn = rk4_step(f, x, dt);
[xn, ~] = sanitize_state(xn, x);
E.x = xn; E.P = pd_protect(F*E.P*F' + Qc*dt);
end

function E = ekf_update_generic(E, z, zh, H, R, angidx, sid)
y = z(:) - zh(:);
if ~isempty(angidx), y(angidx) = wrapToPi(y(angidx)); end
S = H*E.P*H' + R; S = (S + S')/2 + 1e-12*eye(size(S));
tab = [10.83 13.82 16.27 18.47 20.52 22.46 24.32 26.12];
gate = tab(min(numel(z), 8));
d2 = y'*(S\y);
if ~isfinite(d2) || d2 > gate
    E.rej(sid) = E.rej(sid) + 1; E.consec(sid) = E.consec(sid) + 1;
    if ~isfinite(d2) || E.consec(sid) < 15, return; end
    R = 10*R; S = H*E.P*H' + R; S = (S + S')/2 + 1e-12*eye(size(S));   % recover from lock-out
end
E.consec(sid) = 0; E.acc(sid) = E.acc(sid) + 1;
K = (E.P*H')/S;
E.x = E.x + K*y;
Jo = eye(12) - K*H; E.P = pd_protect(Jo*E.P*Jo' + K*R*K');
[E.x, ~] = sanitize_state(E.x, E.x);
end

function E = ekf_update_gps(E, S, cfg)
sc = cfg.sens; k = cfg.ekf.R_inflate;
z = [S.gps_pos; S.gps_vel];
Hp = zeros(3,12); Hp(1,10) = 1; Hp(2,11) = 1; Hp(3,12) = 1;
hv = @(x) dcm_b2n(x(7),x(8),x(9))*x(1:3);
hv0 = hv(E.x); Hv = numjac(hv, E.x, hv0);
R = k*diag([sc.gps_sig_pos^2 sc.gps_sig_pos^2 sc.gps_sig_alt^2 sc.gps_sig_vel^2*[1 1 1]]);
E = ekf_update_generic(E, z, [E.x(10:12); hv0], [Hp; Hv], R, [], 1);
end

function E = ekf_update_baro(E, zb, cfg)
H = zeros(1,12); H(12) = 1;
E = ekf_update_generic(E, zb, E.x(12), H, cfg.ekf.R_inflate*cfg.sens.baro_sig^2, [], 2);
end

function E = ekf_update_mag(E, zm, cfg)
H = zeros(1,12); H(9) = 1;
E = ekf_update_generic(E, zm, E.x(9), H, cfg.ekf.R_inflate*cfg.sens.mag_sig^2, 1, 3);
end

function E = ekf_update_airspeed(E, za, wm, cfg)
ha = @(x) norm(x(1:3) - dcm_b2n(x(7),x(8),x(9))'*wm);
h0 = ha(E.x); H = numjac(ha, E.x, h0);
E = ekf_update_generic(E, za, h0, H, cfg.ekf.R_inflate*cfg.sens.asp_sig^2, [], 4);
end

function fb = fb_model(x, u, p, wm)
[~, fb] = nasa_wind_dynamics(x, u, p, wm); fb = fb(:);
end

function E = ekf_update_imu(E, gyro, accel, u, p, wm, cfg)
H = zeros(3,12); H(1,4) = 1; H(2,5) = 1; H(3,6) = 1;
E = ekf_update_generic(E, gyro, E.x(4:6), H, eye(3)*cfg.ekf.gyro_sig^2, [], 5);
E.imu_k = E.imu_k + 1;
if mod(E.imu_k, cfg.ekf.accel_div) == 0
    hf = @(x) fb_model(x, u, p, wm);
    f0 = hf(E.x); Ha = numjac(hf, E.x, f0);
    E = ekf_update_generic(E, accel, f0, Ha, eye(3)*cfg.ekf.accel_sig^2, [], 6);
end
end

function E = ekf_update_uplink(E, pos, cfg)
H = zeros(3,12); H(1,10) = 1; H(2,11) = 1; H(3,12) = 1;
s = cfg.ekf.uplink_sig;
E = ekf_update_generic(E, pos(:), E.x(10:12), H, diag([s^2 s^2 (0.5*s)^2]), [], 7);
end

function ac = estimator_step(ac, S, t, cfg)
E = ac.E; p = cfg.ac; p.eh = ac.f.eh; wm = wind_mean_ned(E.x(12), cfg);
E = ekf_predict(E, ac.ueff, p, wm, cfg.dt, cfg.ekf.Qc);
E = ekf_update_imu(E, S.gyro, S.accel, ac.ueff, p, wm, cfg);
if S.gps_new, E = ekf_update_gps(E, S, cfg); end
if S.baro_new, E = ekf_update_baro(E, S.baro, cfg); end
if S.mag_new, E = ekf_update_mag(E, S.mag, cfg); end
if S.asp_new, E = ekf_update_airspeed(E, S.asp, wm, cfg); end
if ac.radar_new
    if (t - ac.sens.last_gps) > cfg.ekf.gps_denied_T, E = ekf_update_uplink(E, ac.radar_pos, cfg); end
    ac.radar_new = false;
end
if any(~isfinite(E.x)) || any(~isfinite(E.P(:)))
    E.x = ac.xh_last; E.P = 10*cfg.ekf.P0; E.resets = E.resets + 1;
else
    ac.xh_last = E.x;
end
ac.E = E;
end

%% ======================================================================
%  WAYPOINTS, 4D PREDICTION, REPLANNING, RESTRICTED ZONES
%% ======================================================================
function fp = waypoint_manager(fp, cmd, varargin)
switch cmd
    case 'init'
        route = varargin{1};
        fp = struct('wps',route,'temp',false(size(route,1),1),'active',2,'Vcruise',varargin{2}, ...
            'dv',0,'arrived',false,'ver',1,'orig',route, ...
            'man',struct('active',false,'t',0,'dpsi',0,'dalt',0,'dv',0));
    case 'update'
        pos = varargin{1}; psi = varargin{2}; cfg = varargin{3}; cap = cfg.nav.capture;
        while fp.active <= size(fp.wps,1)
            d = fp.wps(fp.active,1:2) - pos(1:2); dist = hypot(d(1), d(2));
            behind = (d(1)*cos(psi) + d(2)*sin(psi)) < 0;
            if dist < cap || (behind && dist < 3*cap), fp.active = fp.active + 1; fp.ver = fp.ver + 1;
            else, break; end
        end
        if fp.active > size(fp.wps,1), fp.arrived = true; end
    case 'insert'
        idx = varargin{1}; wp = varargin{2}; tmp = varargin{3}; n = size(fp.wps,1);
        idx = max(1, min(idx, n+1));
        fp.wps = [fp.wps(1:idx-1,:); wp(:)'; fp.wps(idx:end,:)];
        fp.temp = [fp.temp(1:idx-1); tmp; fp.temp(idx:end)];
        if idx < fp.active, fp.active = fp.active + 1; end
        fp.ver = fp.ver + 1;
    case 'remove'
        idx = varargin{1};
        if idx >= 1 && idx <= size(fp.wps,1)
            if idx < fp.active, fp.active = fp.active - 1; end
            fp.wps(idx,:) = []; fp.temp(idx) = []; fp.ver = fp.ver + 1;
        end
    case 'reorder'
        perm = varargin{1}; old = fp.active;
        fp.wps = fp.wps(perm,:); fp.temp = fp.temp(perm);
        k = find(perm == old, 1); if isempty(k), k = 1; end
        fp.active = k; fp.ver = fp.ver + 1;
    case 'setdest'
        fp.wps(end,:) = varargin{1}(:)'; fp.orig(end,:) = varargin{1}(:)';
        fp.arrived = false; fp.ver = fp.ver + 1;
    case 'recover'
        n = size(fp.wps,1); keep = ~fp.temp;
        nb = sum(fp.temp(1:min(fp.active-1, n)));
        fp.wps = fp.wps(keep,:); fp.temp = false(sum(keep),1);
        fp.active = max(1, fp.active - nb); fp.dv = 0;
        fp.man = struct('active',false,'t',0,'dpsi',0,'dalt',0,'dv',0); fp.ver = fp.ver + 1;
    otherwise
        error('waypoint_manager: unknown command %s', cmd);
end
end

function pred = trajectory_predictor(st, fp, cfg, Vtarget, wind_ne, horizon, dtp)
% 4D prediction: bank-limited direct-to navigation along the flight plan.
% st: pos [x y h], psi, V. Returns t, pos (M+1 x 3), psi, V.
M = round(horizon/dtp); nv = cfg.nav; g = cfg.g;
t = (0:M)'*dtp; pos = zeros(M+1,3); psiA = zeros(M+1,1); VA = zeros(M+1,1);
x = st.pos(1); y = st.pos(2); h = st.pos(3); psi = st.psi; V = st.V;
pos(1,:) = [x y h]; psiA(1) = psi; VA(1) = V;
wi = fp.active; wps = fp.wps; nw = size(wps,1); wne = wind_ne(:)';
for k = 1:M
    while wi <= nw
        d = wps(wi,1:2) - [x y]; dist = hypot(d(1), d(2));
        behind = (d(1)*cos(psi) + d(2)*sin(psi)) < 0;
        if dist < nv.capture || (behind && dist < 3*nv.capture), wi = wi + 1; else, break; end
    end
    if wi <= nw
        psit = atan2(wps(wi,2) - y, wps(wi,1) - x); ht = wps(wi,3);
    else
        psit = psi; ht = h;
    end
    wmax = g*tan(nv.phi_nav)/max(V,30);
    psid = clp(nv.Kpsi*wrapToPi(psit - psi), -wmax, wmax);
    psi = psi + psid*dtp;
    V = V + clp(Vtarget - V, -nv.acc_max*dtp, nv.acc_max*dtp);
    h = h + clp(nv.Kh*(ht - h), -nv.climb_max, nv.climb_max)*dtp;
    x = x + (V*cos(psi) + wne(1))*dtp; y = y + (V*sin(psi) + wne(2))*dtp;
    pos(k+1,:) = [x y h]; psiA(k+1) = psi; VA(k+1) = V;
end
pred = struct('t',t,'pos',pos,'psi',psiA,'V',VA);
end

function [inside, dist] = restricted_zone_checker(zones, P, buf)
% P: n x 3 positions. inside(n x nz): within zone (+buffer, altitude band). dist: signed horizontal
% distance to the zone boundary (negative inside).
n = size(P,1); nz = numel(zones); inside = false(n, max(nz,1)); dist = inf(n, max(nz,1));
for z = 1:nz
    Zn = zones(z);
    switch Zn.type
        case {'circle','cylinder'}
            d = hypot(P(:,1) - Zn.center(1), P(:,2) - Zn.center(2)) - Zn.radius;
        otherwise
            Q = Zn.poly; m = size(Q,1); dm = inf(n,1);
            for e = 1:m
                a = Q(e,:); b = Q(mod(e,m)+1,:); ab = b - a; L2 = max(ab*ab', 1e-9);
                tt = clp(((P(:,1) - a(1))*ab(1) + (P(:,2) - a(2))*ab(2))/L2, 0, 1);
                dm = min(dm, hypot(P(:,1) - (a(1) + tt*ab(1)), P(:,2) - (a(2) + tt*ab(2))));
            end
            inp = inpolygon(P(:,1), P(:,2), Q(:,1), Q(:,2));
            d = dm.*(1 - 2*inp);
    end
    inAlt = P(:,3) >= Zn.hmin - 100 & P(:,3) <= Zn.hmax + 100;
    dist(:,z) = d; inside(:,z) = (d < buf) & inAlt;
end
end

function [fp, info] = trajectory_replanner(fp, st, mode, arg, cfg)
info = struct('ok',true,'zone',0);
switch mode
    case 'maneuver'
        L = max(cfg.res.L_min, st.V*cfg.res.L_time); psi = st.psi + arg.dpsi;
        if arg.dpsi ~= 0 || arg.dalt ~= 0
            wp = [st.pos(1) + L*cos(psi), st.pos(2) + L*sin(psi), clp(st.pos(3) + arg.dalt, cfg.lim.h_min, cfg.lim.h_max)];
            fp = waypoint_manager(fp, 'insert', fp.active, wp, true);
        end
        fp.dv = arg.dv;
        fp.man = struct('active',true,'t',arg.t,'dpsi',arg.dpsi,'dalt',arg.dalt,'dv',arg.dv);
    case 'zone'
        [fp, info] = zone_detour(fp, st, arg, cfg);
    case 'recover'
        fp = waypoint_manager(fp, 'recover');
end
end

function [fp, info] = zone_detour(fp, st, zones, cfg)
info = struct('ok',false,'zone',0);
if fp.active > size(fp.wps,1), return; end
pred0 = trajectory_predictor(st, fp, cfg, st.V + fp.dv, [0 0], cfg.sep.T_zone, cfg.sep.dtp);
[ins0, ~] = restricted_zone_checker(zones, pred0.pos, cfg.sep.zone_buf);
zi = find(any(ins0,1), 1); if isempty(zi), info.ok = true; return; end
info.zone = zi; c = zones(zi).bc; Pst = st.pos(1:2); T = fp.wps(fp.active,1:2);
d = T - Pst; u = d/max(norm(d),1); n = [-u(2) u(1)]; alt = fp.wps(fp.active,3);
Rb = zones(zi).br + cfg.sep.zone_buf + 1500; best = inf; bestfp = fp;
for s = [-1 1]
    for ex = [0 800 1600 2800]
        wp = [c + s*n*(Rb + ex), alt];
        f2 = waypoint_manager(fp, 'insert', fp.active, wp, true);
        pr = trajectory_predictor(st, f2, cfg, st.V + f2.dv, [0 0], cfg.sep.T_zone, cfg.sep.dtp);
        [ins, ~] = restricted_zone_checker(zones, pr.pos, cfg.sep.zone_buf);
        len = norm(wp(1:2) - Pst) + norm(T - wp(1:2));
        if ~any(ins(:)) && len < best, best = len; bestfp = f2; end
    end
end
if isfinite(best), fp = bestfp; info.ok = true; end
end

%% ======================================================================
%  CONFLICT DETECTION, CLASSIFICATION, RESOLUTION
%% ======================================================================
function [smin, kmin, tloss, dh, dv] = pair_sep(pa, pb, cfg)
M = min(size(pa.pos,1), size(pb.pos,1));
dx = pa.pos(1:M,1) - pb.pos(1:M,1); dy = pa.pos(1:M,2) - pb.pos(1:M,2);
dhv = hypot(dx, dy); dvv = abs(pa.pos(1:M,3) - pb.pos(1:M,3));
s = max(dhv/cfg.sep.H, dvv/cfg.sep.V);
[smin, kmin] = min(s);
loss = s < cfg.sep.detect_factor;
if any(loss), tloss = pa.t(find(loss,1)); else, tloss = inf; end
dh = dhv(kmin); dv = dvv(kmin);
end

function C = conflict_detector_4d(preds, cfg)
% Time-aligned pairwise detection. Severity: CRITICAL if loss within T_crit (or already violated),
% WARNING if within T_warn, NORMAL otherwise. Returns only encounters with predicted loss.
C = struct('i',{},'j',{},'tloss',{},'tcpa',{},'smin',{},'dh',{},'dv',{},'sev',{},'sevname',{},'pos',{});
na = numel(preds);
for i = 1:na-1
    for j = i+1:na
        if isempty(preds{i}) || isempty(preds{j}), continue; end
        [smin, kmin, tloss, dh, dv] = pair_sep(preds{i}, preds{j}, cfg);
        if ~isfinite(tloss), continue; end
        if tloss <= cfg.sep.T_crit, sev = 2; elseif tloss <= cfg.sep.T_warn, sev = 1; else, sev = 0; end
        names = {'NORMAL','WARNING','CRITICAL'};
        pc = 0.5*(preds{i}.pos(kmin,:) + preds{j}.pos(kmin,:));
        C(end+1) = struct('i',i,'j',j,'tloss',tloss,'tcpa',preds{i}.t(kmin),'smin',smin,'dh',dh,'dv',dv, ...
            'sev',sev,'sevname',names{sev+1},'pos',pc); %#ok<AGROW>
    end
end
end

function [best, found] = evaluate_candidates(W, a, preds, level)
cfg = W.cfg; res = cfg.res; ac = W.ac(a); fp0 = ac.fp;
st = track_state(W.atc.tr(a)); wm = wind_mean_ned(st.pos(3), cfg);
fpb = waypoint_manager(fp0, 'recover');
wal = wm(1)*cos(st.psi) + wm(2)*sin(st.psi);
base = trajectory_predictor(st, fpb, cfg, fpb.Vcruise + wal, [0 0], cfg.sep.horizon, cfg.sep.dtp);
others = find(arrayfun(@(q) q.alive, W.ac)); others(others == a) = [];
dpsiS = res.dpsi; daltS = res.dalt; dvS = res.dv;
if level > 1, dpsiS = [dpsiS, [-90 -75 75 90]*pi/180]; daltS = [daltS, -900, 900]; end
engine = strcmp(cfg.modes{ac.mode}, 'ENGINE_FAILURE');
best = struct('cost',inf,'smin',-inf,'dpsi',0,'dalt',0,'dv',0,'pred',[],'valid',false); found = false;
fb = struct('cost',inf,'smin',-inf,'dpsi',0,'dalt',0,'dv',0,'pred',[],'valid',false);
for dpsi = [0 dpsiS]
    for dalt = daltS
        for dv = dvS
            if st.pos(3) + dalt < cfg.lim.h_min + 100 || st.pos(3) + dalt > cfg.lim.h_max, continue; end
            Vt = fpb.Vcruise + dv; if Vt < cfg.lim.Vcmd_min || Vt > cfg.lim.Vcmd_max, continue; end
            if engine && (dalt > 0 || dv > 0), continue; end
            arg = struct('dpsi',dpsi,'dalt',dalt,'dv',dv,'t',W.t);
            [f2, ~] = trajectory_replanner(fpb, st, 'maneuver', arg, cfg);
            pr = trajectory_predictor(st, f2, cfg, Vt + wal, [0 0], cfg.sep.horizon, cfg.sep.dtp);
            smin = inf;
            for o = others
                if isempty(preds{o}), continue; end
                smin = min(smin, pair_sep(pr, preds{o}, cfg));
            end
            zint = 0;
            if ~isempty(W.zones)
                [ins, ~] = restricted_zone_checker(W.zones, pr.pos, cfg.sep.zone_buf); zint = any(ins(:));
            end
            M = min(size(pr.pos,1), size(base.pos,1));
            dev = mean(hypot(pr.pos(1:M,1) - base.pos(1:M,1), pr.pos(1:M,2) - base.pos(1:M,2)))/cfg.sep.H;
            J = res.w_sep*max(0, cfg.sep.margin - smin)^2 + res.w_route*dev + res.w_alt*abs(dalt)/cfg.sep.V ...
                + res.w_spd*abs(dv)/15 + res.w_man*abs(dpsi)/(45*pi/180) + res.w_zone*zint;
            cand = struct('cost',J,'smin',smin,'dpsi',dpsi,'dalt',dalt,'dv',dv,'pred',pr,'valid',smin >= cfg.sep.detect_factor && ~zint);
            if cand.valid && J < best.cost, best = cand; found = true; end
            if smin > fb.smin || (smin == fb.smin && J < fb.cost), fb = cand; end
        end
    end
end
if ~found, best = fb; end
end

function [W, preds, nres] = conflict_resolver(W, C, preds, t, level)
% Evaluates heading/altitude/speed candidates with a multi-term cost, commits the best maneuver for the
% yielding aircraft and updates its prediction so later resolutions see the modified trajectory.
cfg = W.cfg; nres = 0;
if isempty(C), return; end
[~, ord] = sortrows([-[C.sev]' [C.tloss]']); C = C(ord);
for c = 1:numel(C)
    if C(c).sev < 1, continue; end
    ia = C(c).i; ib = C(c).j;
    if isempty(preds{ia}) || isempty(preds{ib}), continue; end
    if pair_sep(preds{ia}, preds{ib}, cfg) >= cfg.sep.detect_factor, continue; end
    cand = [ia ib]; ok = false(1,2);
    for q = 1:2
        a = cand(q); ac = W.ac(a);
        ok(q) = ac.alive && ~ac.f.comm_out && ac.beh.atc && ~ac.beh.level;
    end
    yl = cand(ok);
    if isempty(yl), continue; end
    man = arrayfun(@(a) W.ac(a).fp.man.active && (t - W.ac(a).fp.man.t) < cfg.sep.hold, yl);
    if level == 1 && any(~man) && any(man), yl = yl(~man); end
    if numel(yl) > 1, yl = sort(yl, 'descend'); end
    for a = yl
        [best, found] = evaluate_candidates(W, a, preds, level);
        if isempty(best.pred), continue; end
        st = track_state(W.atc.tr(a));
        arg = struct('dpsi',best.dpsi,'dalt',best.dalt,'dv',best.dv,'t',t);
        fpb = waypoint_manager(W.ac(a).fp, 'recover');
        [W.ac(a).fp, ~] = trajectory_replanner(fpb, st, 'maneuver', arg, cfg);
        preds{a} = best.pred; nres = nres + 1; W.atc.stats.maneuvers = W.atc.stats.maneuvers + 1;
        W = log_event(W, t, sprintf('RESOLUTION %s: dHdg=%+.0f deg, dAlt=%+.0f m, dV=%+.0f m/s, min sep=%.2f %s', ...
            W.ac(a).name, best.dpsi*180/pi, best.dalt, best.dv, best.smin, tern(found,'(conflict-free)','(BEST EFFORT)')));
        if pair_sep(preds{ia}, preds{ib}, cfg) >= cfg.sep.margin, break; end
    end
end
end

function W = update_conflict_stats(W, C)
A = W.atc; na = numel(W.ac); cur = zeros(na);
for c = 1:numel(C)
    if C(c).sev >= 1, cur(C(c).i, C(c).j) = max(cur(C(c).i, C(c).j), C(c).sev); end
end
for i = 1:na-1
    for j = i+1:na
        has = cur(i,j) >= 1;
        if has && ~A.act(i,j)
            A.act(i,j) = true; A.peak(i,j) = 0; A.los(i,j) = false; A.stats.detected = A.stats.detected + 1;
        end
        if has && cur(i,j) > A.peak(i,j)
            if cur(i,j) == 1, A.stats.warnings = A.stats.warnings + 1; end
            if cur(i,j) == 2, A.stats.criticals = A.stats.criticals + 1; end
            A.peak(i,j) = cur(i,j);
        end
        if ~has && A.act(i,j)
            A.act(i,j) = false;
            if A.los(i,j), A.stats.unresolved = A.stats.unresolved + 1; else, A.stats.resolved = A.stats.resolved + 1; end
        end
    end
end
W.atc = A;
end

function W = atc_cycle(W, t)
cfg = W.cfg; na = numel(W.ac); A = W.atc;
for i = 1:na, A.tr(i) = atc_track_predict(A.tr(i), t); end
W.atc = A; preds = cell(1,na);
for i = 1:na
    if ~W.ac(i).alive, continue; end
    st = track_state(A.tr(i)); fp = W.ac(i).fp; wm = wind_mean_ned(st.pos(3), cfg);
    wal = wm(1)*cos(st.psi) + wm(2)*sin(st.psi);
    preds{i} = trajectory_predictor(st, fp, cfg, fp.Vcruise + fp.dv + wal, [0 0], cfg.sep.horizon, cfg.sep.dtp);
end
% ---- restricted zones: predicted intrusion -> replan around ----
W.zone_alert(:) = 0;
for i = 1:na
    if isempty(preds{i}) || isempty(W.zones), continue; end
    [ins, ~] = restricted_zone_checker(W.zones, preds{i}.pos, cfg.sep.zone_buf);
    if any(ins(:))
        W.zone_alert(any(ins,1)) = max(1, W.zone_alert(any(ins,1)));
        W.atc.stats.zone_warn = W.atc.stats.zone_warn + 1;
        st = track_state(A.tr(i)); [fp2, info] = trajectory_replanner(W.ac(i).fp, st, 'zone', W.zones, cfg);
        if info.ok && fp2.ver ~= W.ac(i).fp.ver
            W.ac(i).fp = fp2; W.atc.stats.zone_replans = W.atc.stats.zone_replans + 1;
            W = log_event(W, t, sprintf('ZONE REPLAN %s around %s', W.ac(i).name, W.zones(info.zone).name));
            wm = wind_mean_ned(st.pos(3), cfg); wal = wm(1)*cos(st.psi) + wm(2)*sin(st.psi);
            preds{i} = trajectory_predictor(st, fp2, cfg, fp2.Vcruise + fp2.dv + wal, [0 0], cfg.sep.horizon, cfg.sep.dtp);
        elseif ~info.ok
            W = log_event(W, t, sprintf('ZONE WARNING %s: predicted intrusion, no clean detour found', W.ac(i).name));
        end
    end
end
% ---- conflict detection / resolution (iterative) ----
C = conflict_detector_4d(preds, cfg);
for it = 1:cfg.res.max_iter
    if ~any([C.sev] >= 1), break; end
    [W, preds, nres] = conflict_resolver(W, C, preds, t, it);
    C = conflict_detector_4d(preds, cfg);
    if nres == 0, break; end
end
% ---- recovery to the original plan once clear ----
sevAc = zeros(1,na);
for c = 1:numel(C)
    if C(c).sev >= 1, sevAc([C(c).i C(c).j]) = max(sevAc([C(c).i C(c).j]), C(c).sev); end
end
for i = 1:na
    if sevAc(i) == 0, W.atc.clear_t(i) = W.atc.clear_t(i) + cfg.dt_atc; else, W.atc.clear_t(i) = 0; end
    fp = W.ac(i).fp;
    if W.ac(i).alive && fp.man.active && (t - fp.man.t) > cfg.sep.hold && W.atc.clear_t(i) >= cfg.sep.T_clear
        fp2 = waypoint_manager(fp, 'recover'); st = track_state(A.tr(i)); wm = wind_mean_ned(st.pos(3), cfg);
        wal = wm(1)*cos(st.psi) + wm(2)*sin(st.psi);
        pr = trajectory_predictor(st, fp2, cfg, fp2.Vcruise + wal, [0 0], cfg.sep.horizon, cfg.sep.dtp);
        free = true;
        for o = 1:na
            if o ~= i && ~isempty(preds{o}) && pair_sep(pr, preds{o}, cfg) < cfg.sep.margin, free = false; break; end
        end
        if free && ~isempty(W.zones)
            [ins, ~] = restricted_zone_checker(W.zones, pr.pos, cfg.sep.zone_buf); free = ~any(ins(:));
        end
        if free
            W.ac(i).fp = fp2; preds{i} = pr; W.atc.stats.recoveries = W.atc.stats.recoveries + 1;
            W = log_event(W, t, sprintf('RECOVERY %s: conflict clear, resuming original flight plan', W.ac(i).name));
        else
            W.atc.clear_t(i) = 0;
        end
    end
    W.ac(i).atc_sev = sevAc(i);
end
C = conflict_detector_4d(preds, cfg);
W = update_conflict_stats(W, C);
if isfield(W, 'predictor') && ~isempty(W.predictor)
    try
        W.atc.ml_conf = W.predictor.evaluate_airspace(W, cfg);
        for q = 1:numel(W.atc.ml_conf)
            if W.atc.ml_conf(q).is_conflict && W.atc.ml_conf(q).disagreement
                W = log_event(W, t, sprintf('AI EARLY WARNING %s-%s: P(conf)=%.2f, TCPA=%.0fs, est sep=%.0fm', ...
                    W.ac(W.atc.ml_conf(q).i).name, W.ac(W.atc.ml_conf(q).j).name, ...
                    W.atc.ml_conf(q).p_conflict, W.atc.ml_conf(q).tcpa, W.atc.ml_conf(q).dcpa_h));
            end
        end
    catch
    end
end
W.atc.preds = preds; W.atc.conf = C; W.atc.pred_ver = W.atc.pred_ver + 1;
end

%% ======================================================================
%  EMERGENCY MANAGEMENT
%% ======================================================================
function ac = update_emergency(ac, t, cfg)
dem = cfg.dt_em; xh = ac.E.x;
href = ac.fp.wps(min(ac.fp.active, size(ac.fp.wps,1)), 3);
if ac.ref.init, href = ac.ref.h; end
if abs(xh(12) - href) > cfg.em.alt_dev, ac.alt_t = ac.alt_t + dem; else, ac.alt_t = 0; end
if ~ac.bank_flag && abs(xh(7)) > cfg.em.bank_on, ac.bank_flag = true; end
if ac.bank_flag && abs(xh(7)) < cfg.em.bank_off, ac.bank_flag = false; end
if ac.atc_sev == 2, ac.crit_t = ac.crit_t + dem; else, ac.crit_t = 0; end
fl.engine = mean(ac.f.eh) < 0.85;
fl.gps = (t - ac.sens.last_gps) > cfg.ekf.gps_denied_T;
fl.comm = ac.f.comm_out;
fl.alt_dev = ac.alt_t > cfg.em.alt_t;
fl.bank = ac.bank_flag;
fl.emerg = ac.crit_t > cfg.em.crit_t || ac.in_los;
fl.avoid = ac.fp.man.active;
fl.warn = ac.atc_sev >= 1;
[m, mt, beh, ev] = emergency_manager(ac, fl, cfg, t);
ac.mode = m; ac.mode_t = mt; ac.beh = beh;
if ~isempty(ev), ac.pending_ev = ev; end
end

function [mode, mode_t, beh, ev] = emergency_manager(ac, fl, cfg, t)
% Modes (cfg.modes): 1 NORMAL 2 WARNING 3 CONFLICT_AVOIDANCE 4 EMERGENCY 5 ENGINE_FAILURE
% 6 GPS_FAILURE 7 COMMUNICATION_LOSS 8 ALTITUDE_DEVIATION 9 EXCESSIVE_BANK
rank = [1 2 3 7 8 6 5 4 9];
cand = 1;
if fl.warn, cand = 2; end
if fl.avoid, cand = 3; end
if fl.alt_dev, cand = 8; end
if fl.comm, cand = 7; end
if fl.gps, cand = 6; end
if fl.emerg, cand = 4; end
if fl.engine, cand = 5; end
if fl.bank, cand = 9; end
mode = ac.mode; mode_t = ac.mode_t; ev = '';
if cand ~= ac.mode && ~((t - ac.mode_t) < cfg.em.dwell && rank(cand) <= rank(ac.mode))
    ev = sprintf('mode %s -> %s', cfg.modes{ac.mode}, cfg.modes{cand}); mode = cand; mode_t = t;
end
lim = cfg.lim; d2r = pi/180; V0 = ac.fp.Vcruise;
beh = struct('phi_max',lim.phi_cmd,'gam_max',lim.gam,'Vover',NaN,'qh',1,'level',false,'atc',true);
switch cfg.modes{mode}
    case 'CONFLICT_AVOIDANCE', beh.phi_max = 30*d2r; beh.gam_max = 0.12;
    case 'EMERGENCY', beh.phi_max = 35*d2r; beh.gam_max = 0.14;
    case 'ENGINE_FAILURE', beh.phi_max = 15*d2r; beh.gam_max = 0.06; beh.Vover = max(0.85*V0, lim.Vcmd_min);
    case 'GPS_FAILURE', beh.phi_max = 18*d2r; beh.gam_max = 0.08;
    case 'COMMUNICATION_LOSS', beh.phi_max = 22*d2r; beh.atc = false;
    case 'ALTITUDE_DEVIATION', beh.phi_max = 15*d2r; beh.gam_max = 0.14; beh.qh = 4;
    case 'EXCESSIVE_BANK', beh.phi_max = 10*d2r; beh.gam_max = 0.05; beh.level = true;
end
end

%% ======================================================================
%  MPC
%% ======================================================================
function [Phi, Gam, Dv] = build_mpc_prediction_matrices(Al, Bl, dl)
% LTV prediction: z_{k+1} = A_k z_k + B_k u_k + d_k, k=0..N-1 (cells Al{k+1} etc.)
% Z=[z_1;...;z_N] = Phi*z_0 + Gam*U + Dv, U=[u_0;...;u_{N-1}]. Dimensions are verified.
N = numel(Al); nz = size(Al{1},1); nu = size(Bl{1},2);
assert(numel(Bl) == N && numel(dl) == N, 'build_mpc_prediction_matrices: cell length mismatch');
Phi = zeros(nz*N, nz); Gam = zeros(nz*N, nu*N); Dv = zeros(nz*N, 1);
Pp = eye(nz); Dp = zeros(nz,1); Gp = zeros(nz, nu*N);
for i = 1:N
    assert(isequal(size(Al{i}),[nz nz]) && isequal(size(Bl{i}),[nz nu]) && numel(dl{i}) == nz, ...
        'build_mpc_prediction_matrices: dimension error at step %d', i);
    Pi = Al{i}*Pp; Di = Al{i}*Dp + dl{i}(:); Gi = Al{i}*Gp; Gi(:, (i-1)*nu + (1:nu)) = Bl{i};
    r = (i-1)*nz + (1:nz); Phi(r,:) = Pi; Dv(r) = Di; Gam(r,:) = Gi;
    Pp = Pi; Dp = Di; Gp = Gi;
end
assert(isequal(size(Phi),[nz*N nz]) && isequal(size(Gam),[nz*N nu*N]) && numel(Dv) == nz*N);
end

function U = qp_box_solve(H, f, lb, ub)
% Toolbox-independent box-constrained QP (projected active-set Newton).
U = clp(-(H\f), lb, ub);
for it = 1:40
    g = H*U + f;
    free = ~((U <= lb + 1e-9 & g > 0) | (U >= ub - 1e-9 & g < 0));
    if ~any(free), break; end
    stp = zeros(size(U)); stp(free) = -(H(free,free)\g(free));
    Un = clp(U + stp, lb, ub);
    if norm(Un - U) < 1e-9, break; end
    U = Un;
end
end

function u = guidance_fallback(z0, ref, beh, cfg)
% Safe controller: bank from bearing to a look-ahead reference point, gamma to reach reference altitude.
N = numel(ref.x); la = min(N, 6); g = cfg.g;
psid = atan2(ref.y(la) - z0(2), ref.x(la) - z0(1));
rate = clp(0.35*wrapToPi(psid - z0(4)), -0.08, 0.08);
phi = clp(atan(z0(5)*rate/g), -beh.phi_max, beh.phi_max);
gam = clp(atan2(ref.h(la) - z0(3), z0(5)*la*cfg.dt_mpc), -beh.gam_max, beh.gam_max);
u = [phi; gam; clp(ref.V(1), cfg.lim.Vcmd_min, cfg.lim.Vcmd_max)];
end

function [u, info] = mpc_controller(z0, ref, uprev, beh, cfg)
% z0=[x y h psi V phi gamma]'; ref fields x,y,h,psi,V,gam,phiff (N x 1); uprev=[phi_c gam_c V_c]'.
persistent qpOK qpOpts
if isempty(qpOK)
    qpOK = exist('quadprog','file') == 2;
    if qpOK
        try qpOK = license('test','Optimization_Toolbox') == 1; catch, qpOK = false; end
    end
    qpOpts = [];
    if qpOK
        try qpOpts = optimoptions('quadprog','Display','off'); catch, try qpOpts = optimset('Display','off'); catch, qpOpts = []; end, end
    end
end
mp = cfg.mpc; N = mp.N; dtm = cfg.dt_mpc; nz = 7; nu = 3; g = cfg.g; lim = cfg.lim;
info = struct('solver','none','exitflag',NaN,'time',0,'fallback',false,'reason','ok');
tS = tic; Al = cell(1,N); Bl = cell(1,N); dl = cell(1,N);
w = ref.wind(:);
for k = 1:N
    psr = ref.psi(k); Vr = max(ref.V(k), 30); gr = ref.gam(k); phr = ref.phiff(k);
    A = zeros(nz); B = zeros(nz,nu); d = zeros(nz,1); cg = cos(gr); sg = sin(gr);
    f1 = Vr*cg*cos(psr) + w(1); f2 = Vr*cg*sin(psr) + w(2); f3 = Vr*sg;
    A(1,4) = -Vr*cg*sin(psr); A(1,5) = cg*cos(psr); A(1,7) = -Vr*sg*cos(psr);
    A(2,4) =  Vr*cg*cos(psr); A(2,5) = cg*sin(psr); A(2,7) = -Vr*sg*sin(psr);
    A(3,5) = sg; A(3,7) = Vr*cg;
    A(4,6) = g/(Vr*cos(phr)^2); A(4,5) = -g*tan(phr)/Vr^2;
    A(5,5) = -1/mp.tau_V; A(6,6) = -1/mp.tau_phi; A(7,7) = -1/mp.tau_gam;
    B(5,3) = 1/mp.tau_V; B(6,1) = 1/mp.tau_phi; B(7,2) = 1/mp.tau_gam;
    d(1) = f1 - A(1,4)*psr - A(1,5)*Vr - A(1,7)*gr;
    d(2) = f2 - A(2,4)*psr - A(2,5)*Vr - A(2,7)*gr;
    d(3) = f3 - A(3,5)*Vr - A(3,7)*gr;
    d(4) = g*tan(phr)/Vr - A(4,5)*Vr - A(4,6)*phr;
    Mx = expm([A B d; zeros(nu+1, nz+nu+1)]*dtm);
    Al{k} = Mx(1:nz,1:nz); Bl{k} = Mx(1:nz, nz+(1:nu)); dl{k} = Mx(1:nz, nz+nu+1);
end
[Phi, Gam, Dv] = build_mpc_prediction_matrices(Al, Bl, dl);
Cy = [eye(5) zeros(5,2)]; Cb = kron(eye(N), Cy);
Qy = diag(mp.Qy); Qy(3,3) = Qy(3,3)*beh.qh;
Qb = kron(eye(N), Qy); Qb(end-4:end, end-4:end) = mp.term_factor*Qy;
Rb = kron(eye(N), diag(mp.Ru)); Sb = kron(eye(N), diag(mp.Su));
Dm = eye(nu*N) - kron(diag(ones(N-1,1), -1), eye(nu));
d0 = [uprev(:); zeros(nu*(N-1),1)];
Yr = reshape([ref.x(:)'; ref.y(:)'; ref.h(:)'; ref.psi(:)'; ref.V(:)'], 5*N, 1);
Un = reshape([zeros(1,N); ref.gam(:)'; ref.V(:)'], nu*N, 1);
Zf = Phi*z0(:) + Dv; y0 = Cb*Zf; Gy = Cb*Gam;
H = 2*(Gy'*Qb*Gy + Rb + Dm'*Sb*Dm); H = (H + H')/2 + 1e-8*eye(nu*N);
f = 2*(Gy'*Qb*(y0 - Yr) - Rb*Un - Dm'*Sb*d0);
lb = repmat([-beh.phi_max; -beh.gam_max; lim.Vcmd_min], N, 1);
ub = repmat([ beh.phi_max;  beh.gam_max; lim.Vcmd_max], N, 1);
rate = repmat([lim.rate_phi; lim.rate_gam; lim.rate_V]*dtm, N, 1);
Sph = kron(eye(N), [0 0 0 0 0 1 0]); Sgm = kron(eye(N), [0 0 0 0 0 0 1]);
phb = max(1.1*beh.phi_max, abs(z0(6))) + 0.02; gmb = max(1.1*beh.gam_max, abs(z0(7))) + 0.02;
Ain = [Dm; -Dm; Sph*Gam; -Sph*Gam; Sgm*Gam; -Sgm*Gam];
bin = [rate + d0; rate - d0; phb - Sph*Zf; phb + Sph*Zf; gmb - Sgm*Zf; gmb + Sgm*Zf];
U = []; ok = false;
if qpOK && mp.use_quadprog
    try
        [U, ~, ef] = quadprog(H, f, Ain, bin, [], [], lb, ub, [], qpOpts);
        info.solver = 'quadprog'; info.exitflag = ef; ok = ef > 0 && all(isfinite(U));
        if ~ok, info.reason = sprintf('quadprog exitflag %d', ef); end
    catch ME
        info.reason = ['quadprog error: ' ME.message]; qpOK = false; ok = false;
    end
else
    try
        U = qp_box_solve(H, f, lb, ub); info.solver = 'boxQP'; ok = all(isfinite(U));
        if ok, U(1:nu) = clp(U(1:nu), uprev(:) - rate(1:nu), uprev(:) + rate(1:nu)); end
    catch ME
        info.reason = ['boxQP error: ' ME.message]; ok = false;
    end
end
if ok
    u = U(1:nu);
    if any(u < lb(1:nu) - 1e-6) || any(u > ub(1:nu) + 1e-6), ok = false; info.reason = 'input constraint violated'; end
end
if ~ok
    u = guidance_fallback(z0, ref, beh, cfg); info.fallback = true;
    if strcmp(info.reason,'ok'), info.reason = 'MPC failed'; end
    u = clp(u, uprev(:) - rate(1:nu), uprev(:) + rate(1:nu));
end
u = clp(u, lb(1:nu), ub(1:nu));
if any(~isfinite(u)), u = [0; 0; clp(ref.V(1), lim.Vcmd_min, lim.Vcmd_max)]; info.fallback = true; info.reason = 'NaN output'; end
info.time = toc(tS);
end

function ac = run_mpc_cycle(ac, cfg)
beh = ac.beh; xh = ac.E.x; fp = ac.fp; N = cfg.mpc.N; dtm = cfg.dt_mpc; lim = cfg.lim;
wm = wind_mean_ned(xh(12), cfg); Rn = dcm_b2n(xh(7),xh(8),xh(9)); vned = Rn*xh(1:3);
Vair = max(norm(xh(1:3) - Rn'*wm), 30); gam0 = asin(clp(-vned(3)/Vair, -0.5, 0.5));
z0 = [xh(10); xh(11); xh(12); xh(9); Vair; xh(7); gam0];
Vtar = fp.Vcruise + fp.dv; if ~isnan(beh.Vover), Vtar = min(Vtar, beh.Vover); end
Vtar = clp(Vtar, lim.Vcmd_min, lim.Vcmd_max);
r = ac.ref;
if ~r.init || hypot(r.x - z0(1), r.y - z0(2)) > cfg.mpc.max_lag || abs(r.h - z0(3)) > 400
    r = struct('init',true,'x',z0(1),'y',z0(2),'h',z0(3),'psi',z0(4),'V',z0(5));
end
st = struct('pos',[r.x r.y r.h],'psi',r.psi,'V',r.V);
pr = trajectory_predictor(st, fp, cfg, Vtar, wm(1:2), N*dtm, dtm);
psu = zeros(N,1); prev = z0(4);
for k = 1:N, psu(k) = prev + wrapToPi(pr.psi(k+1) - prev); prev = psu(k); end
pdot = diff([pr.psi(1) - wrapToPi(pr.psi(1) - z0(4)) + wrapToPi(pr.psi(1) - z0(4)); psu])/dtm;
ref.x = pr.pos(2:end,1); ref.y = pr.pos(2:end,2); ref.h = pr.pos(2:end,3); ref.psi = psu;
ref.V = pr.V(2:end); ref.wind = wm(1:2);
ref.gam = atan2(diff(pr.pos(:,3))/dtm, ref.V);
ref.phiff = clp(atan(ref.V.*pdot/cfg.g), -beh.phi_max, beh.phi_max);
if beh.level
    u3 = [0; 0; Vtar]; info = struct('solver','override','fallback',false,'time',0,'reason','wings-level recovery');
else
    [u3, info] = mpc_controller(z0, ref, ac.ctl.u3, beh, cfg);
end
r.x = pr.pos(2,1); r.y = pr.pos(2,2); r.h = pr.pos(2,3); r.psi = pr.psi(2); r.V = pr.V(2);
ac.ref = r; ac.ctl.u3 = u3;
[al, ~] = compute_trim(cfg.ac, ref.V(1), ref.h(1), 1, ref.gam(1));
ac.refcur = [ref.x(1) ref.y(1) ref.h(1) pr.psi(2) ref.V(1) ref.gam(1) + al];
ac.mpc.t = ac.mpc.t + info.time; ac.mpc.n = ac.mpc.n + 1; ac.mpc.last_t = info.time;
ac.mpc.solver = info.solver; ac.mpc.last_reason = info.reason;
if info.fallback, ac.mpc.nfb = ac.mpc.nfb + 1; end
end

function [cmd, ctl] = inner_loop(ac, cfg)
% Attitude/rate autopilot: [phi_cmd gamma_cmd V_cmd] -> [throttle elevator aileron rudder]
xh = ac.E.x; ctl = ac.ctl; u3 = ctl.u3; beh = ac.beh; ic = cfg.ic; g = cfg.g; dt = cfg.dt;
wm = wind_mean_ned(xh(12), cfg); Rn = dcm_b2n(xh(7),xh(8),xh(9));
Va = max(norm(xh(1:3) - Rn'*wm), 30);
phi = xh(7); th = xh(8); P = xh(4); Q = xh(5); R = xh(6);
phic = clp(u3(1), -beh.phi_max, beh.phi_max); gamc = clp(u3(2), -beh.gam_max, beh.gam_max);
Vc = clp(u3(3), cfg.lim.Vcmd_min, cfg.lim.Vcmd_max);
n = min(1/max(cos(phi), 0.5), 2);
pe = cfg.ac; pe.eh = ac.f.eh;
[alt_, de_t, thr_ff] = compute_trim(pe, Va, xh(12), n, gamc);
de = de_t + ic.kth*(th - (gamc + alt_)) + ic.kq*Q;
da = ic.kphi*(phic - phi) - ic.kp*P;
dr = ic.kdr*(R - g*tan(clp(phi,-1,1))/Va);
eV = Vc - Va; lim_i = ic.int_lim/ic.Ki;
ctl.int_V = clp(ctl.int_V + eV*dt, -lim_i, lim_i);
thr = thr_ff + ic.Kv*eV + ic.Ki*ctl.int_V;
cmd = [clp(thr,0,1); de; da; dr];
if any(~isfinite(cmd)), cmd = ac.act; end
end

function ac = control_step(ac, cfg, t)
xh = ac.E.x;
ac.fp = waypoint_manager(ac.fp, 'update', xh(10:12)', xh(9), cfg);
if t >= ac.t_next_mpc - 1e-9
    ac = run_mpc_cycle(ac, cfg);
    ac.t_next_mpc = ac.t_next_mpc + cfg.dt_mpc;
end
[ac.cmd, ac.ctl] = inner_loop(ac, cfg);
end

%% ======================================================================
%  WORLD STEP, SAFETY, LOGGING
%% ======================================================================
function W = step_world(W, t)
cfg = W.cfg; dt = cfg.dt; na = numel(W.ac); k = W.k;
W = inject_failures(W, t);
doEM = mod(k-1, round(cfg.dt_em/dt)) == 0;
for i = 1:na
    ac = W.ac(i);
    if ~ac.alive, continue; end
    [ac.wn, ac.wst] = wind_model(ac.wst, ac.x(12), norm(ac.x(1:3)), dt, cfg);
    p_true = ac.p; p_true.eh = ac.f.eh;
    [~, fb] = nasa_wind_dynamics(ac.x, ac.ueff, p_true, ac.wn);
    [S, ac.sens] = sensor_model(ac.sens, ac.x, fb, ac.wn, t, cfg, ac.f);
    ts = tic; ac = estimator_step(ac, S, t, cfg); ac.t_ekf = ac.t_ekf + toc(ts); ac.n_ekf = ac.n_ekf + 1;
    if doEM
        ac = update_emergency(ac, t, cfg);
        if ~isempty(ac.pending_ev)
            W = log_event(W, t, sprintf('%s: %s', ac.name, ac.pending_ev)); ac.pending_ev = '';
            W.em_events = W.em_events + 1;
        end
    end
    ac = control_step(ac, cfg, t);
    [ac.act, ac.sat] = actuator_model(ac.act, ac.cmd, cfg.act, dt, ac.f);
    ac.ueff = [ac.act(1); ac.act(2:4).*ac.f.eff(2:4)'];
    ueff = ac.ueff; wn = ac.wn;
    fdyn = @(xx) aircraft_dynamics(xx, ueff, p_true, wn);
    xn = rk4_step(fdyn, ac.x, dt);
    [xn, bad] = sanitize_state(xn, ac.x);
    ac.x = xn;
    ac.n_steps = ac.n_steps + 1; ac.sat_cnt = ac.sat_cnt + double(ac.sat(:)); ac.use_sum = ac.use_sum + abs(ac.act(:));
    W.ac(i) = ac;
    W = safety_monitor(W, i, bad, t);
end
W = surveillance_step(W, t);
if mod(k-1, round(cfg.dt_atc/dt)) == 0, W = atc_cycle(W, t); end
W = separation_monitor(W, t);
if mod(k-1, round(cfg.dt_log/dt)) == 0, W = log_step(W, t); end
end

function W = safety_monitor(W, i, bad, t)
cfg = W.cfg; ac = W.ac(i); x = ac.x; lim = cfg.lim;
Rn = dcm_b2n(x(7),x(8),x(9)); wb = Rn'*ac.wn;
Va = norm(x(1:3) - wb); alpha = atan2(x(3) - wb(3), max(x(1) - wb(1), 1));
chk = struct('bank', abs(x(7)) > lim.phi_hard, 'alpha', abs(alpha) > lim.alpha_max, ...
    'speed', Va < lim.Vstall_warn || Va > 1.4*lim.Vcmd_max, 'ground', x(12) <= 0.5);
names = fieldnames(chk);
for q = 1:numel(names)
    nm = names{q};
    if chk.(nm) && ~ac.vf.(nm)
        W.viol.(nm) = W.viol.(nm) + 1; W = log_event(W, t, sprintf('SAFETY VIOLATION %s: %s', ac.name, nm));
    end
    ac.vf.(nm) = chk.(nm);
end
if bad, W.viol.clamp = W.viol.clamp + 1; end
if chk.ground, ac.alive = false; W = log_event(W, t, sprintf('%s GROUND IMPACT - aircraft removed', ac.name)); end
if ~isempty(W.zones)
    [ins, ~] = restricted_zone_checker(W.zones, [x(10) x(11) x(12)], 0);
    inz = any(ins);
    if inz && ~ac.vf.zone
        W.viol.zone = W.viol.zone + 1; W.zone_alert(ins) = 2;
        W = log_event(W, t, sprintf('SAFETY VIOLATION %s: restricted-zone intrusion', ac.name));
    end
    ac.vf.zone = inz;
end
W.ac(i) = ac;
end

function W = separation_monitor(W, t)
cfg = W.cfg; na = numel(W.ac); in_los_any = false(1,na);
for i = 1:na-1
    for j = i+1:na
        if ~W.ac(i).alive || ~W.ac(j).alive, continue; end
        xi = W.ac(i).x; xj = W.ac(j).x;
        dh = hypot(xi(10) - xj(10), xi(11) - xj(11)); dv = abs(xi(12) - xj(12)); d3 = sqrt(dh^2 + dv^2);
        W.sep.minH = min(W.sep.minH, dh); W.sep.minV = min(W.sep.minV, dv); W.sep.min3D = min(W.sep.min3D, d3);
        los = dh < cfg.sep.H && dv < cfg.sep.V;
        if los
            W.sep.los_samples = W.sep.los_samples + 1; in_los_any([i j]) = true; W.atc.los(i,j) = true;
            if ~W.sep.in_los(i,j)
                W.sep.los_events = W.sep.los_events + 1; W.viol.los = W.viol.los + 1;
                if ~W.atc.act(i,j), W.atc.stats.unresolved = W.atc.stats.unresolved + 1; end
                W = log_event(W, t, sprintf('LOSS OF SEPARATION %s-%s: dh=%.0f m dv=%.0f m', W.ac(i).name, W.ac(j).name, dh, dv));
            end
        end
        W.sep.in_los(i,j) = los;
    end
end
for i = 1:na, W.ac(i).in_los = in_los_any(i); end
end

function W = log_step(W, t)
na = numel(W.ac); n = W.nlog + 1; if n > numel(W.L.t), return; end
W.nlog = n; W.L.t(n) = t; mh = inf;
for i = 1:na
    ac = W.ac(i);
    W.L.x(:,n,i) = ac.x; W.L.xh(:,n,i) = ac.E.x; W.L.u(:,n,i) = ac.act;
    W.L.ref(:,n,i) = ac.refcur(:); W.L.mode(n,i) = ac.mode; W.L.sat(:,n,i) = double(ac.sat(:));
    for j = i+1:na
        if ac.alive && W.ac(j).alive, mh = min(mh, hypot(ac.x(10) - W.ac(j).x(10), ac.x(11) - W.ac(j).x(11))); end
    end
end
W.L.minH(n) = mh;
end

%% ======================================================================
%  METRICS, REPORT, MONTE CARLO
%% ======================================================================
function M = calculate_performance_metrics(W)
cfg = W.cfg; na = numel(W.ac); n = W.nlog; L = W.L; idx = find(L.t(1:n) > 5);
if isempty(idx), idx = 1:n; end
M.na = na; M.T = W.t_end; M.names = {W.ac.name};
M.pos_rmse = zeros(1,na); M.alt_rmse = M.pos_rmse; M.hdg_rmse = M.pos_rmse; M.spd_rmse = M.pos_rmse;
M.pitch_rmse = M.pos_rmse; M.ekf_rmse = zeros(12,na); M.ekf_pos_rmse = M.pos_rmse; M.ekf_alt_rmse = M.pos_rmse;
M.ekf_hdg_rmse = M.pos_rmse; M.ekf_spd_rmse = M.pos_rmse; M.act_use = zeros(4,na); M.sat_pct = zeros(4,na);
for i = 1:na
    X = squeeze(L.x(:,idx,i)); Xh = squeeze(L.xh(:,idx,i)); Rf = squeeze(L.ref(:,idx,i));
    if size(X,1) ~= 12, X = X'; Xh = Xh'; Rf = Rf'; end
    Va = sqrt(sum(X(1:3,:).^2,1));
    M.pos_rmse(i) = sqrt(mean((X(10,:) - Rf(1,:)).^2 + (X(11,:) - Rf(2,:)).^2));
    M.alt_rmse(i) = sqrt(mean((X(12,:) - Rf(3,:)).^2));
    M.hdg_rmse(i) = sqrt(mean(wrapToPi(X(9,:) - Rf(4,:)).^2));
    M.spd_rmse(i) = sqrt(mean((Va - Rf(5,:)).^2));
    M.pitch_rmse(i) = sqrt(mean((X(8,:) - Rf(6,:)).^2));
    err = Xh - X; err(7:9,:) = wrapToPi(err(7:9,:));
    M.ekf_rmse(:,i) = sqrt(mean(err.^2, 2));
    M.ekf_pos_rmse(i) = sqrt(mean(err(10,:).^2 + err(11,:).^2)); M.ekf_alt_rmse(i) = M.ekf_rmse(12,i);
    M.ekf_hdg_rmse(i) = M.ekf_rmse(9,i); M.ekf_spd_rmse(i) = sqrt(mean((sqrt(sum(Xh(1:3,:).^2,1)) - Va).^2));
    U = squeeze(L.u(:,1:n,i)); if size(U,1) ~= 4, U = U'; end
    M.act_use(:,i) = mean(abs(U), 2);
    M.sat_pct(:,i) = 100*W.ac(i).sat_cnt/max(W.ac(i).n_steps,1);
end
M.minH = W.sep.minH; M.minV = W.sep.minV; M.min3D = W.sep.min3D; M.los_events = W.sep.los_events;
st = W.atc.stats; still = sum(sum(W.atc.act));
M.conflicts = st.detected; M.warnings = st.warnings; M.criticals = st.criticals; M.resolved = st.resolved;
M.unresolved = st.unresolved + still; M.maneuvers = st.maneuvers; M.recoveries = st.recoveries;
M.zone_replans = st.zone_replans; M.viol = W.viol;
M.viol_total = W.viol.los + W.viol.zone + W.viol.bank + W.viol.alpha + W.viol.speed + W.viol.ground + W.viol.clamp + W.viol.nan;
M.mpc_time_mean = mean(arrayfun(@(a) a.mpc.t/max(a.mpc.n,1), W.ac)); M.mpc_time_max = max(arrayfun(@(a) a.mpc.last_t, W.ac));
M.ekf_time_mean = mean(arrayfun(@(a) a.t_ekf/max(a.n_ekf,1), W.ac));
M.mpc_fallbacks = sum(arrayfun(@(a) a.mpc.nfb, W.ac)); M.mpc_calls = sum(arrayfun(@(a) a.mpc.n, W.ac));
M.ekf_resets = sum(arrayfun(@(a) a.E.resets, W.ac)); M.em_events = W.em_events;
M.ekf_rej = sum(cell2mat(arrayfun(@(a) a.E.rej(:), W.ac, 'UniformOutput', false)), 2)';
M.crashed = sum(~arrayfun(@(a) a.alive, W.ac));
M.pass = (M.los_events == 0) && (W.viol.zone == 0) && (M.crashed == 0) && (W.viol.ground == 0);
end

function txt = generate_test_report(M, cfg, W)
L = {};
L{end+1} = repmat('=', 1, 78);
L{end+1} = sprintf('TEST REPORT  scenario: %s   (research simulation prototype - not certified)', cfg.scenario);
L{end+1} = repmat('=', 1, 78);
L{end+1} = sprintf('Duration: %.1f s | Aircraft: %d | Wall time: %.1f s | seed %d', M.T, M.na, W.wall_time, cfg.seed);
L{end+1} = sprintf('Conflicts detected: %d (warnings %d, critical %d) | resolved %d | unresolved %d | maneuvers %d | recoveries %d', ...
    M.conflicts, M.warnings, M.criticals, M.resolved, M.unresolved, M.maneuvers, M.recoveries);
L{end+1} = sprintf('Min separation: horizontal %.0f m | vertical %.0f m | 3D %.0f m | loss-of-separation events %d', ...
    M.minH, M.minV, M.min3D, M.los_events);
L{end+1} = sprintf('Restricted zones: replans %d | violations %d', M.zone_replans, M.viol.zone);
L{end+1} = sprintf('Safety violations (total %d): LoS %d, zone %d, bank %d, alpha %d, speed %d, ground %d, state-clamp %d', ...
    M.viol_total, M.viol.los, M.viol.zone, M.viol.bank, M.viol.alpha, M.viol.speed, M.viol.ground, M.viol.clamp);
L{end+1} = sprintf('MPC: calls %d | fallbacks %d | mean %.2f ms | max %.2f ms | EKF: mean %.3f ms, resets %d | emergency/mode events %d', ...
    M.mpc_calls, M.mpc_fallbacks, 1e3*M.mpc_time_mean, 1e3*M.mpc_time_max, 1e3*M.ekf_time_mean, M.ekf_resets, M.em_events);
L{end+1} = sprintf('EKF innovation rejections [gps baro mag asp gyro accel uplink]: %s', mat2str(M.ekf_rej));
L{end+1} = repmat('-', 1, 78);
L{end+1} = sprintf('%-6s %8s %8s %8s %8s %8s | %8s %8s %8s', 'AC','posRMSE','altRMSE','hdgRMSE','spdRMSE','pitRMSE','EKFpos','EKFalt','EKFhdg');
for i = 1:M.na
    L{end+1} = sprintf('%-6s %7.1fm %7.1fm %6.2fdeg %6.2fm/s %6.2fdeg | %7.1fm %7.1fm %6.2fdeg', M.names{i}, M.pos_rmse(i), ...
        M.alt_rmse(i), M.hdg_rmse(i)*180/pi, M.spd_rmse(i), M.pitch_rmse(i)*180/pi, M.ekf_pos_rmse(i), M.ekf_alt_rmse(i), ...
        M.ekf_hdg_rmse(i)*180/pi); %#ok<AGROW>
end
for i = 1:M.na
    L{end+1} = sprintf('%-6s actuator mean|pos| thr %.2f de %.3f da %.3f dr %.3f | saturation %% thr %.1f de %.1f da %.1f dr %.1f', ...
        M.names{i}, M.act_use(:,i), M.sat_pct(:,i)); %#ok<AGROW>
end
if isfield(W.atc, 'ml_conf') && ~isempty(W.atc.ml_conf)
    n_ml = numel(W.atc.ml_conf);
    n_crit = sum([W.atc.ml_conf.warning_level] == 2);
    n_dis = sum([W.atc.ml_conf.disagreement]);
    L{end+1} = sprintf('AI Conflict Predictor (Dual-Monitor): %d alerts (Critical: %d) | Rule Disagreements: %d', ...
        n_ml, n_crit, n_dis);
end
L{end+1} = sprintf('Result: %s', tern(M.pass, 'PASS (no loss of separation, zone violation or crash)', 'FAIL / CHECK EVENTS'));
ev = W.events; ne = size(ev,1);
if ne > 0
    L{end+1} = repmat('-', 1, 78); L{end+1} = 'EVENT LOG (last 25):';
    for q = max(1, ne-24):ne, L{end+1} = sprintf('  t=%6.1fs  %s', ev{q,1}, ev{q,2}); end %#ok<AGROW>
end
txt = strjoin(L, newline);
end

function cfg = randomize_config(cfg, seed)
rng(seed); cfg.seed = seed;
for i = 1:numel(cfg.aircraft)
    p = cfg.ac;
    p.m = p.m*(1 + 0.10*(2*rand - 1)); p.CLa = p.CLa*(1 + 0.08*(2*rand - 1));
    p.Cma = p.Cma*(1 + 0.10*(2*rand - 1)); p.Tmax = p.Tmax*(1 + 0.10*(2*rand - 1));
    p.CD0 = p.CD0*(1 + 0.10*(2*rand - 1)); p.Iyy = p.Iyy*(1 + 0.10*(2*rand - 1));
    cfg.aircraft(i).p = p;
    cfg.aircraft(i).route(1,1:2) = cfg.aircraft(i).route(1,1:2) + 200*(2*rand(1,2) - 1);
    cfg.aircraft(i).route(2:end,2) = cfg.aircraft(i).route(2:end,2) + 150*(2*rand - 1);
end
cfg.wind.speed = cfg.wind.speed*(0.5 + rand); cfg.wind.dir_from = 2*pi*rand;
cfg.wind.turb = cfg.wind.turb*(0.5 + 1.5*rand); cfg.sens.noise_scale = 0.8 + 1.7*rand;
cfg.act.tau = cfg.act.tau.*(1 + 0.3*(2*rand(4,1) - 1)); cfg.act.rate = cfg.act.rate.*(1 + 0.2*(2*rand(4,1) - 1));
cfg.act.hi(2:4) = cfg.act.hi(2:4)*(1 + 0.1*(2*rand - 1)); cfg.act.lo(2:4) = -cfg.act.hi(2:4);
for k = 1:numel(cfg.fail)
    sh = 10*(2*rand - 1); cfg.fail(k).t_on = max(0, cfg.fail(k).t_on + sh);
end
end

function MC = run_monte_carlo(cfg0)
mc = cfg0.mc; names = mc.scenarios; ns = numel(names); nt = mc.trials;
MC.rows = []; MC.names = names; MC.trials = nt;
passN = zeros(1,ns); minH = nan(ns,nt); los = nan(ns,nt); ekf = nan(ns,nt); fbk = nan(ns,nt); trk = nan(ns,nt);
fprintf('\nMONTE-CARLO: %d scenarios x %d trials (seed0 %d)\n', ns, nt, mc.seed0);
for s = 1:ns
    for k = 1:nt
        cfg = apply_scenario(cfg0, names{s}); cfg.T = mc.T; cfg.viz.enable = false;
        cfg = randomize_config(cfg, mc.seed0 + 1000*s + k);
        try
            R = run_simulation(cfg); M = R.M;
            passN(s) = passN(s) + M.pass; minH(s,k) = M.minH; los(s,k) = M.los_events;
            ekf(s,k) = mean(M.ekf_pos_rmse); fbk(s,k) = M.mpc_fallbacks; trk(s,k) = mean(M.pos_rmse);
        catch ME
            fprintf('  trial %s/%d crashed: %s\n', names{s}, k, ME.message);
        end
    end
    fprintf('  %-22s pass %2d/%2d | minH mean %.0f m (min %.0f) | LoS events %.1f | EKF pos RMSE %.1f m | MPC fallbacks %.1f\n', ...
        names{s}, passN(s), nt, nanmean_(minH(s,:)), min(minH(s,:)), nanmean_(los(s,:)), nanmean_(ekf(s,:)), nanmean_(fbk(s,:)));
end
MC.pass = passN; MC.minH = minH; MC.los = los; MC.ekf_pos_rmse = ekf; MC.fallbacks = fbk; MC.track_rmse = trk;
MC.success_rate = sum(passN)/(ns*nt);
fprintf('Overall success rate: %.1f %%\n', 100*MC.success_rate);
end

function m = nanmean_(x)
x = x(isfinite(x)); if isempty(x), m = NaN; else, m = mean(x); end
end

function print_batch_summary(res)
fprintf('\n%s\nBATCH SUMMARY\n%s\n', repmat('=',1,78), repmat('=',1,78));
fprintf('%-26s %5s %8s %6s %6s %9s %5s\n', 'scenario','LoS','minH[m]','confl','unres','viol','pass');
for k = 1:numel(res)
    M = res{k}.M;
    fprintf('%-26s %5d %8.0f %6d %6d %9d %5s\n', res{k}.cfg.scenario, M.los_events, M.minH, M.conflicts, ...
        M.unresolved, M.viol_total, tern(M.pass,'yes','NO'));
end
end

function print_final_states(W)
fprintf('\nFINAL AIRCRAFT STATES (t = %.1f s)\n', W.t_end);
fprintf('%-5s %-20s %9s %9s %8s %7s %7s %6s %6s %6s\n','ID','mode','N[m]','E[m]','alt[m]','Va','hdg','roll','pitch','alive');
for i = 1:numel(W.ac)
    ac = W.ac(i); x = ac.x; Va = norm(x(1:3) - dcm_b2n(x(7),x(8),x(9))'*ac.wn);
    fprintf('%-5s %-20s %9.0f %9.0f %8.0f %7.1f %6.1f %5.1f %5.1f %6d\n', ac.name, W.cfg.modes{ac.mode}, x(10), x(11), ...
        x(12), Va, mod(x(9)*180/pi,360), x(7)*180/pi, x(8)*180/pi, ac.alive);
end
end

function a = merge_struct(a, b)
f = fieldnames(b);
for k = 1:numel(f)
    if isfield(a, f{k}) && isstruct(a.(f{k})) && isstruct(b.(f{k})) && isscalar(a.(f{k})) && isscalar(b.(f{k}))
        a.(f{k}) = merge_struct(a.(f{k}), b.(f{k}));
    else
        a.(f{k}) = b.(f{k});
    end
end
end

%% ======================================================================
%  LIVE VISUALIZATION (single figure: 3D world + radar + HUD + panels)
%% ======================================================================
function [V, F] = build_aircraft_mesh()
% Body frame: x forward, y right, z down [m]. Quad faces: fuselage, wings, tail.
xr = [11 8 4 -4 -10 -13]; rr = [0.05 1.1 1.5 1.5 0.9 0.25]; n = 8; th = (0:n-1)*2*pi/n;
V = zeros(6*n,3);
for k = 1:6, V((k-1)*n + (1:n),:) = [xr(k)*ones(n,1), rr(k)*cos(th)', rr(k)*sin(th)']; end
F = zeros(5*n,4);
for k = 1:5
    for j = 1:n
        a = (k-1)*n + j; b = (k-1)*n + mod(j,n) + 1; c = k*n + mod(j,n) + 1; d = k*n + j;
        F((k-1)*n + j,:) = [a b c d];
    end
end
nv = size(V,1);
Wl = [3 -1.3 0.3; -2 -9 0.1; -4 -9 0.1; -1 -1.3 0.3]; Wr = Wl; Wr(:,2) = -Wl(:,2);
Hl = [-9 -1 0; -11.5 -5 0; -13 -5 0; -11 -1 0];       Hr = Hl; Hr(:,2) = -Hl(:,2);
Vt = [-8 0 -1.2; -11.5 0 -5.5; -13 0 -5.5; -12 0 -1.2];
parts = {Wl, Wr, Hl, Hr, Vt};
for q = 1:numel(parts)
    V = [V; parts{q}]; F = [F; nv+1 nv+2 nv+3 nv+4]; nv = nv + 4; %#ok<AGROW>
end
% engine nacelles slung under each wing (graphics add-on)
for sgn = [-1 1]
    [bx, by, bz] = ndgrid([2.4 -2.8], sgn*3.6 + [-0.7 0.7], [0.1 1.5]);
    V = [V; bx(:) by(:) bz(:)]; %#ok<AGROW>
    F = [F; nv + [1 3 7 5; 2 4 8 6; 1 2 6 5; 3 4 8 7; 1 2 4 3; 5 6 8 7]]; nv = nv + 8; %#ok<AGROW>
end
end

function M = hgmat(R3, pos)
M = eye(4); M(1:3,1:3) = R3; M(1:3,4) = pos(:);
end

function viz_key(src, evt)
c = getappdata(src, 'ctl'); if isempty(c), return; end
switch evt.Key
    case {'1','2','3','4','5','6','7','8','9'}, c.focus = str2double(evt.Key);
    case 'c'
        modes = {'fixed','follow','orbit','top','auto'}; k = find(strcmp(modes, c.cam)); c.cam = modes{mod(k,5) + 1}; c.cam_changed = true;
    case 'r', c.radar = ~c.radar;
    case 't', c.hud = ~c.hud;
    case 'p', c.pred = ~c.pred;
    case 'm', c.perf = ~c.perf;
    case 'g', c.pfd = ~c.pfd;
    case 'a', c.anim = ~c.anim;
    case 'space', c.pause = ~c.pause;
    case 'x', c.stop = true;
end
setappdata(src, 'ctl', c);
end

function viz_close(src, ~)
c = getappdata(src, 'ctl'); if ~isempty(c), c.stop = true; setappdata(src, 'ctl', c); end
delete(src);
end

function G = init_visualization(W, cfg)
V = cfg.viz; na = numel(W.ac); zs = V.z_exag; ext = V.ext; bg = [0.06 0.07 0.09]; fg = [0.85 0.88 0.9];
G.fig = figure('Name','Unified Aircraft Flight Control & Autonomous ATC - research prototype', ...
    'NumberTitle','off','Color',bg,'Position',[30 30 1560 880],'KeyPressFcn',@viz_key, ...
    'CloseRequestFcn',@viz_close,'Renderer','opengl');
setappdata(G.fig,'ctl',struct('focus',1,'cam',V.camera,'radar',V.show_radar,'hud',V.show_hud,'pred',V.show_pred, ...
    'perf',V.show_perf,'pfd',V.show_pfd,'anim',V.animate,'pause',false,'stop',false,'cam_changed',true));
G.zs = zs; G.na = na; G.last = struct('radar',true,'hud',true,'perf',true,'pred',true,'pfd',true);
% ---------- 3D axes ----------
ax = axes('Parent',G.fig,'Position',[0.02 0.27 0.58 0.71],'Color',[0.08 0.10 0.14],'XColor',fg,'YColor',fg,'ZColor',fg, ...
    'GridColor',[0.6 0.6 0.6],'GridAlpha',0.25,'Box','on'); G.ax = ax; hold(ax,'on'); grid(ax,'on');
xlim(ax, ext(1:2)); ylim(ax, ext(3:4)); zlim(ax, [0 ext(5)*zs]); daspect(ax,[1 1 1]);
xlabel(ax,'East (km)'); ylabel(ax,'North (km)'); zlabel(ax,sprintf('Altitude (km x%g)',zs));
view(ax,-38,24); light('Parent',ax,'Position',[-1 -1 1],'Style','infinite'); set(ax,'Clipping','on');
G.ttl = title(ax,'LIVE 3D AIRSPACE  -  research simulation prototype (not a certified system)','Color',fg,'FontSize',10);
patch('Parent',ax,'XData',[ext(1) ext(2) ext(2) ext(1)],'YData',[ext(3) ext(3) ext(4) ext(4)],'ZData',[0 0 0 0], ...
    'FaceColor',[0.10 0.18 0.12],'FaceAlpha',0.9,'EdgeColor','none');
gx = ceil(ext(1)/2)*2:2:ext(2); gy = ceil(ext(3)/2)*2:2:ext(4); gX = []; gY = [];
for q = gx, gX = [gX q q nan]; gY = [gY ext(3) ext(4) nan]; end %#ok<AGROW>
for q = gy, gX = [gX ext(1) ext(2) nan]; gY = [gY q q nan]; end %#ok<AGROW>
line('Parent',ax,'XData',gX,'YData',gY,'ZData',zeros(size(gX)),'Color',[0.3 0.4 0.3],'LineWidth',0.5);
% ---------- translucent altitude reference planes (graphics add-on) ----------
if V.altitude_planes
    for ha = [1500 3000 4500]
        zz = ha/1000*zs;
        patch('Parent',ax,'XData',[ext(1) ext(2) ext(2) ext(1)],'YData',[ext(3) ext(3) ext(4) ext(4)],'ZData',zz*ones(1,4), ...
            'FaceColor',[0.3 0.5 0.8],'FaceAlpha',0.035,'EdgeColor',[0.3 0.45 0.7],'EdgeAlpha',0.25,'LineWidth',0.4);
        text('Parent',ax,'Position',[ext(2) ext(3) zz],'String',sprintf(' %.1f km',ha/1000),'Color',[0.5 0.65 0.85],'FontSize',7);
    end
end
% ---------- 3D radar site: range rings + rotating sweep wedge on the ground (graphics add-on) ----------
G.rc = [cfg.radar.pos(2) cfg.radar.pos(1)]/1000; G.sw3 = gobjects(0); thr = linspace(0, 2*pi, 90);
for rrk = [4.5 9 13.5]
    line('Parent',ax,'XData',G.rc(1) + rrk*sin(thr),'YData',G.rc(2) + rrk*cos(thr),'ZData',0.004*ones(size(thr)),'Color',[0.2 0.6 0.35],'LineWidth',0.6);
end
line('Parent',ax,'XData',G.rc(1),'YData',G.rc(2),'ZData',0.01,'LineStyle','none','Marker','^','MarkerSize',9,'MarkerFaceColor',[0.3 1 0.5],'Color',[0.3 1 0.5]);
if V.radar_sweep3d
    G.sw3 = patch('Parent',ax,'XData',[G.rc(1); G.rc(1); G.rc(1)],'YData',[G.rc(2); G.rc(2); G.rc(2)],'ZData',0.006*ones(3,1), ...
        'FaceColor',[0.1 1 0.4],'FaceAlpha',0.16,'EdgeColor','none');
end
% ---------- compass rose on the ground (graphics add-on) ----------
cx = ext(1) + 3.2; cy = ext(3) + 3.2; cr = 2.0; zc = 0.003; thc = linspace(0, 2*pi, 73);
line('Parent',ax,'XData',cx + cr*cos(thc),'YData',cy + cr*sin(thc),'ZData',zc*ones(size(thc)),'Color',[0.55 0.85 0.65],'LineWidth',1.2);
line('Parent',ax,'XData',cx + 0.7*cr*cos(thc),'YData',cy + 0.7*cr*sin(thc),'ZData',zc*ones(size(thc)),'Color',[0.35 0.6 0.45],'LineWidth',0.6);
rX = []; rY = [];
for a = 0:30:330
    r1 = cr*(1 - 0.12*(mod(a,90) == 0)); aa = a*pi/180;
    rX = [rX cx + r1*sin(aa) cx + 1.1*cr*sin(aa) nan]; rY = [rY cy + r1*cos(aa) cy + 1.1*cr*cos(aa) nan]; %#ok<AGROW>
end
line('Parent',ax,'XData',rX,'YData',rY,'ZData',zc*ones(size(rX)),'Color',[0.55 0.85 0.65],'LineWidth',0.8);
patch('Parent',ax,'XData',cx + [0 -0.22 0.22]*cr,'YData',cy + [1.0 0.45 0.45]*cr,'ZData',zc*ones(1,3), ...
    'FaceColor',[1 0.3 0.3],'EdgeColor','none');
text('Parent',ax,'Position',[cx cy + 1.35*cr zc],'String','N','Color',[1 0.45 0.45],'FontSize',9,'FontWeight','bold','HorizontalAlignment','center');
text('Parent',ax,'Position',[cx + 1.3*cr cy zc],'String','E','Color',[0.7 0.9 0.8],'FontSize',8,'HorizontalAlignment','center');
text('Parent',ax,'Position',[cx cy - 1.35*cr zc],'String','S','Color',[0.7 0.9 0.8],'FontSize',8,'HorizontalAlignment','center');
text('Parent',ax,'Position',[cx - 1.3*cr cy zc],'String','W','Color',[0.7 0.9 0.8],'FontSize',8,'HorizontalAlignment','center');
% ---------- zones ----------
G.zone = gobjects(1, numel(W.zones));
for z = 1:numel(W.zones)
    Zn = W.zones(z); hmin = max(Zn.hmin,0); hmax = min(Zn.hmax, ext(5)*1000);
    if strcmp(Zn.type,'polygon'), P = Zn.poly; else
        th = linspace(0,2*pi,41)'; P = [Zn.center(1) + Zn.radius*cos(th), Zn.center(2) + Zn.radius*sin(th)];
    end
    E = P(:,2)/1000; N = P(:,1)/1000; m = size(P,1); z0 = hmin/1000*zs; z1 = hmax/1000*zs;
    Xs = []; Ys = []; Zs = [];
    for e = 1:m
        e2 = mod(e,m) + 1; Xs = [Xs [E(e);E(e2);E(e2);E(e)]]; Ys = [Ys [N(e);N(e2);N(e2);N(e)]]; Zs = [Zs [z0;z0;z1;z1]]; %#ok<AGROW>
    end
    G.zone(z) = patch('Parent',ax,'XData',Xs,'YData',Ys,'ZData',Zs,'FaceColor',[0.9 0.15 0.15],'FaceAlpha',0.18, ...
        'EdgeColor',[1 0.3 0.3],'LineWidth',0.5);
    patch('Parent',ax,'XData',E,'YData',N,'ZData',z1*ones(m,1),'FaceColor',[0.9 0.15 0.15],'FaceAlpha',0.12,'EdgeColor',[1 0.3 0.3]);
    text('Parent',ax,'Position',[mean(E) mean(N) z1],'String',Zn.name,'Color',[1 0.5 0.5],'FontSize',8,'HorizontalAlignment','center');
end
% ---------- aircraft ----------
[Vb, Fb] = build_aircraft_mesh(); Vb = Vb*V.ac_scale/1000;
G.hgt = gobjects(1,na); G.mesh = G.hgt; G.trueL = G.hgt; G.ekfL = G.hgt; G.predL = G.hgt; G.planL = G.hgt;
G.mk = G.hgt; G.wpL = G.hgt; G.destL = G.hgt; G.velL = G.hgt; G.hdgL = G.hgt; G.sepL = G.hgt; G.txt = G.hgt; G.beacon = G.hgt;
G.tr_true = nan(V.trail_len,3,na); G.tr_ekf = nan(V.trail_len,3,na); G.tr_i = 0;
G.trSeg = gobjects(na, V.trail_segs); G.navP = gobjects(1,na); G.navS = G.navP; G.strobe = G.navP; G.camT = [nan nan nan];
G.alert = text('Parent',ax,'Units','normalized','Position',[0.5 0.94 0],'String','','Color',[1 0.25 0.25],'FontSize',15,'FontWeight','bold', ...
    'HorizontalAlignment','center','BackgroundColor',[0.10 0.02 0.02],'EdgeColor',[1 0.25 0.25],'Margin',5,'Visible','off','Clipping','off');
th = linspace(0,2*pi,25); rr = cfg.sep.H/2/1000; hv = cfg.sep.V/2/1000*zs;
sx = [rr*cos(th) nan rr*cos(th) nan]; sy = [rr*sin(th) nan rr*sin(th) nan]; sz = [-hv*ones(1,25) nan hv*ones(1,25) nan];
for q = 1:4, a = (q-1)*pi/2; sx = [sx rr*cos(a) rr*cos(a) nan]; sy = [sy rr*sin(a) rr*sin(a) nan]; sz = [sz -hv hv nan]; end %#ok<AGROW>
G.sepX = sx; G.sepY = sy; G.sepZ = sz;
for i = 1:na
    c = W.ac(i).color;
    G.hgt(i) = hgtransform('Parent',ax);
    G.mesh(i) = patch('Parent',G.hgt(i),'Faces',Fb,'Vertices',Vb,'FaceColor',[0.2 0.9 0.3],'EdgeColor',[0.05 0.05 0.05], ...
        'LineWidth',0.5,'FaceLighting','flat','AmbientStrength',0.6,'DiffuseStrength',0.7);
    nsg = V.trail_segs; bg0 = [0.08 0.10 0.14];
    for q = 1:nsg      % fading trail: oldest segment dim and thin, newest bright and thick
        G.trSeg(i,q) = line('Parent',ax,'XData',nan,'YData',nan,'ZData',nan,'Color',bg0 + (c - bg0)*(0.15 + 0.85*q/nsg),'LineWidth',1 + 2.4*q/nsg);
    end
    G.trueL(i) = G.trSeg(i,nsg);
    sc3 = V.ac_scale/1000;      % navigation lights + tail strobe, parented to the aircraft transform
    G.navP(i) = line('Parent',G.hgt(i),'XData',-3*sc3,'YData',-9*sc3,'ZData',0.1*sc3,'LineStyle','none','Marker','o','MarkerSize',5,'MarkerFaceColor',[1 0.1 0.1],'Color',[1 0.1 0.1]);
    G.navS(i) = line('Parent',G.hgt(i),'XData',-3*sc3,'YData',9*sc3,'ZData',0.1*sc3,'LineStyle','none','Marker','o','MarkerSize',5,'MarkerFaceColor',[0.1 1 0.2],'Color',[0.1 1 0.2]);
    G.strobe(i) = line('Parent',G.hgt(i),'XData',-12.5*sc3,'YData',0,'ZData',-5.6*sc3,'LineStyle','none','Marker','o','MarkerSize',8,'MarkerFaceColor',[1 1 1],'Color',[1 1 1]);
    G.ekfL(i) = line('Parent',ax,'XData',nan,'YData',nan,'ZData',nan,'Color',c,'LineStyle',':','LineWidth',1.3);
    G.predL(i) = line('Parent',ax,'XData',nan,'YData',nan,'ZData',nan,'Color',[0.2 0.85 1],'LineStyle','--','LineWidth',1.2);
    G.planL(i) = line('Parent',ax,'XData',nan,'YData',nan,'ZData',nan,'Color',[1 1 1],'LineStyle','-.','LineWidth',0.8);
    G.wpL(i) = line('Parent',ax,'XData',nan,'YData',nan,'ZData',nan,'LineStyle','none','Marker','d','MarkerSize',6, ...
        'MarkerFaceColor',[1 1 1],'Color',[1 1 1]);
    G.destL(i) = line('Parent',ax,'XData',nan,'YData',nan,'ZData',nan,'LineStyle','none','Marker','p','MarkerSize',13, ...
        'MarkerFaceColor',[1 0.85 0.2],'Color',[1 0.85 0.2]);
    G.velL(i) = line('Parent',ax,'XData',[0 0],'YData',[0 0],'ZData',[0 0],'Color',[1 1 0.4],'LineWidth',1.5);
    G.hdgL(i) = line('Parent',ax,'XData',[0 0],'YData',[0 0],'ZData',[0 0],'Color',[0.4 1 1],'LineWidth',1.5);
    G.sepL(i) = line('Parent',ax,'XData',nan,'YData',nan,'ZData',nan,'Color',[0.5 0.55 0.65],'LineWidth',0.6);
    G.mk(i) = line('Parent',ax,'XData',nan,'YData',nan,'ZData',nan,'LineStyle','none','Marker','o','MarkerSize',16,'LineWidth',2.2,'Color',[0.2 0.9 0.3]);
    G.beacon(i) = line('Parent',ax,'XData',nan,'YData',nan,'ZData',nan,'LineStyle','none','Marker','^','MarkerSize',10, ...
        'MarkerFaceColor',[1 0.2 1],'Color',[1 0.2 1],'Visible','off');
    G.txt(i) = text('Parent',ax,'Position',[0 0 0],'String','','Color',[1 1 1],'FontSize',11,'FontWeight','bold','Interpreter','none','BackgroundColor',[0.05 0.07 0.1],'Margin',2);
end
% ---------- ground shadows + altitude drop lines (graphics add-on) ----------
G.shd = gobjects(1,na); G.drop = gobjects(1,na);
for i = 1:na
    c = W.ac(i).color;
    G.drop(i) = line('Parent',ax,'XData',[0 0],'YData',[0 0],'ZData',[0 0],'Color',0.7*c,'LineStyle',':','LineWidth',0.9);
    G.shd(i) = line('Parent',ax,'XData',nan,'YData',nan,'ZData',nan,'LineStyle','none','Marker','o','MarkerSize',11, ...
        'MarkerFaceColor',[0.02 0.05 0.03],'MarkerEdgeColor',0.6*c);
end
G.last_col = zeros(na,3);
np = max(na*(na-1)/2, 1); G.confL = gobjects(1,np); G.confM = G.confL;
for k = 1:np
    G.confL(k) = line('Parent',ax,'XData',[0 0],'YData',[0 0],'ZData',[0 0],'Color',[1 0 0],'LineWidth',2.5,'Visible','off');
    G.confM(k) = line('Parent',ax,'XData',0,'YData',0,'ZData',0,'LineStyle','none','Marker','o','MarkerSize',14, ...
        'Color',[1 0 0],'LineWidth',2,'Visible','off');
end
[wx, wy, wz] = meshgrid([-8 8], [4 18], [1.5 3.0 5.0]*zs);
G.windQ = quiver3(ax, wx(:), wy(:), wz(:), zeros(numel(wx),1), zeros(numel(wx),1), zeros(numel(wx),1), 0, ...
    'Color',[0.45 0.75 1],'LineWidth',1.2,'MaxHeadSize',0.6);
G.wind_pos = [wx(:) wy(:) wz(:)]; G.wind_t = -inf; G.pred_ver = -1; G.plan_ver = -ones(1,na);
% ---------- radar ----------
rg = V.radar_range_km; axR = axes('Parent',G.fig,'Position',[0.62 0.55 0.37 0.43],'Color',[0 0.04 0.02], ...
    'XColor',[0 0.6 0.2],'YColor',[0 0.6 0.2],'Box','on'); G.axR = axR; hold(axR,'on'); axis(axR,'equal');
xlim(axR,[-rg rg]); ylim(axR,[-rg rg]); set(axR,'XTick',[],'YTick',[]);
title(axR,'ATC RADAR  (N up, track = ADS-B/radar fusion)','Color',[0.3 1 0.5],'FontSize',9);
tt = linspace(0,2*pi,100);
for rr2 = (rg/6):(rg/6):rg
    line('Parent',axR,'XData',rr2*sin(tt),'YData',rr2*cos(tt),'Color',[0 0.45 0.15],'LineWidth',0.6);
    text('Parent',axR,'Position',[0 rr2],'String',sprintf('%gkm',round(rr2)),'Color',[0 0.7 0.25],'FontSize',7,'VerticalAlignment','bottom');
end
line('Parent',axR,'XData',[-rg rg],'YData',[0 0],'Color',[0 0.35 0.12]); line('Parent',axR,'XData',[0 0],'YData',[-rg rg],'Color',[0 0.35 0.12]);
text('Parent',axR,'Position',[0 rg*0.97],'String','N','Color',[0.3 1 0.5],'FontWeight','bold','HorizontalAlignment','center');
for z = 1:numel(W.zones)
    Zn = W.zones(z);
    if strcmp(Zn.type,'polygon'), P = Zn.poly; else
        th2 = linspace(0,2*pi,41)'; P = [Zn.center(1) + Zn.radius*cos(th2), Zn.center(2) + Zn.radius*sin(th2)];
    end
    patch('Parent',axR,'XData',(P(:,2) - cfg.radar.pos(2))/1000,'YData',(P(:,1) - cfg.radar.pos(1))/1000, ...
        'FaceColor',[0.9 0.1 0.1],'FaceAlpha',0.25,'EdgeColor',[1 0.3 0.3]);
end
G.sweep = gobjects(1,3);
for q = 1:3, G.sweep(q) = line('Parent',axR,'XData',[0 0],'YData',[0 0],'Color',[0 1 0.3]*(1 - 0.3*(q-1)),'LineWidth',2 - 0.5*(q-1)); end
G.rBlip = gobjects(1,na); G.rHdg = G.rBlip; G.rTxt = G.rBlip; G.rPred = G.rBlip; G.rTrail = G.rBlip;
G.rtr = nan(60,2,na);
for i = 1:na
    G.rPred(i) = line('Parent',axR,'XData',nan,'YData',nan,'Color',[0.2 0.85 1],'LineStyle','--');
    G.rTrail(i) = line('Parent',axR,'XData',nan,'YData',nan,'Color',[0.2 0.7 0.3],'LineStyle',':');
    G.rHdg(i) = line('Parent',axR,'XData',[0 0],'YData',[0 0],'Color',[0.4 1 0.5],'LineWidth',1.2);
    G.rBlip(i) = line('Parent',axR,'XData',0,'YData',0,'LineStyle','none','Marker','s','MarkerSize',8,'MarkerFaceColor',[0.2 1 0.3],'Color',[0.2 1 0.3]);
    G.rTxt(i) = text('Parent',axR,'Position',[0 0],'String','','Color',[0.7 1 0.7],'FontSize',7,'Interpreter','none');
end
G.radar_all = [axR; allchild(axR)];
% ---------- telemetry HUD and performance panels ----------
G.axH = axes('Parent',G.fig,'Position',[0.62 0.27 0.18 0.26],'Color',[0.03 0.04 0.06],'XColor','none','YColor','none');
xlim(G.axH,[0 1]); ylim(G.axH,[0 1]); hold(G.axH,'on');
G.hud = text('Parent',G.axH,'Position',[0.02 0.98],'String','','Color',[0.7 1 0.8],'FontName','Courier New','FontSize',7.5, ...
    'VerticalAlignment','top','Interpreter','none');
G.hud_all = [G.axH; allchild(G.axH)];
G.axP = axes('Parent',G.fig,'Position',[0.81 0.27 0.18 0.26],'Color',[0.03 0.04 0.06],'XColor','none','YColor','none');
xlim(G.axP,[0 1]); ylim(G.axP,[0 1]); hold(G.axP,'on');
G.perf = text('Parent',G.axP,'Position',[0.02 0.98],'String','','Color',[1 0.9 0.6],'FontName','Courier New','FontSize',7.5, ...
    'VerticalAlignment','top','Interpreter','none');
% ---------- live time-series insets ----------
Nts = ceil(cfg.T/(cfg.dt*V.update_every)) + 20; G.ts.n = 0; G.ts.t = nan(1,Nts); G.ts.minH = nan(1,Nts);
G.ts.ekf = nan(1,Nts); G.ts.trk = nan(1,Nts);
G.axS = axes('Parent',G.fig,'Position',[0.05 0.05 0.17 0.16],'Color',[0.06 0.08 0.1],'XColor',fg,'YColor',fg); hold(G.axS,'on'); grid(G.axS,'on');
G.lSep = line('Parent',G.axS,'XData',nan,'YData',nan,'Color',[0.3 0.9 1],'LineWidth',1.4);
line('Parent',G.axS,'XData',[0 cfg.T],'YData',[cfg.sep.H cfg.sep.H]/1000,'Color',[1 0.3 0.3],'LineStyle','--');
xlim(G.axS,[0 cfg.T]); ylim(G.axS,[0 max(5, ext(4)-ext(3))]); title(G.axS,'min horizontal separation (km)','Color',fg,'FontSize',8);
G.axE = axes('Parent',G.fig,'Position',[0.27 0.05 0.15 0.16],'Color',[0.06 0.08 0.1],'XColor',fg,'YColor',fg); hold(G.axE,'on'); grid(G.axE,'on');
G.lEkf = line('Parent',G.axE,'XData',nan,'YData',nan,'Color',[1 0.8 0.3],'LineWidth',1.4);
G.lTrk = line('Parent',G.axE,'XData',nan,'YData',nan,'Color',[0.6 1 0.5],'LineWidth',1.4);
xlim(G.axE,[0 cfg.T]); title(G.axE,'EKF err (yel) / track err (grn) [m]','Color',fg,'FontSize',8);
% ---------- attitude indicator / artificial horizon for the focus aircraft (graphics add-on) ----------
G.axF = axes('Parent',G.fig,'Position',[0.44 0.02 0.16 0.245],'Color',[0.03 0.04 0.06],'XColor','none','YColor','none');
hold(G.axF,'on'); xlim(G.axF,[-1.1 1.1]); ylim(G.axF,[-1.75 1.1]); daspect(G.axF,[1 1 1]);
thf = linspace(0, 2*pi, 73);
G.pfdSky = patch('Parent',G.axF,'XData',cos(thf),'YData',sin(thf),'FaceColor',[0.25 0.52 0.85],'EdgeColor','none');
G.pfdGnd = patch('Parent',G.axF,'XData',nan,'YData',nan,'FaceColor',[0.55 0.36 0.18],'EdgeColor','none');
G.pfdHor = line('Parent',G.axF,'XData',nan,'YData',nan,'Color',[1 1 1],'LineWidth',1.6);
G.pfdLad = line('Parent',G.axF,'XData',nan,'YData',nan,'Color',[1 1 1],'LineWidth',0.9);
line('Parent',G.axF,'XData',cos(thf),'YData',sin(thf),'Color',[0.75 0.8 0.85],'LineWidth',2);
tkX = []; tkY = [];
for a = [-60 -45 -30 -20 -10 0 10 20 30 45 60]
    r2 = 1.07 + 0.04*any(a == [-60 -30 0 30 60]); aa = a*pi/180;
    tkX = [tkX 1.02*sin(aa) r2*sin(aa) nan]; tkY = [tkY 1.02*cos(aa) r2*cos(aa) nan]; %#ok<AGROW>
end
line('Parent',G.axF,'XData',tkX,'YData',tkY,'Color',[0.8 0.85 0.9],'LineWidth',1);
G.pfdRoll = patch('Parent',G.axF,'XData',[0 -0.05 0.05],'YData',[0.98 0.88 0.88],'FaceColor',[1 0.85 0.1],'EdgeColor','none');
line('Parent',G.axF,'XData',[-0.55 -0.2 -0.2],'YData',[0 0 -0.08],'Color',[1 0.85 0.1],'LineWidth',2.5);
line('Parent',G.axF,'XData',[0.2 0.2 0.55],'YData',[-0.08 0 0],'Color',[1 0.85 0.1],'LineWidth',2.5);
line('Parent',G.axF,'XData',0,'YData',0,'LineStyle','none','Marker','s','MarkerSize',4,'MarkerFaceColor',[1 0.85 0.1],'Color',[1 0.85 0.1]);
G.pfdTxt = text('Parent',G.axF,'Position',[-1.05 -1.18],'String','','Color',[0.7 1 0.8],'FontName','Courier New','FontSize',7.5, ...
    'VerticalAlignment','top','Interpreter','none');
text('Parent',G.axF,'Position',[-1.08 1.08],'String','ATTITUDE','Color',[0.6 0.75 0.85],'FontSize',7,'VerticalAlignment','top');
G.pfd_all = [G.axF; allchild(G.axF)];
% ---------- legend / colour conventions ----------
G.axL = axes('Parent',G.fig,'Position',[0.62 0.02 0.37 0.23],'Color',[0.03 0.04 0.06],'XColor','none','YColor','none');
xlim(G.axL,[0 1]); ylim(G.axL,[0 1]); hold(G.axL,'on');
items = {[0.2 0.9 0.3],'s','Aircraft NORMAL'; [1 0.75 0.1],'s','Aircraft WARNING conflict'; [1 0.15 0.15],'s','Aircraft CRITICAL conflict'; ...
         [1 0.2 1],'^','EMERGENCY / abnormal mode'; [0.8 0.8 0.8],'-','Solid (aircraft colour): TRUE trajectory'; ...
         [0.8 0.8 0.8],':','Dotted: EKF-estimated trajectory'; [0.2 0.85 1],'--','Cyan dashed: predicted 4D trajectory'; ...
         [1 1 1],'-.','White dash-dot: planned/replanned route'; [1 1 1],'d','Waypoint (star = destination)'; ...
         [0.9 0.15 0.15],'s','Restricted / no-fly zone'; [0.5 0.55 0.65],'-','Separation cylinder (H/2, V/2)'};
for q = 1:size(items,1)
    yq = 0.95 - (q-1)*0.085; col = items{q,1}; mk = items{q,2};
    if any(strcmp(mk,{'s','^','d'}))
        line('Parent',G.axL,'XData',0.04,'YData',yq,'LineStyle','none','Marker',mk,'MarkerFaceColor',col,'Color',col,'MarkerSize',7);
    else
        line('Parent',G.axL,'XData',[0.01 0.07],'YData',[yq yq],'LineStyle',mk,'Color',col,'LineWidth',1.5);
    end
    text('Parent',G.axL,'Position',[0.1 yq],'String',items{q,3},'Color',[0.85 0.88 0.9],'FontSize',7.5,'VerticalAlignment','middle');
end
text('Parent',G.axL,'Position',[0.62 0.95],'String',sprintf('KEYS\n1-9 focus AC\nc camera (5 modes)\nr radar  t HUD\np predictions\nm perf panels\ng attitude PFD\na animation\nspace pause\nx stop'), ...
    'Color',[0.7 0.85 1],'FontSize',7.5,'VerticalAlignment','top');
drawnow;
end

function [G, stopReq] = live_update(G, W, cfg, t)
stopReq = false;
if ~ishghandle(G.fig), stopReq = true; return; end
ctl = getappdata(G.fig,'ctl');
while ctl.pause && ishghandle(G.fig) && ~ctl.stop
    drawnow; pause(0.05); ctl = getappdata(G.fig,'ctl');
end
if ~ishghandle(G.fig) || ctl.stop, stopReq = true; return; end
ctl.focus = min(max(ctl.focus,1), G.na);
if ctl.anim
    G = update_3d_visualization(G, W, cfg, t, ctl);
    if ctl.radar, G = update_radar_display(G, W, cfg, t); end
    if ctl.hud, update_telemetry(G, W, cfg, t, ctl); end
    if ctl.perf, G = update_perf_panels(G, W, cfg, t, ctl); end
    if ctl.pfd, update_pfd(G, W, ctl); end
    if G.last.pfd ~= ctl.pfd, set(G.pfd_all,'Visible',onoff(ctl.pfd)); G.last.pfd = ctl.pfd; end
    if G.last.radar ~= ctl.radar, set(G.radar_all,'Visible',onoff(ctl.radar)); G.last.radar = ctl.radar; end
    if G.last.hud ~= ctl.hud, set(G.hud_all,'Visible',onoff(ctl.hud)); G.last.hud = ctl.hud; end
    if G.last.perf ~= ctl.perf, set([G.axP; allchild(G.axP)],'Visible',onoff(ctl.perf)); G.last.perf = ctl.perf; end
    drawnow limitrate;
else
    drawnow limitrate;
end
end

function s = onoff(c)
if c, s = 'on'; else, s = 'off'; end
end

function G = update_3d_visualization(G, W, cfg, t, ctl)
zs = G.zs; na = G.na; Pm = [0 1 0; 1 0 0; 0 0 -1]; V = cfg.viz;
G.tr_i = mod(G.tr_i, V.trail_len) + 1; ti = G.tr_i;
pulse = 0.5 + 0.5*sin(2*pi*1.5*t);          % 1.5 Hz pulse for alerts
nsg = size(G.trSeg,2); edg = round(linspace(0, V.trail_len, nsg + 1));
anyEm = false(1,na);
for i = 1:na
    ac = W.ac(i); x = ac.x;
    pos = [x(11)/1000, x(10)/1000, x(12)/1000*zs];
    Rp = Pm*dcm_b2n(x(7), x(8), x(9));                    % Rplot = P*Rz*Ry*Rx
    set(G.hgt(i), 'Matrix', hgmat(Rp, pos));
    pe = [ac.E.x(11)/1000, ac.E.x(10)/1000, ac.E.x(12)/1000*zs];
    G.tr_true(ti,:,i) = pos; G.tr_ekf(ti,:,i) = pe;
    ord = [ti+1:V.trail_len, 1:ti];
    tt = G.tr_true(ord,:,i); te = G.tr_ekf(ord,:,i);
    for q = 1:nsg
        ix = max(edg(q),1):edg(q+1);
        set(G.trSeg(i,q),'XData',tt(ix,1),'YData',tt(ix,2),'ZData',tt(ix,3));
    end
    set(G.ekfL(i),'XData',te(:,1),'YData',te(:,2),'ZData',te(:,3));
    vel = dcm_b2n(x(7),x(8),x(9))*x(1:3);
    vp = pos + V.vel_vec_sec*[vel(2) vel(1) -vel(3)*zs]/1000;
    set(G.velL(i),'XData',[pos(1) vp(1)],'YData',[pos(2) vp(2)],'ZData',[pos(3) vp(3)]);
    hp = pos + 1.2*Rp(:,1)';
    set(G.hdgL(i),'XData',[pos(1) hp(1)],'YData',[pos(2) hp(2)],'ZData',[pos(3) hp(3)]);
    if V.show_sep
        set(G.sepL(i),'XData',pos(1) + G.sepX,'YData',pos(2) + G.sepY,'ZData',pos(3) + G.sepZ,'Visible',onoff(ac.alive));
    end
    em = ~ismember(ac.mode, [1 2 3]) && ac.alive; anyEm(i) = em;
    if em, col = [1 0.2 1]; elseif ac.atc_sev == 2, col = [1 0.15 0.15]; elseif ac.atc_sev == 1, col = [1 0.75 0.1];
    else, col = [0.2 0.9 0.3]; end
    if ~ac.alive, col = [0.4 0.4 0.4]; end
    if any(G.last_col(i,:) ~= col), set(G.mesh(i),'FaceColor',col); G.last_col(i,:) = col; end
    msz = 16; if ac.alive && ac.atc_sev >= 1, msz = 16 + 8*ac.atc_sev*pulse; end
    set(G.mk(i),'XData',pos(1),'YData',pos(2),'ZData',pos(3),'Color',col,'Visible',onoff(ac.alive),'MarkerSize',msz);
    if V.nav_lights
        set(G.navP(i),'Visible',onoff(ac.alive)); set(G.navS(i),'Visible',onoff(ac.alive));
        set(G.strobe(i),'Visible',onoff(ac.alive && mod(floor(t*2.5),2) == 0));
    end
    set(G.shd(i),'XData',pos(1),'YData',pos(2),'ZData',0.004,'Visible',onoff(ac.alive));
    set(G.drop(i),'XData',[pos(1) pos(1)],'YData',[pos(2) pos(2)],'ZData',[0.004 pos(3)],'Visible',onoff(ac.alive));
    set(G.beacon(i),'XData',pos(1),'YData',pos(2),'ZData',pos(3) + 0.9,'Visible',onoff(em));
    if i == ctl.focus || ac.atc_sev > 0 || em
        lbl = sprintf('%s  %.0f m  %s', ac.name, x(12), cfg.modes{ac.mode});
    else
        lbl = sprintf('%s %.0f m', ac.name, x(12));      % compact label keeps a crowded scene readable
    end
    set(G.txt(i),'Position',pos + [0 0 0.6],'String',lbl);
    if G.plan_ver(i) ~= ac.fp.ver || G.pred_ver ~= W.atc.pred_ver
        wps = ac.fp.wps; rt = [wps(:,2)/1000, wps(:,1)/1000, wps(:,3)/1000*zs];
        set(G.wpL(i),'XData',rt(:,2-1),'YData',rt(:,2),'ZData',rt(:,3));
        set(G.destL(i),'XData',rt(end,1),'YData',rt(end,2),'ZData',rt(end,3));
        a0 = min(ac.fp.active, size(rt,1)); trk = W.atc.tr(i).pos;
        pl = [trk(2)/1000 trk(1)/1000 trk(3)/1000*zs; rt(a0:end,:)];
        set(G.planL(i),'XData',pl(:,1),'YData',pl(:,2),'ZData',pl(:,3));
        G.plan_ver(i) = ac.fp.ver;
    end
end
if G.pred_ver ~= W.atc.pred_ver
    for i = 1:na
        pr = W.atc.preds{i};
        if isempty(pr), set(G.predL(i),'XData',nan,'YData',nan,'ZData',nan); continue; end
        set(G.predL(i),'XData',pr.pos(:,2)/1000,'YData',pr.pos(:,1)/1000,'ZData',pr.pos(:,3)/1000*zs);
    end
    G.pred_ver = W.atc.pred_ver;
end
set(G.predL,'Visible',onoff(ctl.pred));
% conflicts between aircraft pairs (live positions, current ATC classification)
k = 0; C = W.atc.conf; sevCol = [0.5 0.9 0.4; 1 0.75 0.1; 1 0.1 0.1];
for i = 1:na-1
    for j = i+1:na
        k = k + 1; c = find([C.i] == i & [C.j] == j, 1);
        if ~isempty(c) && W.ac(i).alive && W.ac(j).alive
            pi_ = [W.ac(i).x(11) W.ac(i).x(10) W.ac(i).x(12)*zs]/1000; pj = [W.ac(j).x(11) W.ac(j).x(10) W.ac(j).x(12)*zs]/1000;
            cc = sevCol(C(c).sev + 1, :);
            set(G.confL(k),'XData',[pi_(1) pj(1)],'YData',[pi_(2) pj(2)],'ZData',[pi_(3) pj(3)],'Color',cc,'Visible','on');
            set(G.confM(k),'XData',C(c).pos(2)/1000,'YData',C(c).pos(1)/1000,'ZData',C(c).pos(3)/1000*zs,'Color',cc,'Visible','on');
        else
            set(G.confL(k),'Visible','off'); set(G.confM(k),'Visible','off');
        end
    end
end
for z = 1:numel(G.zone)
    if W.zone_alert(z) == 2, set(G.zone(z),'FaceColor',[1 0.6 0],'FaceAlpha',0.35);
    elseif W.zone_alert(z) == 1, set(G.zone(z),'FaceColor',[1 0.35 0.1],'FaceAlpha',0.25);
    else, set(G.zone(z),'FaceColor',[0.9 0.15 0.15],'FaceAlpha',0.18); end
end
nCrit = 0; nWarn = 0;
for c = 1:numel(C)
    if W.ac(C(c).i).alive && W.ac(C(c).j).alive
        if C(c).sev == 2, nCrit = nCrit + 1; elseif C(c).sev == 1, nWarn = nWarn + 1; end
    end
end
if V.alert_banner
    if nCrit > 0
        set(G.alert,'String',sprintf('!! CRITICAL CONFLICT x%d - ATC RESOLVING !!',nCrit),'Color',[1 0.3 0.3],'EdgeColor',[1 0.3 0.3],'Visible',onoff(pulse > 0.25));
    elseif nWarn > 0
        set(G.alert,'String',sprintf('CONFLICT WARNING x%d - ATC RESOLVING',nWarn),'Color',[1 0.78 0.1],'EdgeColor',[1 0.78 0.1],'Visible','on');
    else
        set(G.alert,'Visible','off');
    end
end
if V.radar_sweep3d && ~isempty(G.sw3) && ishghandle(G.sw3(1))
    aw = mod(t*cfg.radar.sweep_rate, 2*pi) - linspace(0, 0.7, 14)';
    set(G.sw3,'XData',[G.rc(1); G.rc(1) + 13.5*sin(aw); G.rc(1)],'YData',[G.rc(2); G.rc(2) + 13.5*cos(aw); G.rc(2)], ...
        'ZData',0.006*ones(16,1));
end
mhs = W.sep.minH; if ~isfinite(mhs), mhs = nan; end
set(G.ttl,'String',sprintf('LIVE 3D AIRSPACE  |  t = %5.1f s  |  %d aircraft  |  conflicts resolved %d  |  min H-sep %.0f m  |  LoS events %d', ...
    t, na, W.atc.stats.resolved, mhs, W.sep.los_events));
if cfg.viz.show_wind && t - G.wind_t >= 2
    for q = 1:size(G.wind_pos,1)
        wv = wind_mean_ned(G.wind_pos(q,3)/zs*1000, cfg); sc = 0.15;
        set(G.windQ,'UData',zeros(size(G.wind_pos,1),1)); 
        U(q,1) = wv(2)*sc; Vv(q,1) = wv(1)*sc; %#ok<AGROW>
    end
    set(G.windQ,'UData',U,'VData',Vv,'WData',zeros(size(U))); G.wind_t = t;
end
% ---- camera ----
fa = W.ac(ctl.focus).x; fpos = [fa(11) fa(10) fa(12)*zs]/1000;
if ctl.cam_changed && ~any(strcmp(ctl.cam, {'follow','auto'}))
    try, camtarget(G.ax,'auto'); campos(G.ax,'auto'); camva(G.ax,'auto'); catch, end %#ok<CTCH>
end
switch ctl.cam
    case 'auto'      % cinematic: orbit the most critical conflict (or the traffic centroid)
        P3 = zeros(0,3);
        for q = 1:na
            if W.ac(q).alive, P3(end+1,:) = [W.ac(q).x(11) W.ac(q).x(10) W.ac(q).x(12)*zs]/1000; end %#ok<AGROW>
        end
        tgt = [0 12 5]; if ~isempty(P3), tgt = mean(P3,1); end
        bS = 0; bT = inf;
        for c = 1:numel(C)
            if C(c).sev >= 1 && W.ac(C(c).i).alive && W.ac(C(c).j).alive && (C(c).sev > bS || (C(c).sev == bS && C(c).tloss < bT))
                pa3 = [W.ac(C(c).i).x(11) W.ac(C(c).i).x(10) W.ac(C(c).i).x(12)*zs]/1000;
                pb3 = [W.ac(C(c).j).x(11) W.ac(C(c).j).x(10) W.ac(C(c).j).x(12)*zs]/1000;
                tgt = 0.5*(pa3 + pb3); bS = C(c).sev; bT = C(c).tloss;
            end
        end
        if any(isnan(G.camT)), G.camT = tgt; else, G.camT = G.camT + 0.12*(tgt - G.camT); end
        azc = t*7*pi/180; elc = 28*pi/180; Rc = 13;
        campos(G.ax, G.camT + [Rc*cos(elc)*sin(azc) Rc*cos(elc)*cos(azc) Rc*sin(elc)]);
        camtarget(G.ax, G.camT); camva(G.ax, 38);
    case 'fixed'
        if ctl.cam_changed, view(G.ax,-38,24); camva(G.ax,'auto'); end
    case 'top'
        if ctl.cam_changed, view(G.ax,0,90); end
    case 'follow'
        hd = W.ac(ctl.focus).x(9); off = [-sin(hd) -cos(hd) 0]*(-1)*3 + [0 0 1.2];
        camtarget(G.ax, fpos); campos(G.ax, fpos - [sin(hd) cos(hd) 0]*3 + [0 0 1.2]); camva(G.ax,35); %#ok<NASGU>
    case 'orbit'
        az = mod(t*6, 360); view(G.ax, az, 24);
end
if ctl.cam_changed
    c2 = getappdata(G.fig,'ctl'); c2.cam_changed = false; setappdata(G.fig,'ctl',c2); ctl.cam_changed = false;
end
end

function G = update_radar_display(G, W, cfg, t)
rg = cfg.viz.radar_range_km; C0 = cfg.radar.pos; na = G.na; th = mod(t*cfg.radar.sweep_rate, 2*pi);
for q = 1:3
    a = th - (q-1)*0.12;
    set(G.sweep(q),'XData',[0 rg*sin(a)],'YData',[0 rg*cos(a)]);
end
sevCol = [0.2 1 0.3; 1 0.75 0.1; 1 0.15 0.15];
for i = 1:na
    tr = W.atc.tr(i); ac = W.ac(i);
    e = (tr.pos(2) - C0(2))/1000; n = (tr.pos(1) - C0(1))/1000;
    inr = ac.alive && hypot(e,n) <= rg;
    G.rtr(:,:,i) = [G.rtr(2:end,:,i); e n];
    if ~ismember(ac.mode,[1 2 3]), col = [1 0.2 1]; else, col = sevCol(ac.atc_sev + 1, :); end
    hd = atan2(tr.vel(2), tr.vel(1)); L = 3;
    set(G.rBlip(i),'XData',e,'YData',n,'Color',col,'MarkerFaceColor',col,'Visible',onoff(inr));
    set(G.rHdg(i),'XData',[e e + L*sin(hd)],'YData',[n n + L*cos(hd)],'Color',col,'Visible',onoff(inr));
    aiTag = '';
    if isfield(W.atc, 'ml_conf') && ~isempty(W.atc.ml_conf)
        for q = 1:numel(W.atc.ml_conf)
            if (W.atc.ml_conf(q).i == i || W.atc.ml_conf(q).j == i) && W.atc.ml_conf(q).p_conflict >= 0.50
                aiTag = sprintf(' AI:%0.0f%%', W.atc.ml_conf(q).p_conflict*100);
                break;
            end
        end
    end
    set(G.rTxt(i),'Position',[e + 0.6 n + 0.6],'Color',col,'Visible',onoff(inr), ...
        'String',sprintf('%s FL%03.0f %s%s', ac.name, tr.pos(3)*3.28084/100, sev_label(ac.atc_sev), aiTag));
    set(G.rTrail(i),'XData',G.rtr(:,1,i),'YData',G.rtr(:,2,i));
    pr = W.atc.preds{i};
    if ~isempty(pr)
        set(G.rPred(i),'XData',(pr.pos(:,2) - C0(2))/1000,'YData',(pr.pos(:,1) - C0(1))/1000,'Visible',onoff(inr));
    else, set(G.rPred(i),'XData',nan,'YData',nan); end
end
end

function update_telemetry(G, W, cfg, t, ctl)
ac = W.ac(ctl.focus); x = ac.x; Rn = dcm_b2n(x(7),x(8),x(9)); v = Rn*x(1:3); wb = Rn'*ac.wn;
Va = norm(x(1:3) - wb); gs = hypot(v(1), v(2)); vs = -v(3);
err = ac.E.x - x; perr = hypot(err(10), err(11));
sev = {'NORMAL','WARNING','CRITICAL'}; mh = W.sep.minH; if ~isfinite(mh), mh = nan; end
wm = wind_mean_ned(x(12), cfg);
flag = ''; if ac.f.gps_out, flag = [flag ' GPS-OUT']; end
if ac.f.comm_out, flag = [flag ' COMM-OUT']; end
if mean(ac.f.eh) < 0.85, flag = [flag ' ENG-DEG']; end
if isempty(flag), flag = ' none'; end
s = sprintf(['TELEMETRY  t=%6.1f s\nAC %s [id %d]\nMODE %s\nN %9.0f m  E %9.0f m\nALT %6.0f m  VS %+5.1f m/s\n' ...
    'Va %5.1f  GS %5.1f m/s\nHDG %5.1f  ROLL %+6.1f\nPITCH %+5.1f YAW %5.1f deg\nTHR %4.2f  ELEV %+6.3f\nAIL %+6.3f  RUD %+6.3f rad\n' ...
    'EKF pos err %6.1f m\nMIN SEP(all) %6.0f m\nCONFLICT %s\nEMERGENCY %s\nFAIL:%s\nWIND %4.1f m/s from %03.0f deg'], ...
    t, ac.name, ac.id, cfg.modes{ac.mode}, x(10), x(11), x(12), vs, Va, gs, mod(atan2(v(2),v(1))*180/pi,360), ...
    x(7)*180/pi, x(8)*180/pi, mod(x(9)*180/pi,360), ac.act(1), ac.act(2), ac.act(3), ac.act(4), perr, mh, ...
    sev{ac.atc_sev + 1}, tern(ismember(ac.mode,[1 2 3]), 'none', cfg.modes{ac.mode}), flag, hypot(wm(1),wm(2)), ...
    mod(atan2(-wm(2),-wm(1))*180/pi,360));
set(G.hud,'String',s);
end

function update_pfd(G, W, ctl)
% Artificial horizon for the focus aircraft: sky/ground split, pitch ladder, roll pointer, readouts.
ac = W.ac(ctl.focus); x = ac.x; r = x(7); pit = x(8); psi = x(9);
Rn = dcm_b2n(x(7), x(8), x(9)); v = Rn*x(1:3); wb = Rn'*ac.wn; Va = norm(x(1:3) - wb); vs = -v(3);
d = -pit*180/pi/35;                          % horizon offset on screen (35 deg per unit radius)
pp = d*cos(r); cr = cos(r); sr = sin(r);     % horizon distance along the horizon normal
if pp >= 1
    th = linspace(0, 2*pi, 73); gu = cos(th); gv = sin(th);
elseif pp <= -1
    gu = nan; gv = nan;
else
    ph = linspace(pi - asin(pp), 2*pi + asin(pp), 41); gu = cos(ph); gv = sin(ph);
end
set(G.pfdGnd, 'XData', gu*cr - gv*sr, 'YData', gu*sr + gv*cr);
if abs(pp) < 1
    w = sqrt(1 - pp^2); hu = [-w w]; hv = [pp pp];
    set(G.pfdHor, 'XData', hu*cr - hv*sr, 'YData', hu*sr + hv*cr);
else
    set(G.pfdHor, 'XData', nan, 'YData', nan);
end
lu = []; lv = [];
for a = [-20 -10 10 20]
    vv = pp + a/35; hw = 0.12 + 0.10*(abs(a) == 20);
    if hypot(hw, vv) < 0.92, lu = [lu -hw hw nan]; lv = [lv vv vv nan]; end %#ok<AGROW>
end
if isempty(lu), lu = nan; lv = nan; end
set(G.pfdLad, 'XData', lu*cr - lv*sr, 'YData', lu*sr + lv*cr);
tri = [0 0.98; -0.05 0.88; 0.05 0.88];
set(G.pfdRoll, 'XData', tri(:,1)*cr - tri(:,2)*sr, 'YData', tri(:,1)*sr + tri(:,2)*cr);
set(G.pfdTxt, 'String', sprintf('%s  HDG %03.0f  BNK%+4.0f\nIAS %4.0f m/s  ALT %5.0f m\nVS %+5.1f m/s  PIT%+4.0f', ...
    ac.name, mod(psi*180/pi, 360), r*180/pi, Va, x(12), vs, pit*180/pi));
end

function G = update_perf_panels(G, W, ~, t, ctl)
na = G.na; ac = W.ac(ctl.focus); st = W.atc.stats;
mpcN = sum(arrayfun(@(a) a.mpc.n, W.ac)); mpcF = sum(arrayfun(@(a) a.mpc.nfb, W.ac));
sat = 100*ac.sat_cnt/max(ac.n_steps,1); rej = ac.E.rej;
s = sprintf(['PERFORMANCE / DIAGNOSTICS\nConflicts det %d (W%d C%d)\nResolved %d  Unresolved %d\nManeuvers %d  Recov %d\n' ...
    'Zone replans %d  viol %d\nLoS events %d  safety viol %d\nMinSep H %5.0f V %5.0f m\n--- %s ---\n' ...
    'EKF rej g%d b%d m%d a%d i%d c%d\nEKF resets %d  P(pos) %5.1f\nMPC solver %s %.1f ms\nMPC fallbacks %d/%d\n' ...
    'MPC note: %s\nSat%% thr%4.1f el%4.1f ail%4.1f rud%4.1f'], ...
    st.detected, st.warnings, st.criticals, st.resolved, st.unresolved + sum(sum(W.atc.act)), st.maneuvers, st.recoveries, ...
    st.zone_replans, W.viol.zone, W.sep.los_events, W.viol.los + W.viol.zone + W.viol.bank + W.viol.alpha + W.viol.speed + W.viol.ground, ...
    W.sep.minH, W.sep.minV, ac.name, rej(1), rej(2), rej(3), rej(4), rej(5), rej(6), ac.E.resets, sqrt(max(ac.E.P(10,10),0)), ...
    ac.mpc.solver, 1e3*ac.mpc.last_t, mpcF, mpcN, ac.mpc.last_reason, sat(1), sat(2), sat(3), sat(4));
ai_line = '';
if isfield(W.atc, 'ml_conf') && ~isempty(W.atc.ml_conf)
    p_max = 0; best_q = 0;
    for q = 1:numel(W.atc.ml_conf)
        if W.atc.ml_conf(q).p_conflict > p_max
            p_max = W.atc.ml_conf(q).p_conflict;
            best_q = q;
        end
    end
    if best_q > 0 && p_max >= 0.25
        ai = W.atc.ml_conf(best_q);
        ai_line = sprintf('\nAI DUAL-MON: P=%.2f [%s] TCPA=%.0fs\nAI Sep: %.0fm | Disagree: %s', ...
            ai.p_conflict, ai.warning_name, ai.tcpa, ai.dcpa_h, tern(ai.disagreement, 'YES', 'NO'));
    end
end
s = [s ai_line];
set(G.perf,'String',s);
% time series
n = G.ts.n + 1;
if n <= numel(G.ts.t)
    G.ts.n = n; G.ts.t(n) = t; mh = W.L.minH(max(W.nlog,1));
    if ~isfinite(mh), mh = nan; end
    G.ts.minH(n) = mh/1000;
    err = ac.E.x - ac.x; G.ts.ekf(n) = hypot(err(10), err(11));
    G.ts.trk(n) = hypot(ac.x(10) - ac.refcur(1), ac.x(11) - ac.refcur(2));
    set(G.lSep,'XData',G.ts.t(1:n),'YData',G.ts.minH(1:n));
    set(G.lEkf,'XData',G.ts.t(1:n),'YData',G.ts.ekf(1:n)); set(G.lTrk,'XData',G.ts.t(1:n),'YData',G.ts.trk(1:n));
end
end

function s = sev_label(sv)
nm = {'','WARN','CRIT'}; s = nm{min(max(round(sv),0),2) + 1};
end

%% ######################################################################
%  PART 2 - 3-D AIRCRAFT RADAR COMMAND CENTER (simulated demo traffic)
%% ######################################################################
function AircraftRadar3D()
% AircraftRadar3D - 3D interactive aircraft radar command-center dashboard
% ALL AIRCRAFT DATA IS SIMULATED / DEMO DATA (not real air traffic).
%
% Requires MATLAB R2021a or newer (uifigure, uiaxes 3-D, uitable styles).
% Run:   >> AircraftRadar3D
%
% Mouse (3-D view): left-drag = orbit, scroll = zoom, middle/right-drag = pan,
%                   click an aircraft = select + zoom/follow, hover = info card.
% Also select from the left list or the bottom table (all views stay in sync).

%% ===================== configuration =====================
R     = 100;                 % radar range [NM]
LAT0  = 40.0;  LON0 = -74.0; % DEMO radar origin (simulated)
ZS    = 1/1000;              % altitude ft -> scene units (1 unit = 1000 ft)
ZMAX  = 50;                  % top of the 3-D volume [scene units = 50,000 ft]
WIND  = [12 -4];             % demo wind [east north] knots
DEG   = char(176);

C = struct('bg',[0.015 0.03 0.06],'panel',[0.04 0.08 0.13],'panel2',[0.07 0.12 0.19], ...
    'cyan',[0.10 0.85 1.0],'green',[0.15 0.95 0.55],'amber',[1 0.72 0.15], ...
    'red',[1 0.28 0.28],'txt',[0.86 0.95 1.0],'dim',[0.45 0.62 0.72]);

% id, category, model, callsign, alt(ft), TAS(kt), heading, x(NM), y(NM)
def = { ...
 'AC-001','Passenger', 'Boeing 737-800',       'SKY101',34000,460, 75, -60, -35;
 'AC-002','Cargo',     'Boeing 747-8F',        'CGO202',36000,480,200,  40,  55;
 'AC-003','Fighter',   'F-16 Falcon',          'VIPER3',22000,560,290,  50, -20;
 'AC-004','Business',  'Gulfstream G650',      'JET404',41000,490,140, -30,  60;
 'AC-005','Helicopter','AgustaWestland AW139', 'MED505', 3500,140,330,  15, -25;
 'AC-006','Passenger', 'Airbus A320',          'SKY606',28000,440, 15,  20, -70;
 'AC-007','Cargo',     'Boeing 777F',          'CGO707',33000,470,255,  70,  15;
 'AC-008','Business',  'Cessna Citation X',    'JET808',25000,420,100, -70,  30};

N      = size(def,1);
ids    = def(:,1);  cats = def(:,2);  models = def(:,3);  calls = def(:,4);
ALT    = cell2mat(def(:,5));  TAS = cell2mat(def(:,6));  HDG = cell2mat(def(:,7));
X      = cell2mat(def(:,8));  Y   = cell2mat(def(:,9));
VS     = zeros(N,1);  TR = zeros(N,1);  GS = zeros(N,1);  CRS = zeros(N,1);
tgtHdg = HDG;  tgtAlt = ALT;  tgtTas = TAS;  nextChg = 60 + rand(N,1)*120;
PF = zeros(N,6);  msc = ones(N,1);  baseCol = zeros(N,3);  catCode = cell(N,1);
for i = 1:N
    [PF(i,:), msc(i), baseCol(i,:), catCode{i}] = categoryInfo(cats{i});
end

%% ===================== runtime state =====================
simT = 0;  paused = false;  sel = 0;  hoverIdx = 0;
followOn = false;  zoomGoal = 0;  sweepAng = 0;  M = 32;   % sweep wedge resolution
lastHit = -inf(N,1);  conflict = false(N,1);  status = repmat({'CRUISE'},N,1);
H = repmat({zeros(0,7)},N,1);   % history rows: [simT alt gs vs dist x y]
lastRec = 0;  lastSlow = -1;  tPrev = 0;  downPt = [];  teleIdx = 1;
viewIdx = (1:N)';  searchStr = '';  stateL = 'auto';  stateR = 'auto';  closing = false;
hStatic = gobjects(0);
teleCols   = [2 3 4 5];
teleLabels = {'Altitude (ft)','Ground speed (kt)','Vertical speed (fpm)','Distance from radar (NM)'};
teleNames  = {'Altitude vs Time','Speed vs Time','Vertical Speed vs Time','Distance vs Time'};

%% ===================== UI =====================
fig = uifigure('Name','3D Aircraft Radar Command Center - SIMULATED / DEMO DATA', ...
    'Color',C.bg,'Position',[30 30 1520 900]);
g = uigridlayout(fig,[3 3],'RowHeight',{46,'1x',235},'ColumnWidth',{275,'1x',330}, ...
    'Padding',[6 6 6 6],'RowSpacing',6,'ColumnSpacing',6,'BackgroundColor',C.bg);

% ---- top bar
pTop = mkPanel(g,1,[1 3],'');
gt = uigridlayout(pTop,[1 9],'ColumnWidth',{'1x','1x','1x','1x','1x','1.3x','1.6x',64,64}, ...
    'Padding',[8 4 8 4],'BackgroundColor',C.panel);
lblStat  = mkLbl(gt,'RADAR ONLINE',C.green,12);
lblRange = mkLbl(gt,'',C.txt,12);
lblDet   = mkLbl(gt,'',C.txt,12);
lblTrk   = mkLbl(gt,'',C.txt,12);
lblTime  = mkLbl(gt,'',C.txt,12);
lblSys   = mkLbl(gt,'SYSTEM NOMINAL',C.green,12);
lblDemo  = mkLbl(gt,'SIMULATED / DEMO DATA',C.amber,12);
uibutton(gt,'Text','Tracks','ButtonPushedFcn',@(~,~)togglePanel('L'));
uibutton(gt,'Text','Info','ButtonPushedFcn',@(~,~)togglePanel('R'));

% ---- left panel: list + controls
pL = mkPanel(g,2,1,'AIRCRAFT TRACKING LIST');
gl = uigridlayout(pL,[2 1],'RowHeight',{'1x',300},'Padding',[4 4 4 4],'BackgroundColor',C.panel);
lst = uilistbox(gl,'Items',ids','ItemsData',1:N,'FontName','Consolas','FontSize',10, ...
    'BackgroundColor',C.panel2,'FontColor',C.txt,'ValueChangedFcn',@(s,~)selectAircraft(s.Value,true));
pC = uipanel(gl,'Title','RADAR CONTROLS','BackgroundColor',C.panel,'ForegroundColor',C.cyan);
gc = uigridlayout(pC,[9 2],'RowHeight',repmat({28},1,9),'ColumnWidth',{'1x','1.3x'}, ...
    'Padding',[4 2 4 2],'RowSpacing',2,'BackgroundColor',C.panel);
sldRange = mkSlider(gc,1,'Radar range (NM)',[50 200],R);
sldRange.ValueChangedFcn = @onRange;
sldSweep = mkSlider(gc,2,'Sweep speed (RPM)',[1 30],10);
sldSim   = mkSlider(gc,3,'Simulation speed (x)',[1 100],15);
sldHist  = mkSlider(gc,4,'Track history (pts)',[50 1000],400);
chkAlt   = mkCheck(gc,5,1,'Altitude lines',true);
chkPath  = mkCheck(gc,5,2,'Flight paths',true);
chkLbl   = mkCheck(gc,6,1,'Aircraft labels',true);
chkSweep = mkCheck(gc,6,2,'Radar sweep',true);
chkFollow= mkCheck(gc,7,1,'Follow selected',false);
chkFollow.ValueChangedFcn = @(s,~)setFollow(s.Value);
btnPause = uibutton(gc,'Text','Pause sim','ButtonPushedFcn',@onPause);
btnPause.Layout.Row = 7;  btnPause.Layout.Column = 2;
b1 = uibutton(gc,'Text','Reset camera','ButtonPushedFcn',@(~,~)resetCamera());
b1.Layout.Row = 8;  b1.Layout.Column = 1;
b2 = uibutton(gc,'Text','Fullscreen','ButtonPushedFcn',@toggleFS);
b2.Layout.Row = 8;  b2.Layout.Column = 2;
lNote = uilabel(gc,'Text','Z axis = altitude (1 unit = 1000 ft)','FontColor',C.dim,'FontSize',10);
lNote.Layout.Row = 9;  lNote.Layout.Column = [1 2];

% ---- centre: 3-D radar
ax = uiaxes(g);  ax.Layout.Row = 2;  ax.Layout.Column = 2;

% ---- right panel: aircraft info
pR = mkPanel(g,2,3,'SELECTED AIRCRAFT');
gr = uigridlayout(pR,[1 1],'Padding',[2 2 2 2],'BackgroundColor',C.panel);
infoTbl = uitable(gr,'ColumnName',{'PARAMETER','VALUE'},'RowName',{},'FontName','Consolas', ...
    'FontSize',11,'ColumnWidth',{125,'auto'},'BackgroundColor',[C.panel2;C.panel], ...
    'ForegroundColor',C.txt,'Data',{'-','Select an aircraft'});

% ---- bottom: data table (+ search / sort)
pB = mkPanel(g,3,[1 2],'LIVE TRACK TABLE  (click a row to select)');
gb = uigridlayout(pB,[2 3],'RowHeight',{28,'1x'},'ColumnWidth',{'1x',170,170}, ...
    'Padding',[4 2 4 2],'RowSpacing',4,'BackgroundColor',C.panel);
edSearch = uieditfield(gb,'text','Placeholder','Search ID / type / callsign / status...', ...
    'ValueChangingFcn',@(~,e)setSearch(e.Value));
ddSort = uidropdown(gb,'Items',{'Sort: ID','Sort: Altitude','Sort: Speed','Sort: Distance'}, ...
    'ValueChangedFcn',@(~,~)forceSlow());
lblSim2 = uilabel(gb,'Text','SIMULATED DATA','FontColor',C.amber,'FontWeight','bold', ...
    'HorizontalAlignment','center');
tbl = uitable(gb,'ColumnName',{'ID','Aircraft','Callsign','Altitude (ft)','Speed (kt)', ...
    'Heading','V/S (fpm)','Distance (NM)','Status'},'RowName',{},'FontName','Consolas', ...
    'FontSize',11,'ColumnWidth',{70,170,80,100,90,80,90,110,'auto'}, ...
    'BackgroundColor',[C.panel2;C.panel],'ForegroundColor',C.txt, ...
    'CellSelectionCallback',@onTblSel);
tbl.Layout.Row = 2;  tbl.Layout.Column = [1 3];
styleSel = uistyle('BackgroundColor',[0.05 0.38 0.50],'FontColor',[1 1 1]);
styleAlert = uistyle('FontColor',C.red);

% ---- bottom right: telemetry
pT = mkPanel(g,3,3,'TELEMETRY');
gtel = uigridlayout(pT,[2 1],'RowHeight',{26,'1x'},'Padding',[4 2 4 2],'BackgroundColor',C.panel);
ddTele = uidropdown(gtel,'Items',teleNames,'ValueChangedFcn',@onTele);
ax2 = uiaxes(gtel);
ax2.BackgroundColor = C.panel;  ax2.Color = C.panel2;
ax2.XColor = C.dim;  ax2.YColor = C.dim;  ax2.FontSize = 10;
ax2.GridColor = C.cyan;  ax2.GridAlpha = 0.15;  grid(ax2,'on');  hold(ax2,'on');
ax2.Title.Color = C.txt;  ax2.Title.String = 'Select an aircraft';
xlabel(ax2,'Time (sim min, relative to now)');  ax2.XLabel.Color = C.dim;
ylabel(ax2,teleLabels{1});  ax2.YLabel.Color = C.dim;
try, ax2.Toolbar.Visible = 'off'; catch, end
hTele    = plot(ax2,NaN,NaN,'-','Color',C.cyan,'LineWidth',1.8);
hTeleDot = plot(ax2,NaN,NaN,'o','MarkerFaceColor',C.green,'MarkerEdgeColor','none','MarkerSize',7);

%% ===================== 3-D scene =====================
ax.BackgroundColor = C.bg;  ax.Color = C.bg;
ax.XColor = C.dim;  ax.YColor = C.dim;  ax.ZColor = C.dim;  ax.FontSize = 9;
ax.XGrid = 'on';  ax.YGrid = 'on';  ax.ZGrid = 'on';
ax.GridColor = C.cyan;  ax.GridAlpha = 0.10;  ax.Box = 'off';
hold(ax,'on');  daspect(ax,[1 1 1]);  ax.Projection = 'perspective';
xlabel(ax,'East (NM)');  ylabel(ax,'North (NM)');  zlabel(ax,'Altitude');
title(ax,'SIMULATED / DEMO DATA - NOT REAL AIR TRAFFIC','Color',C.amber,'FontSize',11);
try, ax.Toolbar.Visible = 'off'; catch, end
try, ax.Interactions = [rotateInteraction zoomInteraction panInteraction]; catch, end
light(ax,'Style','infinite','Position',[-1 -1 1.2],'Color',[1 1 1]);
light(ax,'Style','infinite','Position',[1 1 0.5],'Color',[0.25 0.45 0.65]);

buildScene();

% sweep wedge (alpha fades behind the leading edge)
kk  = (0:M-1)';
aOut = 0.5*(1-kk/(M-1)).^2;
aVtx = zeros(2*M,1);  aVtx(1:2:end) = 0.08*aOut;  aVtx(2:2:end) = aOut;
Fs = zeros(M-1,4);
for k = 1:M-1, Fs(k,:) = [2*k-1 2*k 2*k+2 2*k+1]; end
hSweep = patch(ax,'Vertices',zeros(2*M,3),'Faces',Fs,'FaceColor',C.green,'EdgeColor','none', ...
    'FaceAlpha','interp','FaceVertexAlphaData',aVtx,'AlphaDataMapping','none','FaceLighting','none');
hSweepLn = plot3(ax,[0 0],[0 0],[0.08 0.08],'-','Color',C.green,'LineWidth',2);

% overlays
hDrop  = plot3(ax,NaN,NaN,NaN,'-','Color',C.cyan*0.55,'LineWidth',0.6);
hGnd   = plot3(ax,NaN,NaN,NaN,'+','Color',C.cyan*0.7,'MarkerSize',5);
hEcho  = scatter3(ax,X,Y,ALT*ZS,60,repmat(C.green,N,1),'filled','MarkerEdgeColor','none');
hTrail = gobjects(N,1);
for i = 1:N
    hTrail(i) = plot3(ax,NaN,NaN,NaN,'-','Color',[0.05 0.40 0.50],'LineWidth',0.8);
end
hHMark = plot3(ax,NaN,NaN,NaN,'.','Color',C.cyan,'MarkerSize',10);
hStem  = plot3(ax,NaN,NaN,NaN,':','Color',C.cyan*0.7,'LineWidth',0.8);
hProj  = plot3(ax,NaN,NaN,NaN,'--','Color',C.amber,'LineWidth',2);
hPMark = plot3(ax,NaN,NaN,NaN,'o','Color',C.amber,'MarkerSize',4,'MarkerFaceColor',C.amber);
hArrow = quiver3(ax,0,0,0,0,0,0,'AutoScale','off','Color',C.amber,'LineWidth',1.5,'MaxHeadSize',2);
hRing1 = plot3(ax,NaN,NaN,NaN,'-','Color',C.cyan,'LineWidth',2);
hRing2 = plot3(ax,NaN,NaN,NaN,'-','Color',C.cyan,'LineWidth',1.2);

% aircraft meshes
mdl = struct('V',cell(N,1),'F',cell(N,1));
hAC = gobjects(N,1);  hLbl = gobjects(N,1);
for i = 1:N
    Mi = buildModel(cats{i});
    mdl(i).V = Mi.V;  mdl(i).F = Mi.F;
    hAC(i) = patch(ax,'Vertices',Mi.V,'Faces',Mi.F,'FaceColor',baseCol(i,:),'EdgeColor','none', ...
        'FaceLighting','gouraud','BackFaceLighting','lit','AmbientStrength',0.45, ...
        'DiffuseStrength',0.8,'SpecularStrength',0.7,'SpecularExponent',25, ...
        'ButtonDownFcn',@(~,~)selectAircraft(i,true));
    hLbl(i) = text(ax,0,0,0,ids{i},'Color',C.green,'FontName','Consolas','FontSize',11,'FontWeight','bold', ...
        'HorizontalAlignment','center','Interpreter','none','Clipping','off');
end
hHover = text(ax,0,0,0,'','Color',C.txt,'BackgroundColor',[0.03 0.08 0.14],'EdgeColor',C.cyan, ...
    'FontName','Consolas','FontSize',10,'Visible','off','VerticalAlignment','bottom', ...
    'Margin',5,'Interpreter','none','Clipping','off');

% ordering: static scene drawn first
reorderChildren(ax,hStatic);

%% ===================== callbacks + start =====================
fig.WindowButtonMotionFcn = @onMotion;
fig.WindowButtonDownFcn   = @onDown;
fig.WindowButtonUpFcn     = @onUp;
fig.SizeChangedFcn        = @(~,~)applyLayout();
fig.CloseRequestFcn       = @(~,~)closeApp();

stepSim(0);  recordHistory();
resetCamera();
selectAircraft(1,false);
applyLayout();
t0 = tic;
radarTimer = timer('ExecutionMode','fixedSpacing','Period',0.04,'BusyMode','drop', ...
    'TimerFcn',@(~,~)tick());
start(radarTimer);

%% ===================== nested functions =====================
    function p = mkPanel(parent,r,c,ttl)
        p = uipanel(parent,'Title',ttl,'BackgroundColor',C.panel,'ForegroundColor',C.cyan);
        try, p.BorderColor = C.dim*0.6; catch, end
        p.Layout.Row = r;  p.Layout.Column = c;
    end

    function l = mkLbl(parent,txt,col,fs)
        l = uilabel(parent,'Text',txt,'FontColor',col,'FontName','Consolas','FontSize',fs, ...
            'FontWeight','bold','HorizontalAlignment','center');
    end

    function s = mkSlider(parent,row,txt,lim,val)
        l = uilabel(parent,'Text',txt,'FontColor',C.txt,'FontSize',11);
        l.Layout.Row = row;  l.Layout.Column = 1;
        s = uislider(parent,'Limits',lim,'Value',val,'MajorTicks',[],'MinorTicks',[]);
        s.Layout.Row = row;  s.Layout.Column = 2;
    end

    function c = mkCheck(parent,row,col,txt,val)
        c = uicheckbox(parent,'Text',txt,'Value',val,'FontColor',C.txt);
        c.Layout.Row = row;  c.Layout.Column = col;
    end

    function buildScene()
        delete(hStatic(isgraphics(hStatic)));  hStatic = gobjects(0);
        ax.XLim = [-R R];  ax.YLim = [-R R];  ax.ZLim = [0 ZMAX];
        tk = -R:R/4:R;  ax.XTick = tk;  ax.YTick = tk;  ax.ZTick = 0:10:ZMAX;
        ax.ZTickLabel = arrayfun(@(z)sprintf('FL%03d',z*10),ax.ZTick,'UniformOutput',false);
        th = linspace(0,2*pi,181);
        % radar surface
        hStatic(end+1) = patch(ax,'XData',R*cos(th),'YData',R*sin(th),'ZData',zeros(size(th)), ...
            'FaceColor',[0.01 0.07 0.10],'FaceAlpha',0.92,'EdgeColor',C.cyan,'LineWidth',1.5, ...
            'FaceLighting','none');
        % range rings + labels
        for k = 1:5
            r = k*R/5;
            hStatic(end+1) = plot3(ax,r*cos(th),r*sin(th),0.02*ones(size(th)),'-', ...
                'Color',C.cyan*0.55,'LineWidth',0.8); %#ok<AGROW>
            hStatic(end+1) = text(ax,r*0.7071,r*0.7071,0.3,sprintf('%d NM',round(r)), ...
                'Color',C.dim,'FontSize',9,'FontName','Consolas','Clipping','off'); %#ok<AGROW>
        end
        % spokes every 45 degrees
        a = (0:7)*45;
        sx = [zeros(1,8); R*sind(a); nan(1,8)];  sy = [zeros(1,8); R*cosd(a); nan(1,8)];
        hStatic(end+1) = plot3(ax,sx(:),sy(:),0.02*ones(24,1),'-','Color',C.cyan*0.4,'LineWidth',0.6);
        % compass
        names = {'N','NE','E','SE','S','SW','W','NW'};
        for k = 1:8
            hStatic(end+1) = text(ax,1.1*R*sind(a(k)),1.1*R*cosd(a(k)),0,names{k}, ...
                'Color',C.cyan,'FontWeight','bold','FontSize',12,'HorizontalAlignment','center', ...
                'Clipping','off'); %#ok<AGROW>
        end
        % altitude shells + vertical posts
        for z = 10:10:ZMAX
            hStatic(end+1) = plot3(ax,R*cos(th),R*sin(th),z*ones(size(th)),'-', ...
                'Color',C.cyan*0.30,'LineWidth',0.5); %#ok<AGROW>
        end
        px = [R 0 -R 0]; py = [0 R 0 -R];
        for k = 1:4
            hStatic(end+1) = plot3(ax,[px(k) px(k)],[py(k) py(k)],[0 ZMAX],'-', ...
                'Color',C.cyan*0.35,'LineWidth',0.6); %#ok<AGROW>
        end
        % radar tower
        hStatic(end+1) = plot3(ax,[0 0],[0 0],[0 3],'-','Color',C.green,'LineWidth',3);
        hStatic(end+1) = plot3(ax,0,0,3,'o','MarkerFaceColor',C.green,'MarkerEdgeColor','none','MarkerSize',7);
        if exist('hSweep','var') && isgraphics(hSweep)   % re-order after a range change
            reorderChildren(ax,hStatic);
        end
    end

    function resetCamera()
        followOn = false;  zoomGoal = 0;  chkFollow.Value = false;
        ax.CameraTarget = [0 0 ZMAX*0.3];
        ax.CameraPosition = [-1.25*R -1.35*R 0.95*R];
        ax.CameraUpVector = [0 0 1];
        ax.CameraViewAngle = 28;
    end

    function onRange(src,~)
        R = round(src.Value);
        buildScene();  resetCamera();  forceSlow();
    end

    function setFollow(v)
        followOn = v;  if v && sel > 0, zoomGoal = 0; end
    end

    function onPause(~,~)
        paused = ~paused;
        if paused, btnPause.Text = 'Resume sim'; else, btnPause.Text = 'Pause sim'; end
    end

    function toggleFS(~,~)
        try
            if strcmp(fig.WindowState,'fullscreen'), fig.WindowState = 'normal';
            else, fig.WindowState = 'fullscreen'; end
        catch
        end
    end

    function togglePanel(which)
        if which == 'L'
            if g.ColumnWidth{1} ~= 0, stateL = 'off'; else, stateL = 'on'; end
        else
            if g.ColumnWidth{3} ~= 0, stateR = 'off'; else, stateR = 'on'; end
        end
        applyLayout();
    end

    function applyLayout()
        if ~isvalid(fig), return; end
        w = fig.Position(3);
        showL = strcmp(stateL,'on') || (strcmp(stateL,'auto') && w >= 1050);
        showR = strcmp(stateR,'on') || (strcmp(stateR,'auto') && w >= 760);
        wl = 275*showL;  wr = 330*showR;
        if w < 1050 && showL && showR, wl = 220; wr = 240; end
        g.ColumnWidth = {wl,'1x',wr};
        if w < 760, g.RowHeight = {46,'1x',180}; else, g.RowHeight = {46,'1x',235}; end
        pL.Visible = showL;  pR.Visible = showR;  pT.Visible = showR;
    end

    function setSearch(v)
        searchStr = v;  forceSlow();
    end

    function forceSlow()
        lastSlow = -1;
    end

    function onTele(src,~)
        teleIdx = find(strcmp(src.Items,src.Value),1);
        ylabel(ax2,teleLabels{teleIdx});
        updateTelemetry();
    end

    function onTblSel(~,e)
        if isempty(e.Indices), return; end
        r = e.Indices(1);
        if r <= numel(viewIdx), selectAircraft(viewIdx(r),true); end
    end

    function selectAircraft(i,zoom)
        sel = i;
        if zoom
            followOn = true;  chkFollow.Value = true;
            zoomGoal = 26*msc(i)*R/100;
        end
        forceSlow();
    end

    function closeApp()
        closing = true;
        try, stop(radarTimer); delete(radarTimer); catch, end
        delete(fig);
    end

    % ---------- mouse picking (ray vs. aircraft) ----------
    function tf = insideAx()
        tf = false;
        try
            p = getpixelposition(ax,true);  fp = fig.CurrentPoint;
            tf = fp(1) >= p(1) && fp(1) <= p(1)+p(3) && fp(2) >= p(2) && fp(2) <= p(2)+p(4);
        catch
        end
    end

    function idx = pickAircraft()
        idx = 0;
        try, cp = ax.CurrentPoint; catch, return; end
        p0 = cp(1,:);  d = cp(2,:)-p0;  d = d/norm(d);  best = inf;  cam = ax.CameraPosition;
        for i = 1:N
            P = [X(i) Y(i) ALT(i)*ZS];  v = P-p0;  tt = dot(v,d);  dd = norm(v-tt*d);
            thr = 0.022*norm(P-cam) + 2*msc(i)*R/100;
            if dd < thr && dd < best, best = dd; idx = i; end
        end
    end

    function onMotion(~,~)
        if insideAx(), hoverIdx = pickAircraft(); else, hoverIdx = 0; end
    end

    function onDown(~,~)
        downPt = fig.CurrentPoint;
    end

    function onUp(~,~)
        if ~isempty(downPt) && norm(fig.CurrentPoint-downPt) < 4 && insideAx()
            i = pickAircraft();
            if i > 0, selectAircraft(i,true); end
        end
        downPt = [];
    end

    % ---------- simulation ----------
    function stepSim(dt)
        simT = simT + dt;
        for i = 1:N
            if simT >= nextChg(i)
                tgtHdg(i) = mod(HDG(i)+(rand-0.5)*140,360);
                tgtAlt(i) = min(PF(i,4),max(PF(i,3),ALT(i)+(rand-0.5)*12000));
                tgtTas(i) = PF(i,1)+rand*(PF(i,2)-PF(i,1));
                nextChg(i) = simT+90+rand*240;
            end
            if hypot(X(i),Y(i)) > 0.92*R && dot([X(i) Y(i)],[sind(HDG(i)) cosd(HDG(i))]) > 0
                tgtHdg(i) = mod(atan2d(-X(i),-Y(i))+(rand-0.5)*50,360);
                nextChg(i) = simT+120;
            end
            e  = mod(tgtHdg(i)-HDG(i)+180,360)-180;
            tr = max(-PF(i,5),min(PF(i,5),0.4*e));
            TR(i) = tr;  HDG(i) = mod(HDG(i)+tr*dt,360);
            TAS(i) = TAS(i)+max(-3,min(3,0.05*(tgtTas(i)-TAS(i))))*dt;
            vcmd = max(-PF(i,6),min(PF(i,6),0.5*(tgtAlt(i)-ALT(i))));
            VS(i) = VS(i)+(vcmd-VS(i))*min(1,0.3*dt);
            ALT(i) = min(PF(i,4),max(PF(i,3),ALT(i)+VS(i)/60*dt));
            v = TAS(i)/3600*[sind(HDG(i)) cosd(HDG(i))] + WIND/3600;    % NM per second
            X(i) = X(i)+v(1)*dt;  Y(i) = Y(i)+v(2)*dt;
            GS(i) = 3600*norm(v);  CRS(i) = mod(atan2d(v(1),v(2)),360);
        end
    end

    function updateConflicts()
        conflict(:) = false;
        for i = 1:N-1
            for j = i+1:N
                if hypot(X(i)-X(j),Y(i)-Y(j)) < 6 && abs(ALT(i)-ALT(j)) < 1500
                    conflict([i j]) = true;
                end
            end
        end
        for i = 1:N
            if conflict(i), status{i} = 'CONFLICT ALERT';
            elseif VS(i) > 300, status{i} = 'CLIMBING';
            elseif VS(i) < -300, status{i} = 'DESCENDING';
            elseif abs(TR(i)) > 0.4, status{i} = 'TURNING';
            else, status{i} = 'CRUISE';
            end
        end
    end

    function recordHistory()
        for i = 1:N
            H{i}(end+1,:) = [simT ALT(i) GS(i) VS(i) hypot(X(i),Y(i)) X(i) Y(i)];
            nmax = round(sldHist.Value);
            if size(H{i},1) > nmax, H{i} = H{i}(end-nmax+1:end,:); end
        end
    end

    % ---------- main loop ----------
    function tick()
        try
            tickBody();
        catch ME
            if ~closing && isvalid(fig), warning('AircraftRadar3D:tick','%s',ME.message); end
        end
    end

    function tickBody()
        if closing || ~isvalid(fig), return; end
        tNow = toc(t0);  dtR = min(tNow-tPrev,0.15);  tPrev = tNow;
        if ~paused
            dtS = dtR*sldSim.Value;  n = max(1,ceil(dtS/1));
            for k = 1:n, stepSim(dtS/n); end
            updateConflicts();
        end
        updateSweep(dtR,tNow);
        updateModels();
        updateOverlays(tNow);
        updateCamera();
        if tNow-lastRec > 0.2 && ~paused, recordHistory(); lastRec = tNow; end
        if tNow-lastSlow > 0.3, slowUI(); lastSlow = tNow; end
        drawnow limitrate
    end

    function updateSweep(dtR,tNow)
        prev = sweepAng;
        sweepAng = mod(sweepAng+sldSweep.Value*6*dtR,360);
        dth = mod(sweepAng-prev,360);
        b = mod(atan2d(X,Y),360);
        lastHit(mod(sweepAng-b,360) <= dth) = tNow;
        if chkSweep.Value
            ph = sweepAng - 75*(0:M-1)'/(M-1);
            V = zeros(2*M,3);  V(1:2:end,3) = 0.06;
            V(2:2:end,:) = [R*sind(ph) R*cosd(ph) 0.06*ones(M,1)];
            set(hSweep,'Vertices',V,'Visible','on');
            set(hSweepLn,'XData',[0 R*sind(sweepAng)],'YData',[0 R*cosd(sweepAng)],'Visible','on');
        else
            set(hSweep,'Visible','off');  set(hSweepLn,'Visible','off');
        end
        tau = 0.6*60/max(sldSweep.Value,1);
        inten = exp(-(tNow-lastHit)/tau);
        set(hEcho,'XData',X,'YData',Y,'ZData',ALT*ZS, ...
            'CData',max(inten,0.10).*C.green,'SizeData',90+400*inten);
    end

    function updateModels()
        for i = 1:N
            psi = (90-HDG(i))*pi/180;
            p = max(-12,min(12,VS(i)/250))*pi/180;
            r = max(-30,min(30,TR(i)*8))*pi/180;
            Rz = [cos(psi) -sin(psi) 0; sin(psi) cos(psi) 0; 0 0 1];
            Ry = [cos(-p) 0 sin(-p); 0 1 0; -sin(-p) 0 cos(-p)];
            Rx = [1 0 0; 0 cos(r) -sin(r); 0 sin(r) cos(r)];
            Rm = Rz*Ry*Rx;
            V = (mdl(i).V*2.4*msc(i)*R/100)*Rm.' + [X(i) Y(i) ALT(i)*ZS];
            col = baseCol(i,:);
            if i == sel, col = 0.45*col + 0.55*C.cyan; end
            if conflict(i), col = C.red; end
            set(hAC(i),'Vertices',V,'FaceColor',col);
        end
    end

    function updateOverlays(tNow)
        sc = R/100;
        if chkAlt.Value
            set(hDrop,'XData',[X';X';nan(1,N)],'YData',[Y';Y';nan(1,N)], ...
                'ZData',[ALT'*ZS; zeros(1,N); nan(1,N)],'Visible','on');
            set(hGnd,'XData',X,'YData',Y,'ZData',0.05*ones(N,1),'Visible','on');
        else
            set(hDrop,'Visible','off');  set(hGnd,'Visible','off');
        end
        for i = 1:N
            set(hLbl(i),'Position',[X(i) Y(i) ALT(i)*ZS+7*msc(i)*sc], ...
                'String',sprintf('%s  FL%03.0f',ids{i},ALT(i)/100), ...
                'Visible',iff(chkLbl.Value,'on','off'), ...
                'Color',iff(conflict(i),C.red,C.green));
        end
        % hover card
        if hoverIdx > 0
            i = hoverIdx;
            set(hHover,'Position',[X(i) Y(i) ALT(i)*ZS+12*msc(i)*sc],'Visible','on','String',{ ...
                ids{i}, models{i}, ['Altitude: ' fmtK(ALT(i)) ' ft'], ...
                sprintf('Speed: %.0f knots',GS(i)), sprintf('Heading: %03.0f%s',HDG(i),DEG), ...
                ['Status: ' status{i}]});
        else
            set(hHover,'Visible','off');
        end
        % selection ring, projected path
        if sel > 0
            i = sel;  pz = ALT(i)*ZS;
            rr = (3.2+0.5*sin(3*tNow))*msc(i)*sc;  t2 = linspace(0,2*pi,60);
            set(hRing1,'XData',X(i)+rr*cos(t2),'YData',Y(i)+rr*sin(t2),'ZData',pz*ones(size(t2)));
            set(hRing2,'XData',X(i)+rr*cos(t2),'YData',Y(i)*ones(size(t2)),'ZData',pz+rr*sin(t2));
            tm = (0:30:900)';
            hd = HDG(i)+TR(i)*min(tm,60);
            vx = TAS(i)/3600*sind(hd)+WIND(1)/3600;  vy = TAS(i)/3600*cosd(hd)+WIND(2)/3600;
            px = X(i)+cumtrapz(tm,vx);  py = Y(i)+cumtrapz(tm,vy);
            dA = sign(VS(i))*min(abs(VS(i))*tm/60,abs(tgtAlt(i)-ALT(i)));
            pzz = (ALT(i)+dA)*ZS;
            set(hProj,'XData',px,'YData',py,'ZData',pzz);
            ia = 4:6:numel(tm);  nv = hypot(vx(ia),vy(ia));
            set(hPMark,'XData',px(1:6:end),'YData',py(1:6:end),'ZData',pzz(1:6:end));
            set(hArrow,'XData',px(ia),'YData',py(ia),'ZData',pzz(ia), ...
                'UData',4*sc*vx(ia)./nv,'VData',4*sc*vy(ia)./nv,'WData',zeros(size(ia)));
        else
            set([hRing1 hRing2 hProj hPMark],'XData',NaN,'YData',NaN,'ZData',NaN);
        end
    end

    function updateCamera()
        if ~(followOn && sel > 0), return; end
        tgt = [X(sel) Y(sel) ALT(sel)*ZS];
        old = ax.CameraTarget;  nt = old+(tgt-old)*0.18;
        ax.CameraPosition = ax.CameraPosition+(nt-old);
        ax.CameraTarget = nt;
        if zoomGoal > 0
            off = ax.CameraPosition-nt;  dist = norm(off);
            nd = dist+(zoomGoal-dist)*0.12;
            ax.CameraPosition = nt+off/dist*nd;
            if abs(nd-zoomGoal) < 0.5, zoomGoal = 0; end
        end
    end

    % ---------- slow UI refresh (~3 Hz) ----------
    function slowUI()
        lblRange.Text = sprintf('RANGE %d NM',R);
        lblDet.Text   = sprintf('DETECTED %d',sum(hypot(X,Y) <= R));
        lblTrk.Text   = sprintf('ACTIVE TRACKS %d',N);
        lblTime.Text  = ['TIME ' datestr(now,'HH:MM:SS')];
        if any(conflict)
            lblSys.Text = sprintf('ALERT: %d IN CONFLICT',sum(conflict));  lblSys.FontColor = C.red;
        else
            lblSys.Text = 'SYSTEM NOMINAL';  lblSys.FontColor = C.green;
        end
        refreshList();  refreshTable();  refreshInfo();  updateTrails();  updateTelemetry();
    end

    function refreshList()
        it = cell(1,N);
        for i = 1:N
            it{i} = sprintf('%s %s FL%03.0f %3.0fkt %s',ids{i},catCode{i},ALT(i)/100,GS(i),status{i});
        end
        lst.Items = it;  lst.ItemsData = 1:N;
        if sel > 0, lst.Value = sel; end
    end

    function refreshTable()
        q = lower(strtrim(searchStr));
        hay = lower(strcat(ids,{' '},cats,{' '},models,{' '},calls,{' '},status));
        keep = find(contains(hay,q));
        dist = hypot(X,Y);
        switch ddSort.Value
            case 'Sort: Altitude', [~,o] = sort(ALT(keep),'descend');
            case 'Sort: Speed',    [~,o] = sort(GS(keep),'descend');
            case 'Sort: Distance', [~,o] = sort(dist(keep));
            otherwise,             o = 1:numel(keep);
        end
        viewIdx = keep(o);
        D = cell(numel(viewIdx),9);
        for r = 1:numel(viewIdx)
            i = viewIdx(r);
            D(r,:) = {ids{i},models{i},calls{i},round(ALT(i)),round(GS(i)), ...
                sprintf('%03.0f%s',HDG(i),DEG),round(VS(i)),round(dist(i)*10)/10,status{i}};
        end
        tbl.Data = D;
        try, removeStyle(tbl); catch, end
        rs = find(viewIdx == sel);
        if ~isempty(rs), addStyle(tbl,styleSel,'row',rs); end
        ra = find(conflict(viewIdx));
        if ~isempty(ra), addStyle(tbl,styleAlert,'row',ra); end
    end

    function refreshInfo()
        if sel == 0, return; end
        i = sel;
        lat = LAT0+Y(i)/60;  lon = LON0+X(i)/(60*cosd(LAT0));
        infoTbl.Data = { ...
            'Aircraft ID',ids{i}; 'Aircraft Type',cats{i}; 'Aircraft Model',models{i}; ...
            'Callsign',calls{i}; 'Status',status{i}; ...
            'Latitude',fmtLL(lat,'N','S',DEG); 'Longitude',fmtLL(lon,'E','W',DEG); ...
            'Altitude',[fmtK(ALT(i)) ' ft']; 'Ground Speed',sprintf('%.0f kt',GS(i)); ...
            'Air Speed',sprintf('%.0f kt',TAS(i)); 'Heading',sprintf('%03.0f%s',HDG(i),DEG); ...
            'Vertical Speed',sprintf('%+.0f fpm',VS(i)); ...
            'Distance',sprintf('%.1f NM',hypot(X(i),Y(i))); ...
            'Flight Direction',sprintf('%s (%03.0f%s)',compass16(CRS(i)),CRS(i),DEG); ...
            'Radar X / Y',sprintf('%+.1f / %+.1f NM',X(i),Y(i)); ...
            'Last Update',datestr(now,'HH:MM:SS'); 'Data Source','SIMULATED / DEMO'};
    end

    function updateTrails()
        for i = 1:N
            h = H{i};
            if chkPath.Value && ~isempty(h)
                set(hTrail(i),'XData',[h(:,6);X(i)],'YData',[h(:,7);Y(i)],'ZData',[h(:,2);ALT(i)]*ZS,'Visible','on');
            else
                set(hTrail(i),'Visible','off');
            end
            if i == sel, set(hTrail(i),'Color',C.cyan,'LineWidth',2.2);
            else, set(hTrail(i),'Color',[0.05 0.40 0.50],'LineWidth',0.8); end
        end
        if sel > 0 && ~isempty(H{sel}) && chkPath.Value
            h = H{sel};  hx = [h(:,6);X(sel)];  hy = [h(:,7);Y(sel)];  hz = [h(:,2);ALT(sel)]*ZS;
            mk = numel(hx):-12:1;
            set(hHMark,'XData',hx(mk),'YData',hy(mk),'ZData',hz(mk),'Visible','on');
            if chkAlt.Value
                nn = numel(mk);
                set(hStem,'XData',reshape([hx(mk)';hx(mk)';nan(1,nn)],[],1), ...
                    'YData',reshape([hy(mk)';hy(mk)';nan(1,nn)],[],1), ...
                    'ZData',reshape([hz(mk)';zeros(1,nn);nan(1,nn)],[],1),'Visible','on');
            else
                set(hStem,'Visible','off');
            end
        else
            set(hHMark,'Visible','off');  set(hStem,'Visible','off');
        end
    end

    function updateTelemetry()
        if sel == 0 || isempty(H{sel}), return; end
        h = H{sel};  t = (h(:,1)-h(end,1))/60;  y = h(:,teleCols(teleIdx));
        set(hTele,'XData',t,'YData',y);
        set(hTeleDot,'XData',0,'YData',y(end));
        ax2.XLim = [min(t(1),-0.01) 0.02];
        ax2.Title.String = sprintf('%s  -  %s  (SIMULATED)',ids{sel},teleNames{teleIdx});
    end
end

%% ===================== local helper functions =====================
function r = iff(c,a,b)
    if c, r = a; else, r = b; end
end

function s = fmtK(n)
    s = regexprep(sprintf('%d',round(n)),'\B(?=(\d{3})+(?!\d))',',');
end

function s = fmtLL(v,pos,neg,deg)
    if v >= 0, h = pos; else, h = neg; end
    s = sprintf('%.4f%s %s',abs(v),deg,h);
end

function s = compass16(a)
    nm = {'N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'};
    s = nm{mod(round(a/22.5),16)+1};
end

function [pf,sc,col,code] = categoryInfo(cat)
    % pf = [minTAS maxTAS minAlt maxAlt maxTurn(deg/s) maxClimb(fpm)]
    switch cat
        case 'Passenger',  pf = [380 520 12000 39000 2.5  2500]; sc = 1.00; col = [0.92 0.94 0.97]; code = 'PAX';
        case 'Cargo',      pf = [400 520 15000 41000 2.0  2000]; sc = 1.20; col = [0.70 0.76 0.85]; code = 'CGO';
        case 'Fighter',    pf = [350 700  5000 45000 8.0 12000]; sc = 0.80; col = [0.45 0.50 0.55]; code = 'FTR';
        case 'Business',   pf = [380 520 15000 45000 3.0  3500]; sc = 0.75; col = [0.95 0.93 0.85]; code = 'BIZ';
        otherwise,         pf = [ 80 160   500  8000 6.0  1500]; sc = 0.70; col = [0.95 0.65 0.20]; code = 'HEL';
    end
end

% ---------- lightweight triangle-mesh aircraft models (x fwd, y left, z up) ----------
function Mo = buildModel(cat)
    switch cat
        case 'Passenger'
            Mo = mergeM(ell([0 0 0],3.5,0.45,0.45), swept(0.6,1.7,0.5,3.2,1.6,-0.12,0.3), ...
                swept(-2.7,0.9,0.35,1.2,0.6,0.05,0.1), vfin(-2.0,1.2,1.5,0.3,0.9), ...
                ell([0.3 1.1 -0.4],0.6,0.22,0.22), ell([0.3 -1.1 -0.4],0.6,0.22,0.22));
        case 'Cargo'
            Mo = mergeM(ell([0 0 0],4.0,0.85,0.85), swept(0.9,2.0,0.8,3.6,1.3,0.45,0), ...
                vfin(-2.9,1.4,2.0,0.4,1.0), swept(-3.6,0.9,0.4,1.5,0.6,2.35,0), ...
                ell([0.3 1.2 -0.2],0.6,0.25,0.25), ell([0.3 -1.2 -0.2],0.6,0.25,0.25), ...
                ell([0.0 2.4 -0.1],0.6,0.25,0.25), ell([0.0 -2.4 -0.1],0.6,0.25,0.25));
        case 'Fighter'
            Mo = mergeM(ell([0 0 0],3.2,0.32,0.32), ell([0.9 0 0.2],0.7,0.16,0.14), ...
                swept(0.4,2.3,0.35,1.7,1.9,-0.02,0), swept(-2.4,0.9,0.3,0.9,0.6,0,0), ...
                vfin(-1.8,1.0,1.1,0.2,0.7), ell([-3.0 0 0],0.4,0.2,0.2));
        case 'Business'
            Mo = mergeM(ell([0 0 0],2.8,0.42,0.42), swept(0.2,1.3,0.45,2.5,1.0,-0.15,0.15), ...
                vfin(-2.0,1.0,1.4,0.3,0.8), swept(-2.9,0.7,0.3,0.9,0.4,1.75,0), ...
                ell([-1.7 0.65 0.3],0.6,0.2,0.2), ell([-1.7 -0.65 0.3],0.6,0.2,0.2));
        otherwise   % helicopter
            Mo = mergeM(ell([0 0 0],1.5,0.6,0.6), ell([-2.7 0 0.15],1.6,0.12,0.12), ...
                vfin(-3.8,0.6,0.9,0.15,0.3), ell([0 0 0.55],0.1,0.1,0.3), ...
                poly([-2.6 -0.1 0.85; 2.6 -0.1 0.85; 2.6 0.1 0.85; -2.6 0.1 0.85]), ...
                poly([-0.1 -2.6 0.85; 0.1 -2.6 0.85; 0.1 2.6 0.85; -0.1 2.6 0.85]));
    end
end

function Mo = ell(c,a,b,cc)
    [xs,ys,zs] = sphere(10);
    fv = surf2patch(a*xs+c(1),b*ys+c(2),cc*zs+c(3),'triangles');
    Mo.V = fv.vertices;  Mo.F = fv.faces;
end

function Mo = poly(P)
    k = size(P,1);
    Mo.V = P;  Mo.F = [ones(k-2,1) (2:k-1)' (3:k)'];
end

function Mo = swept(xl,cr,ct,span,sw,z,dih)   % mirrored swept wing / tailplane
    P = [xl 0 z; xl-cr 0 z; xl-sw-ct span z+dih; xl-sw span z+dih];
    Q = P;  Q(:,2) = -Q(:,2);  Q = Q([4 3 2 1],:);
    Mo = mergeM(poly(P),poly(Q));
end

function Mo = vfin(xl,cr,h,z0,sw)              % vertical stabiliser
    Mo = poly([xl 0 z0; xl-cr 0 z0; xl-sw-0.4*cr 0 z0+h; xl-sw 0 z0+h]);
end

function Mo = mergeM(varargin)
    V = [];  F = [];
    for k = 1:nargin
        m = varargin{k};
        F = [F; m.F+size(V,1)]; %#ok<AGROW>
        V = [V; m.V]; %#ok<AGROW>
    end
    Mo.V = V;  Mo.F = F;
end

function reorderChildren(axh, stat)
    % put the static scene objects behind the dynamic ones (no ismember on graphics handles)
    ch = axh.Children;  isS = false(size(ch));
    for q = 1:numel(ch)
        isS(q) = any(ch(q) == stat(:));
    end
    axh.Children = [ch(~isS); stat(:)];
end