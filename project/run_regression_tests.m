function test_summary = run_regression_tests(verbose)
%RUN_REGRESSION_TESTS  Automated regression test suite for Unified ATC Simulation
%   Tests all 15 scenarios and verifies stability, metrics computation, and safety invariants.
%
%   Usage:
%       test_summary = run_regression_tests()
%       test_summary = run_regression_tests(true)   % verbose output

if nargin < 1, verbose = false; end

scenarios = { ...
    'nominal', 'head_on', 'crossing', 'overtaking', 'crosswind', ...
    'gps_failure', 'engine_failure', 'multi_conflict', 'restricted_zone', ...
    'sensor_noise', 'actuator_degradation', 'comm_loss', ...
    'altitude_bank_deviation', 'dense_airspace', 'converging_hub' ...
};

nScen = numel(scenarios);
results = struct( ...
    'scenario', scenarios, ...
    'status', cell(1, nScen), ...
    'duration_s', zeros(1, nScen), ...
    'sim_T', zeros(1, nScen), ...
    'minH_m', zeros(1, nScen), ...
    'minV_m', zeros(1, nScen), ...
    'los_events', zeros(1, nScen), ...
    'conflicts', zeros(1, nScen), ...
    'resolved', zeros(1, nScen), ...
    'ekf_rmse_m', zeros(1, nScen), ...
    'mpc_fallbacks', zeros(1, nScen), ...
    'error_msg', cell(1, nScen) ...
);

fprintf('========================================================================================\n');
fprintf('                STARTING REGRESSION TEST SUITE (%d SCENARIOS)                           \n', nScen);
fprintf('========================================================================================\n');
fprintf('%-24s | %-6s | %-8s | %-8s | %-4s | %-6s | %-9s | %-7s\n', ...
    'Scenario', 'Status', 'minH [m]', 'minV [m]', 'LoS', 'Confl', 'EKFpos[m]', 'Time[s]');
fprintf('----------------------------------------------------------------------------------------\n');

totalTic = tic;
nPassed = 0;

for k = 1:nScen
    scName = scenarios{k};
    testTic = tic;
    testT = 30; % Test duration for rapid regression coverage
    if any(strcmp(scName, {'gps_failure', 'engine_failure', 'altitude_bank_deviation'}))
        testT = 50; % Allow failure injection window to trigger
    end
    
    opts = struct('T', testT, 'viz', struct('enable', false), 'seed', 42 + k);
    
    try
        res = UnifiedFlightRadar3D(scName, opts);
        elapsed = toc(testTic);
        M = res.M;
        W = res.W;
        
        % Validate invariant: No NaNs/Infs in active aircraft state log
        activeX = W.L.x(:, 1:W.nlog, :);
        validState = all(isfinite(activeX), 'all') && (W.viol.nan == 0);
        
        results(k).status = 'PASS';
        results(k).duration_s = elapsed;
        results(k).sim_T = M.T;
        results(k).minH_m = M.minH;
        results(k).minV_m = M.minV;
        results(k).los_events = M.los_events;
        results(k).conflicts = M.conflicts;
        results(k).resolved = M.resolved;
        results(k).ekf_rmse_m = mean(M.ekf_pos_rmse);
        results(k).mpc_fallbacks = M.mpc_fallbacks;
        results(k).error_msg = '';
        
        if ~validState
            results(k).status = 'FAIL';
            results(k).error_msg = 'NaN/Inf detected in aircraft state logs';
        elseif M.crashed > 0
            results(k).status = 'WARN';
            results(k).error_msg = sprintf('%d aircraft marked dead', M.crashed);
        end
        
        if strcmp(results(k).status, 'PASS')
            nPassed = nPassed + 1;
        end
        
        fprintf('%-24s | %-6s | %8.0f | %8.0f | %4d | %6d | %9.2f | %7.2f\n', ...
            scName, results(k).status, M.minH, M.minV, M.los_events, M.conflicts, ...
            results(k).ekf_rmse_m, elapsed);
            
    catch ME
        elapsed = toc(testTic);
        results(k).status = 'ERROR';
        results(k).duration_s = elapsed;
        results(k).error_msg = ME.message;
        
        fprintf('%-24s | %-6s | %8s | %8s | %4s | %6s | %9s | %7.2f (Error: %s)\n', ...
            scName, 'ERROR', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', elapsed, ME.message);
    end
end

totalElapsed = toc(totalTic);
fprintf('----------------------------------------------------------------------------------------\n');
fprintf('REGRESSION SUMMARY: %d/%d passed (%.1f%%) in %.2f seconds.\n', ...
    nPassed, nScen, (nPassed / nScen) * 100, totalElapsed);
fprintf('========================================================================================\n\n');

test_summary = struct( ...
    'total', nScen, ...
    'passed', nPassed, ...
    'success_rate', nPassed / nScen, ...
    'total_elapsed_s', totalElapsed, ...
    'results', results ...
);
end
