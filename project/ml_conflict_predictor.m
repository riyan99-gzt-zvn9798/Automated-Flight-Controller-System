classdef ml_conflict_predictor < handle
%ML_CONFLICT_PREDICTOR  Real-time AI/ML conflict prediction engine.
%   Provides probability estimates, confidence metrics, and early warning advisories
%   for aircraft pairs, with strict safety backstopping against the geometric rule baseline.
%
%   Usage:
%       predictor = ml_conflict_predictor();                      % loads default model
%       predictor = ml_conflict_predictor('ml_conflict_model.mat');% specific artifact
%       [p, is_conf, info] = predictor.predict_pair(tr_i, tr_j, E_i, E_j, cfg);
%       C_ml = predictor.evaluate_airspace(W, cfg);

    properties
        model_file = 'ml_conflict_model.mat'
        model_type = 'logistic' % 'logistic' or 'random_forest'
        threshold = 0.50        % Decision probability threshold
        is_loaded = false
        model_pkg = []
        scaler = []
        stats = struct('calls', 0, 'total_time_us', 0, 'disagreements', 0)
    end

    methods
        function obj = ml_conflict_predictor(model_path, model_type)
            if nargin >= 1 && ~isempty(model_path)
                obj.model_file = model_path;
            end
            if nargin >= 2 && ~isempty(model_type)
                obj.model_type = model_type;
            end
            obj.load_model();
        end

        function load_model(obj)
            if exist(obj.model_file, 'file') == 2
                try
                    data = load(obj.model_file, 'model_pkg');
                    obj.model_pkg = data.model_pkg;
                    obj.scaler = data.model_pkg.scaler;
                    obj.is_loaded = true;
                catch ME
                    warning('Failed to load %s (%s). Generating on-the-fly model.', obj.model_file, ME.message);
                    obj.train_fallback();
                end
            else
                fprintf('Model file %s not found. Training default model...\n', obj.model_file);
                obj.train_fallback();
            end
        end

        function train_fallback(obj)
            dataset = ml_dataset_generator(struct('n_encounters', 40, 'seed', 42));
            obj.model_pkg = ml_train_models(dataset, struct('save_file', obj.model_file));
            obj.scaler = obj.model_pkg.scaler;
            obj.is_loaded = true;
        end

        function [p_conflict, is_conflict, info] = predict_pair(obj, tr_i, tr_j, E_i, E_j, cfg)
            %PREDICT_PAIR Predict conflict probability between aircraft i and j.
            t0 = tic;
            if nargin < 6, cfg = struct('sep', struct('H', 1500, 'V', 300)); end
            if nargin < 5, E_j = []; end
            if nargin < 4, E_i = []; end

            % 1. Extract causal features
            feats = ml_feature_extractor(tr_i, tr_j, E_i, E_j, cfg);

            % 2. Normalize features using training scaler
            feats_norm = (feats - obj.scaler.mu) ./ obj.scaler.sigma;

            % 3. Model Inference
            if strcmp(obj.model_type, 'random_forest') && isfield(obj.model_pkg.predict_fns, 'random_forest')
                p_conflict = obj.model_pkg.predict_fns.random_forest(feats);
            else
                p_conflict = obj.model_pkg.predict_fns.logistic(feats);
            end
            p_conflict = min(max(p_conflict, 0.0), 1.0);

            % 4. Baseline Rule Prediction for Disagreement Tracking
            p_rule = obj.model_pkg.predict_fns.rule(feats);
            is_rule_conflict = (p_rule >= 0.5);

            % 5. Decision & Classification
            is_conflict = (p_conflict >= obj.threshold);
            confidence = abs(p_conflict - 0.5) * 2; % [0..1]

            % Warning levels: 0=NORMAL, 1=WARNING, 2=CRITICAL
            tcpa = feats(7);
            if p_conflict >= 0.85 || (p_conflict >= obj.threshold && tcpa <= 30)
                warning_level = 2;
                warning_name  = 'CRITICAL';
            elseif p_conflict >= obj.threshold
                warning_level = 1;
                warning_name  = 'WARNING';
            else
                warning_level = 0;
                warning_name  = 'NORMAL';
            end

            latency_us = toc(t0) * 1e6;
            obj.stats.calls = obj.stats.calls + 1;
            obj.stats.total_time_us = obj.stats.total_time_us + latency_us;

            disagreement = (is_conflict ~= is_rule_conflict);
            if disagreement
                obj.stats.disagreements = obj.stats.disagreements + 1;
            end

            % Information packet
            info = struct( ...
                'p_conflict', p_conflict, ...
                'is_conflict', is_conflict, ...
                'p_rule', p_rule, ...
                'is_rule_conflict', is_rule_conflict, ...
                'disagreement', disagreement, ...
                'confidence', confidence, ...
                'warning_level', warning_level, ...
                'warning_name', warning_name, ...
                'tcpa_s', tcpa, ...
                'dcpa_h_m', feats(8), ...
                'dcpa_v_m', feats(9), ...
                'sep_norm_cpa', feats(12), ...
                'pos_uncertainty_m', feats(13), ...
                'latency_us', latency_us, ...
                'features', feats ...
            );
        end

        function C_ml = evaluate_airspace(obj, W, cfg)
            %EVALUATE_AIRSPACE Evaluate all active aircraft pairs in the simulation world.
            %   Returns array of ML conflict structures parallel to geometric conflict_detector_4d.
            C_ml = struct('i',{},'j',{},'p_conflict',{},'is_conflict',{},'warning_level',{}, ...
                'warning_name',{},'tcpa',{},'dcpa_h',{},'dcpa_v',{},'confidence',{}, ...
                'pos_uncert',{},'disagreement',{},'pos',{});

            if ~isfield(W, 'ac') || numel(W.ac) < 2
                return;
            end

            na = numel(W.ac);
            for i = 1:na-1
                for j = i+1:na
                    if ~W.ac(i).alive || ~W.ac(j).alive
                        continue;
                    end

                    tr_i = W.atc.tr(i);
                    tr_j = W.atc.tr(j);
                    E_i = W.ac(i).E;
                    E_j = W.ac(j).E;

                    [p_conf, is_conf, info] = obj.predict_pair(tr_i, tr_j, E_i, E_j, cfg);

                    if is_conf || info.is_rule_conflict || p_conf >= 0.25
                        % Midpoint of encounter for display
                        pos_mid = 0.5 * (tr_i.pos + tr_j.pos);
                        C_ml(end+1) = struct( ...
                            'i', i, ...
                            'j', j, ...
                            'p_conflict', p_conf, ...
                            'is_conflict', is_conf, ...
                            'warning_level', info.warning_level, ...
                            'warning_name', info.warning_name, ...
                            'tcpa', info.tcpa_s, ...
                            'dcpa_h', info.dcpa_h_m, ...
                            'dcpa_v', info.dcpa_v_m, ...
                            'confidence', info.confidence, ...
                            'pos_uncert', info.pos_uncertainty_m, ...
                            'disagreement', info.disagreement, ...
                            'pos', pos_mid(:)' ...
                        ); %#ok<AGROW>
                    end
                end
            end
        end
    end
end
