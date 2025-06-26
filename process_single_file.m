function process_single_file(input_file, denoised_file, binary_file, pbn_ready_file, binarization_method, threshold_param)
    % Function to process a single fMRI file, handling different formats
    
    % Start timing the whole process
    total_start_time = tic;
    
    fprintf('\n========== fMRI PREPROCESSING PIPELINE FOR PBN ANALYSIS ==========\n');
    fprintf('Input file: %s\n', input_file);
    fprintf('Steps to be performed: 1) Denoising  2) Binarization\n\n');
    
    %% ===== STEP 1: DENOISING =====
    fprintf('STEP 1: DENOISING\n');
    fprintf('----------------------------------------\n');
    denoising_start_time = tic;
    
    % Load raw ROI time series
    fprintf('Loading raw ROI time series data...\n');
    
    try
        % First check if the file has a "Volume" column 
        % (Based on the example you provided)
        data_table = readtable(input_file);
        var_names = data_table.Properties.VariableNames;
        
        % Check if first column is "Volume" (as in your example)
        if any(strcmpi(var_names, 'Volume'))
            fprintf('Detected file format with Volume column and ROI names as column headers.\n');
            
            % Get ROI names from column headers (excluding "Volume" column)
            roi_names = var_names(~strcmpi(var_names, 'Volume'));
            
            % Extract time series data, skipping the Volume column
            time_series = table2array(data_table(:, ~strcmpi(var_names, 'Volume')))';
            
            fprintf('Successfully extracted %d ROI names from column headers.\n', length(roi_names));
            fprintf('First few ROI names: %s, %s, ...\n', roi_names{1}, roi_names{2});
        else
            % Try other formats (previous approach)
            if any(strcmpi(var_names, 'ROI'))
                % Case where ROI column exists
                roi_names = data_table.ROI;
                numeric_data = data_table;
                numeric_data.ROI = [];
                time_series = table2array(numeric_data);
                
            elseif iscell(data_table{:,1}) || isstring(data_table{:,1})
                % Case where first column might contain ROI names
                roi_names = data_table{:,1};
                time_series = table2array(data_table(:,2:end));
                
            else
                % No obvious ROI names, treat all as data and create generic names
                time_series = table2array(data_table);
                roi_names = var_names;
                
                % If all column names are numeric/generic, generate better names
                if all(contains(roi_names, 'Var')) || all(contains(roi_names, 'x'))
                    roi_names = cell(size(time_series, 2), 1);
                    for j = 1:size(time_series, 2)
                        roi_names{j} = sprintf('ROI_%d', j);
                    end
                    time_series = time_series';
                end
            end
            
            % Check if rows might be time points and columns might be ROIs
            % (We want ROIs as rows, time points as columns for processing)
            if size(time_series, 1) < size(time_series, 2)
                fprintf('Data appears to have ROIs as columns, transposing...\n');
                time_series = time_series';
            end
        end
        
        fprintf('Successfully loaded data with %d ROIs and %d time points.\n', ...
            size(time_series, 1), size(time_series, 2));
        
    catch ME
        fprintf('Error loading data: %s\n', ME.message);
        fprintf('Attempting alternative loading method...\n');
        
        try
            % Try to read as CSV and manually extract ROI names from headers
            fid = fopen(input_file);
            header_line = fgetl(fid);
            fclose(fid);
            
            % Parse header to get column names
            headers = strsplit(header_line, ',');
            
            % Check if first column might be "Volume" 
            if strcmpi(strtrim(headers{1}), 'Volume')
                fprintf('Detected Volume column via manual header extraction.\n');
                
                % Get ROI names (all columns except Volume)
                roi_names = headers(2:end);
                
                % Clean up any quotes or whitespace
                roi_names = cellfun(@(x) strtrim(x), roi_names, 'UniformOutput', false);
                roi_names = cellfun(@(x) strrep(x, '"', ''), roi_names, 'UniformOutput', false);
                
                % Load numeric data
                data = readmatrix(input_file);
                
                % Extract time series (excluding Volume column)
                time_series = data(:, 2:end)';
                
                fprintf('Successfully extracted %d ROI names from CSV header.\n', length(roi_names));
            else
                % No Volume column, treat all columns as ROIs
                roi_names = headers;
                
                % Clean up any quotes or whitespace
                roi_names = cellfun(@(x) strtrim(x), roi_names, 'UniformOutput', false);
                roi_names = cellfun(@(x) strrep(x, '"', ''), roi_names, 'UniformOutput', false);
                
                % Load numeric data
                data = readmatrix(input_file);
                
                % Extract time series
                time_series = data';
            end
            
        catch ME2
            fprintf('Error with alternative loading method: %s\n', ME2.message);
            error('Unable to load the data file. Please check the format.');
        end
    end
    
    % Rest of the processing code (denoising, binarization) remains the same
    % Initialize denoised array
    denoised_data = zeros(size(time_series));
    
    % Get time vector
    num_timepoints = size(time_series, 2);
    time_vector = 1:num_timepoints;
    
    % DENOISING STEP 1: Linear Detrending
    fprintf('Applying linear detrending to remove low-frequency drift...\n');
    for roi = 1:size(time_series, 1)
        % Extract current ROI's time series
        current_ts = time_series(roi, :);
        
        % Apply linear detrending by fitting and removing a linear trend
        [p, ~, mu] = polyfit(time_vector, current_ts, 1);
        trend = polyval(p, time_vector, [], mu);
        detrended_ts = current_ts - trend;
        
        % Store detrended data
        denoised_data(roi, :) = detrended_ts;
    end
    
    % DENOISING STEP 2: Bandpass Filtering
    fprintf('Applying bandpass filtering (0.01-0.1 Hz) to isolate neural signals...\n');
    
    % Assuming TR = 2 seconds (adjust based on your acquisition)
    TR = 2.0;
    
    % Calculate sampling frequency in Hz
    Fs = 1/TR;
    
    % Define bandpass filter parameters
    lowFreq = 0.01;  % 0.01 Hz lower cutoff
    highFreq = 0.1;  % 0.1 Hz upper cutoff
    
    % Check if Signal Processing Toolbox is available
    if exist('bandpass', 'file')
        % Use bandpass function if available
        for roi = 1:size(denoised_data, 1)
            denoised_data(roi, :) = bandpass(denoised_data(roi, :), [lowFreq, highFreq], Fs);
        end
    else
        % Fallback to manual implementation
        fprintf('Signal Processing Toolbox not detected. Using manual filter implementation.\n');
        
        % Convert cutoff frequencies to normalized frequencies
        Wn = [lowFreq, highFreq] / (Fs/2);
        
        % Design the filter
        [b, a] = butter(3, Wn, 'bandpass');
        
        % Apply the filter to each ROI
        for roi = 1:size(denoised_data, 1)
            denoised_data(roi, :) = filtfilt(b, a, denoised_data(roi, :));
        end
    end
    
    % Create output structure for denoised data
    denoised_table = table();
    denoised_table.ROI = roi_names';
    
    % Add timepoint columns
    for t = 1:num_timepoints
        col_name = sprintf('TP%d', t);
        denoised_table.(col_name) = denoised_data(:, t);
    end
    
    % Save denoised data
    fprintf('Saving denoised data to: %s\n', denoised_file);
    writetable(denoised_table, denoised_file);
    
    % Create directory for plots
    [denoised_dir, denoised_name, ~] = fileparts(denoised_file);
    plot_dir = fullfile(denoised_dir, 'preprocessing_plots');
    if ~exist(plot_dir, 'dir')
        mkdir(plot_dir);
    end
    
    % Create diagnostic plots for denoising
    fprintf('Creating diagnostic plots for denoised data...\n');
    
    % Select a subset of ROIs to plot
    max_plots = min(size(time_series, 1), 3);
    plot_indices = round(linspace(1, size(time_series, 1), max_plots));
    
    % Create a figure showing original vs. denoised
    h_denoised = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);
    
    for i = 1:length(plot_indices)
        roi = plot_indices(i);
        subplot(length(plot_indices), 1, i);
        
        % Plot original and denoised
        plot(time_vector, time_series(roi, :), 'b-', 'LineWidth', 1); hold on;
        plot(time_vector, denoised_data(roi, :), 'r-', 'LineWidth', 1);
        
        title(sprintf('ROI: %s', roi_names{roi}), 'Interpreter', 'none');
        if i == length(plot_indices)
            xlabel('Time Point');
        end
        ylabel('Signal');
        legend('Original', 'Denoised');
        grid on;
    end
    
    % Add global title
    sgtitle('Denoising Results: Linear Detrending + Bandpass Filtering');
    
    % Save the figure
    denoised_plot_file = fullfile(plot_dir, [denoised_name '_denoising.png']);
    saveas(h_denoised, denoised_plot_file);
    close(h_denoised);
    
    % Report denoising completion
    denoising_time = toc(denoising_start_time);
    fprintf('Denoising completed in %.2f seconds.\n', denoising_time);
    fprintf('Denoised data saved to: %s\n', denoised_file);
    fprintf('Denoising plots saved to: %s\n', denoised_plot_file);
    
   %% ===== STEP 2: BINARIZATION =====
fprintf('\nSTEP 2: BINARIZATION\n');
fprintf('----------------------------------------\n');
binarization_start_time = tic;

% Initialize binary matrix
binary_data = zeros(size(denoised_data));

% Apply the selected binarization method
fprintf('Applying %s binarization method...\n', binarization_method);

switch lower(binarization_method)
    case 'iterative_hmm'
        % Performs data-driven binarization using an iterative Hidden Markov
        % Model (HMM) with K-Means initialization. This method requires the
        % Statistics and Machine Learning Toolbox.
        fprintf('Applying Iterative HMM binarization method...\n');
        
        % Check for toolbox dependency as a safeguard
        if ~license('test', 'Statistics_Toolbox') && ~license('test', 'Statistics_and_Machine_Learning_Toolbox')
            error(['The ''iterative_hmm'' method requires the Statistics and Machine Learning Toolbox for the kmeans function. ' ...
                   'Please install it or choose a different binarization_method.']);
        end

        % Loop through each ROI to apply the model individually
        for roi = 1:size(denoised_data, 1)
            roi_ts = denoised_data(roi, :);
            num_timepoints = length(roi_ts);
            
            try
                % STEP 1: K-Means Initialization using MATLAB's built-in function
                % 'Replicates' is used to run kmeans multiple times with different
                % initializations to ensure a stable solution.
                [idx, C] = kmeans(roi_ts', 2, 'Replicates', 3);
                
                % Assign cluster centers to mu1 (low) and mu2 (high)
                if C(1) < C(2)
                    mu1 = C(1); mu2 = C(2);
                    low_idx = 1; high_idx = 2;
                else
                    mu1 = C(2); mu2 = C(1);
                    low_idx = 2; high_idx = 1;
                end
                
                % Calculate standard deviation for points within each cluster
                sigma1 = std(roi_ts(idx == low_idx));
                sigma2 = std(roi_ts(idx == high_idx));
                
                % Fallback for edge cases (e.g., a cluster has only one point)
                if isnan(sigma1) || sigma1 < eps, sigma1 = std(roi_ts) * 0.5; end
                if isnan(sigma2) || sigma2 < eps, sigma2 = std(roi_ts) * 0.5; end

                % STEP 2: Iterative Refinement Loop
                num_iterations = 3;
                current_binary_sequence = zeros(1, num_timepoints);

                for iter = 1:num_iterations
                    % Viterbi algorithm to find the most likely state sequence
                    stay_prob = 0.85; switch_prob = 0.15;
                    delta1=zeros(1,num_timepoints); delta2=zeros(1,num_timepoints);
                    path1=zeros(1,num_timepoints); path2=zeros(1,num_timepoints);
                    obs_prob1=zeros(1,num_timepoints); obs_prob2=zeros(1,num_timepoints);
                    
                    for t=1:num_timepoints
                        obs_prob1(t)=exp(-0.5*((roi_ts(t)-mu1)/sigma1)^2)/(sqrt(2*pi)*sigma1);
                        obs_prob2(t)=exp(-0.5*((roi_ts(t)-mu2)/sigma2)^2)/(sqrt(2*pi)*sigma2);
                        obs_prob1(t)=max(obs_prob1(t),eps);
                        obs_prob2(t)=max(obs_prob2(t),eps);
                    end
                    
                    delta1(1)=0.5*obs_prob1(1);
                    delta2(1)=0.5*obs_prob2(1);
                    
                    for t=2:num_timepoints
                        p11=delta1(t-1)*stay_prob; p21=delta2(t-1)*switch_prob;
                        if p11>=p21, delta1(t)=p11*obs_prob1(t); path1(t)=1; else, delta1(t)=p21*obs_prob1(t); path1(t)=2; end
                        p12=delta1(t-1)*switch_prob; p22=delta2(t-1)*stay_prob;
                        if p12>=p22, delta2(t)=p12*obs_prob2(t); path2(t)=1; else, delta2(t)=p22*obs_prob2(t); path2(t)=2; end
                        total_p=delta1(t)+delta2(t);
                        if total_p>eps, delta1(t)=delta1(t)/total_p; delta2(t)=delta2(t)/total_p; end
                    end
                    
                    states=zeros(1,num_timepoints);
                    if delta1(end)>=delta2(end), states(end)=1; else, states(end)=2; end
                    for t=num_timepoints-1:-1:1
                        if states(t+1)==1, states(t)=path1(t+1); else, states(t)=path2(t+1); end
                    end
                    current_binary_sequence=(states==2);

                    % Refine parameters based on the new state assignments
                    low_state_data = roi_ts(current_binary_sequence == 0);
                    high_state_data = roi_ts(current_binary_sequence == 1);
                    if ~isempty(low_state_data) && ~isempty(high_state_data)
                        mu1 = mean(low_state_data); sigma1 = std(low_state_data);
                        mu2 = mean(high_state_data); sigma2 = std(high_state_data);
                        sigma1 = max(sigma1, eps); sigma2 = max(sigma2, eps);
                    else
                        % One state has disappeared, stop iterating
                        break;
                    end
                end
                
                % STEP 3: Post-processing to enforce temporal stability
                min_state_duration = 3;
                validated_binary = current_binary_sequence;
                current_state = validated_binary(1);
                state_start = 1;
                
                for t = 2:num_timepoints
                    if validated_binary(t) ~= current_state
                        state_length = t - state_start;
                        if state_length < min_state_duration && state_start > 1
                            validated_binary(state_start:t-1) = validated_binary(state_start-1);
                        end
                        current_state = validated_binary(t);
                        state_start = t;
                    end
                end
                
                state_length = num_timepoints - state_start + 1;
                if state_length < min_state_duration && state_start > 1
                    validated_binary(state_start:end) = validated_binary(state_start-1);
                end
                
                % Store the final result for this ROI
                binary_data(roi, :) = validated_binary;
                
            catch ME
                % Fallback if HMM fails for any reason
                fprintf('  -> Warning: HMM binarization failed for ROI %s (%s).\n     Using median threshold as a fallback.\n', roi_names{roi}, ME.message);
                binary_data(roi, :) = roi_ts >= median(roi_ts);
            end
        end
        fprintf('Iterative HMM binarization completed.\n');


    case 'median'
        % Binarize using the median value for each ROI
        for roi = 1:size(denoised_data, 1)
            roi_ts = denoised_data(roi, :);
            threshold = median(roi_ts);
            binary_data(roi, :) = roi_ts >= threshold;
        end
        fprintf('Used median thresholding for each ROI\n');
        
    case 'mean'
        % Binarize using the mean value for each ROI
        for roi = 1:size(denoised_data, 1)
            roi_ts = denoised_data(roi, :);
            threshold = mean(roi_ts);
            binary_data(roi, :) = roi_ts >= threshold;
        end
        fprintf('Used mean thresholding for each ROI\n');
        
    case 'kmeans'
        % Binarize using k-means clustering (k=2) for each ROI
        for roi = 1:size(denoised_data, 1)
            roi_ts = denoised_data(roi, :);
            
            % Reshape for kmeans
            roi_ts_reshaped = roi_ts(:);
            
            % Apply k-means with k=2
            [idx, centroids] = kmeans(roi_ts_reshaped, 2);
            
            % Determine which cluster represents the "high" state (1)
            [~, high_cluster] = max(centroids);
            
            % Convert to binary (1 for the high cluster, 0 for the low cluster)
            binary_roi = (idx == high_cluster);
            
            % Reshape back to time series format
            binary_data(roi, :) = reshape(binary_roi, 1, []);
        end
        fprintf('Used k-means clustering (k=2) for each ROI\n');
        
    case 'threshold'
        % Binarize using a specific threshold value or percentile
        if ischar(threshold_param) && strcmpi(threshold_param, 'auto')
            % If 'auto', use the specified percentile of the data
            percentile = 75;  % Default to 75th percentile
            for roi = 1:size(denoised_data, 1)
                roi_ts = denoised_data(roi, :);
                threshold = prctile(roi_ts, percentile);
                binary_data(roi, :) = roi_ts >= threshold;
            end
            fprintf('Used %dth percentile thresholding for each ROI\n', percentile);
        else
            % Use the provided threshold directly
            for roi = 1:size(denoised_data, 1)
                roi_ts = denoised_data(roi, :);
                binary_data(roi, :) = roi_ts >= threshold_param;
            end
            fprintf('Used fixed threshold value of %f for all ROIs\n', threshold_param);
        end
        
    case 'hybrid_resting'
    % Hybrid method optimized for resting state fMRI - SENSITIVE VERSION
    fprintf('Applying hybrid resting state binarization method (sensitive)...\n');
    fprintf('Steps: 1) ROI-specific MAD thresholding 2) Temporal smoothing 3) State validation\n');
    
    % Initialize statistics tracking
    roi_stats = struct();
    
    for roi = 1:size(denoised_data, 1)
        roi_ts = denoised_data(roi, :);
        
        % Step 1: Calculate threshold using Median Absolute Deviation (MAD)
        roi_median = median(roi_ts);
        roi_mad = mad(roi_ts, 1); % MAD with median centering
        
        % MODIFIED: More sensitive threshold - use smaller MAD multiplier
        mad_multiplier = 0.2; % REDUCED from 0.5 to 0.2 for more sensitivity
        threshold = roi_median + mad_multiplier * roi_mad;
        
        % Initial binarization
        initial_binary = roi_ts >= threshold;
        
        % Step 2: MODIFIED - Less aggressive temporal smoothing
        % Only apply smoothing to isolated single-point spikes
        smoothed_binary = initial_binary;
        for t = 2:(length(initial_binary)-1)
            % Only smooth if it's a single isolated spike (differs from both neighbors)
            if initial_binary(t) ~= initial_binary(t-1) && initial_binary(t) ~= initial_binary(t+1)
                % Check if it's really isolated (no similar states in 5-point window)
                window_start = max(1, t-2);
                window_end = min(length(initial_binary), t+2);
                window_values = initial_binary(window_start:window_end);
                
                % Only smooth if this is truly an outlier
                if sum(window_values == initial_binary(t)) == 1
                    % Take majority vote from 3-point window
                    window_sum = sum(initial_binary(t-1:t+1));
                    smoothed_binary(t) = window_sum >= 2;
                end
            end
        end
        
        % Step 3: MODIFIED - Reduced minimum state duration
        min_state_duration = 2; % REDUCED from 3 to 2 TRs for more transitions
        validated_binary = smoothed_binary;
        
        % Find state transitions
        state_changes = find(diff([0, smoothed_binary, 0]) ~= 0);
        
        % Process each state segment - only merge very short segments
        for seg = 1:2:length(state_changes)-1
            start_idx = state_changes(seg);
            end_idx = state_changes(seg+1) - 1;
            segment_length = end_idx - start_idx + 1;
            
            % Only merge segments shorter than minimum duration
            if segment_length < min_state_duration
                % Determine which neighboring state to merge with
                if start_idx > 1 && end_idx < length(validated_binary)
                    % Has both neighbors - merge with the longer one
                    left_length = start_idx - 1;
                    right_length = length(validated_binary) - end_idx;
                    
                    if left_length >= right_length
                        validated_binary(start_idx:end_idx) = validated_binary(start_idx-1);
                    else
                        validated_binary(start_idx:end_idx) = validated_binary(end_idx+1);
                    end
                elseif start_idx > 1
                    % Only left neighbor
                    validated_binary(start_idx:end_idx) = validated_binary(start_idx-1);
                elseif end_idx < length(validated_binary)
                    % Only right neighbor
                    validated_binary(start_idx:end_idx) = validated_binary(end_idx+1);
                end
            end
        end
        
        % Store the final binary result
        binary_data(roi, :) = validated_binary;
        
        % Store statistics for this ROI
        roi_stats(roi).name = roi_names{roi};
        roi_stats(roi).threshold = threshold;
        roi_stats(roi).state_ratio = mean(validated_binary);
        roi_stats(roi).num_transitions = sum(diff(validated_binary) ~= 0);
        roi_stats(roi).median_signal = roi_median;
        roi_stats(roi).mad_signal = roi_mad;
        
        % Print stats for first few ROIs
        if roi <= 5
            fprintf('  ROI %d (%s): %.1f%% high state, %d transitions, threshold=%.3f\n', ...
                roi, roi_names{roi}, roi_stats(roi).state_ratio*100, ...
                roi_stats(roi).num_transitions, threshold);
        end
    end
    
    % Validate overall binarization quality
    overall_transition_rate = mean([roi_stats.num_transitions]);
    mean_high_state_percent = mean([roi_stats.state_ratio]) * 100;
    
    fprintf('Hybrid resting state binarization (sensitive) completed:\n');
    fprintf('  Average transitions per ROI: %.1f\n', overall_transition_rate);
    fprintf('  Average high state percentage: %.1f%%\n', mean_high_state_percent);
    fprintf('  MAD multiplier used: %.1f\n', mad_multiplier);
    fprintf('  Minimum state duration: %d TRs\n', min_state_duration);
    
    % Check for potential issues - adjusted thresholds for sensitive method
    low_activity_rois = sum([roi_stats.state_ratio] < 0.05); % Changed from 0.1
    high_activity_rois = sum([roi_stats.state_ratio] > 0.95); % Changed from 0.9
    low_transition_rois = sum([roi_stats.num_transitions] < 2); % New warning
    
    if low_activity_rois > 0
        fprintf('  Warning: %d ROIs have very low activity (<5%% high state)\n', low_activity_rois);
    end
    if high_activity_rois > 0
        fprintf('  Warning: %d ROIs have very high activity (>95%% high state)\n', high_activity_rois);
    end
    if low_transition_rois > 0
        fprintf('  Note: %d ROIs have very few transitions (<2 transitions)\n', low_transition_rois);
    end
        
    case 'amplitude_percentile'
        % Alternative method: Use amplitude-based percentile thresholding
        fprintf('Applying amplitude-based percentile binarization...\n');
        
        percentile_threshold = 60; % Use 60th percentile as default
        if ~isempty(threshold_param) && isnumeric(threshold_param)
            percentile_threshold = threshold_param;
        end
        
        for roi = 1:size(denoised_data, 1)
            roi_ts = denoised_data(roi, :);
            threshold = prctile(roi_ts, percentile_threshold);
            binary_data(roi, :) = roi_ts >= threshold;
        end
        fprintf('Used %dth percentile thresholding for each ROI\n', percentile_threshold);
    
    case 'iterative_hmm'
    % Iterative HMM method inspired by Mishra et al. (2011)
    fprintf('Applying Iterative HMM binarization method...\n');
    
    num_iterations = 3; % Define how many times to refine
    
    % --- Initial Parameter Estimation (Iteration 0) ---
    % Use the same robust initialization as your 'hmm_resting_revised'
    fprintf('  Iteration 0: Initializing parameters...\n');
    roi_ts = denoised_data(roi, :);
    mu1 = prctile(roi_ts, 25);
    mu2 = prctile(roi_ts, 75);
    sigma1 = std(roi_ts(roi_ts <= median(roi_ts)));
    sigma2 = std(roi_ts(roi_ts > median(roi_ts)));
    
    % Ensure parameters are valid
    sigma1 = max(sigma1, eps * 1000);
    sigma2 = max(sigma2, eps * 1000);
    if abs(mu2 - mu1) < eps * 1000
        mu1 = mean(roi_ts) - std(roi_ts) * 0.5;
        mu2 = mean(roi_ts) + std(roi_ts) * 0.5;
    end

    % --- Iterative Refinement Loop ---
    for iter = 1:num_iterations
        fprintf('  Iteration %d: Running Viterbi and refining parameters...\n', iter);
        
        % Step 1: Run the Viterbi algorithm with the current parameters (mu1, mu2, sigma1, sigma2)
        % (This is the full Viterbi code block from your 'hmm_resting_revised' case)
        % ... [Viterbi algorithm code goes here] ...
        % The output is a binary sequence, let's call it 'current_binary_sequence'
        
        % Step 2: Refine the parameters based on the new sequence
        low_state_data = roi_ts(current_binary_sequence == 0);
        high_state_data = roi_ts(current_binary_sequence == 1);
        
        if ~isempty(low_state_data) && ~isempty(high_state_data)
            mu1 = mean(low_state_data);
            sigma1 = std(low_state_data);
            mu2 = mean(high_state_data);
            sigma2 = std(high_state_data);
            
            % Ensure parameters remain valid
            sigma1 = max(sigma1, eps * 1000);
            sigma2 = max(sigma2, eps * 1000);
        else
            fprintf('  Warning: One state disappeared for ROI %d. Stopping iteration.\n', roi);
            break; % Exit loop if one state is empty
        end
    end
    
    % The final 'current_binary_sequence' is your result for this ROI
    binary_data(roi, :) = current_binary_sequence;
        
    case 'hmm_resting_revised'
    % Revised HMM-based binarization method for resting-state fMRI
    fprintf('Applying revised HMM resting-state binarization method...\n');
    fprintf('Steps: 1) ROI-specific HMM modeling 2) Viterbi algorithm state inference 3) State validation\n');
    
    % Initialize statistics tracking
    roi_stats = struct();
    
    for roi = 1:size(denoised_data, 1)
        roi_ts = denoised_data(roi, :);
        num_timepoints = length(roi_ts);
        
        try
            % Step 1: Establish HMM model for each ROI separately
            roi_mean = mean(roi_ts);
            roi_std = std(roi_ts);
            roi_median = median(roi_ts);
            
            % Robust parameter estimation
            mu1 = prctile(roi_ts, 25);  % Low state mean
            mu2 = prctile(roi_ts, 75);  % High state mean
            
            % Calculate standard deviations for each state
            low_mask = roi_ts <= roi_median;
            high_mask = roi_ts > roi_median;
            
            if sum(low_mask) > 1
                sigma1 = std(roi_ts(low_mask));
            else
                sigma1 = roi_std * 0.5;
            end
            
            if sum(high_mask) > 1
                sigma2 = std(roi_ts(high_mask));
            else
                sigma2 = roi_std * 0.5;
            end
            
            % Ensure minimum variance and state separation
            sigma1 = max(sigma1, eps * 1000);
            sigma2 = max(sigma2, eps * 1000);
            
            if abs(mu2 - mu1) < eps * 1000
                mu1 = roi_mean - roi_std * 0.5;
                mu2 = roi_mean + roi_std * 0.5;
            end
            
            % Transition probabilities
            stay_prob = 0.85;
            switch_prob = 0.15;
            
            % Step 2: Simplified Viterbi algorithm with explicit array handling
            % Initialize arrays
            delta1 = zeros(1, num_timepoints);  % State 1 path probabilities
            delta2 = zeros(1, num_timepoints);  % State 2 path probabilities
            path1 = zeros(1, num_timepoints);   % Best previous state for state 1
            path2 = zeros(1, num_timepoints);   % Best previous state for state 2
            
            % Calculate observation probabilities for all timepoints
            obs_prob1 = zeros(1, num_timepoints);
            obs_prob2 = zeros(1, num_timepoints);
            
            for t = 1:num_timepoints
                % Gaussian probability for each state
                obs_prob1(t) = exp(-0.5 * ((roi_ts(t) - mu1) / sigma1)^2) / (sqrt(2*pi) * sigma1);
                obs_prob2(t) = exp(-0.5 * ((roi_ts(t) - mu2) / sigma2)^2) / (sqrt(2*pi) * sigma2);
                
                % Prevent numerical issues
                obs_prob1(t) = max(obs_prob1(t), eps);
                obs_prob2(t) = max(obs_prob2(t), eps);
            end
            
            % Initialize first timepoint (equal probability)
            delta1(1) = 0.5 * obs_prob1(1);
            delta2(1) = 0.5 * obs_prob2(1);
            path1(1) = 1;  % Dummy value
            path2(1) = 2;  % Dummy value
            
            % Forward pass - explicitly handle each state
            for t = 2:num_timepoints
                % For state 1 at time t
                prob_from_1 = delta1(t-1) * stay_prob;   % Stay in state 1
                prob_from_2 = delta2(t-1) * switch_prob; % Switch from state 2
                
                if prob_from_1 >= prob_from_2
                    delta1(t) = prob_from_1 * obs_prob1(t);
                    path1(t) = 1;  % Came from state 1
                else
                    delta1(t) = prob_from_2 * obs_prob1(t);
                    path1(t) = 2;  % Came from state 2
                end
                
                % For state 2 at time t
                prob_from_1 = delta1(t-1) * switch_prob; % Switch from state 1
                prob_from_2 = delta2(t-1) * stay_prob;   % Stay in state 2
                
                if prob_from_1 >= prob_from_2
                    delta2(t) = prob_from_1 * obs_prob2(t);
                    path2(t) = 1;  % Came from state 1
                else
                    delta2(t) = prob_from_2 * obs_prob2(t);
                    path2(t) = 2;  % Came from state 2
                end
                
                % Normalize to prevent underflow
                total_prob = delta1(t) + delta2(t);
                if total_prob > eps
                    delta1(t) = delta1(t) / total_prob;
                    delta2(t) = delta2(t) / total_prob;
                end
            end
            
            % Backward pass - find optimal state sequence
            states = zeros(1, num_timepoints);
            
            % Find best final state
            if delta1(num_timepoints) >= delta2(num_timepoints)
                states(num_timepoints) = 1;
            else
                states(num_timepoints) = 2;
            end
            
            % Backtrack
            for t = num_timepoints-1:-1:1
                if states(t+1) == 1
                    states(t) = path1(t+1);
                else
                    states(t) = path2(t+1);
                end
            end
            
            % Convert to binary (state 2 = high = 1, state 1 = low = 0)
            initial_binary = (states == 2);
            
            % Step 3: Post-processing
            min_state_duration = 3;
            validated_binary = initial_binary;
            
            % Simple state duration enforcement
            current_state = validated_binary(1);
            state_start = 1;
            
            for t = 2:num_timepoints
                if validated_binary(t) ~= current_state
                    % State change detected
                    state_length = t - state_start;
                    
                    % If previous state was too short, extend it
                    if state_length < min_state_duration && state_start > 1
                        validated_binary(state_start:t-1) = validated_binary(state_start-1);
                    end
                    
                    current_state = validated_binary(t);
                    state_start = t;
                end
            end
            
            % Handle the last segment
            state_length = num_timepoints - state_start + 1;
            if state_length < min_state_duration && state_start > 1
                validated_binary(state_start:end) = validated_binary(state_start-1);
            end
            
            % Store results
            binary_data(roi, :) = validated_binary;
            
            % Store statistics
            roi_stats(roi).name = roi_names{roi};
            roi_stats(roi).mu1 = mu1;
            roi_stats(roi).mu2 = mu2;
            roi_stats(roi).sigma1 = sigma1;
            roi_stats(roi).sigma2 = sigma2;
            roi_stats(roi).state_ratio = mean(validated_binary);
            roi_stats(roi).num_transitions = sum(diff(validated_binary) ~= 0);
            roi_stats(roi).stay_probability = stay_prob;
            
            % Print statistics for first few ROIs
            if roi <= 5
                fprintf('  ROI %d (%s): %.1f%% high state, %d transitions, μ1=%.2f, μ2=%.2f\n', ...
                    roi, roi_names{roi}, roi_stats(roi).state_ratio*100, ...
                    roi_stats(roi).num_transitions, mu1, mu2);
            end
            
        catch ME
            % Fallback to simple median thresholding if HMM fails
            fprintf('  Warning: HMM failed for ROI %d, using median fallback\n', roi);
            threshold = median(roi_ts);
            simple_binary = roi_ts >= threshold;
            binary_data(roi, :) = simple_binary;
            
            % Store fallback statistics
            roi_stats(roi).name = roi_names{roi};
            roi_stats(roi).mu1 = NaN;
            roi_stats(roi).mu2 = NaN;
            roi_stats(roi).sigma1 = NaN;
            roi_stats(roi).sigma2 = NaN;
            roi_stats(roi).state_ratio = mean(simple_binary);
            roi_stats(roi).num_transitions = sum(diff(simple_binary) ~= 0);
            roi_stats(roi).stay_probability = NaN;
        end
    end
    
    % Overall quality assessment
    valid_stats = ~isnan([roi_stats.mu1]);
    if sum(valid_stats) > 0
        overall_transition_rate = mean([roi_stats(valid_stats).num_transitions]);
        mean_high_state_percent = mean([roi_stats(valid_stats).state_ratio]) * 100;
    else
        overall_transition_rate = mean([roi_stats.num_transitions]);
        mean_high_state_percent = mean([roi_stats.state_ratio]) * 100;
    end
    
    fprintf('Revised HMM resting-state binarization completed:\n');
    fprintf('  Average transitions per ROI: %.1f\n', overall_transition_rate);
    fprintf('  Average high state percentage: %.1f%%\n', mean_high_state_percent);
    fprintf('  Successful HMM fits: %d/%d ROIs\n', sum(valid_stats), length(roi_stats));
    
    % Quality checks
    low_activity_rois = sum([roi_stats.state_ratio] < 0.1);
    high_activity_rois = sum([roi_stats.state_ratio] > 0.9);
    
    if low_activity_rois > 0
        fprintf('  Warning: %d ROIs have very low activity (<10%% high state)\n', low_activity_rois);
    end
    if high_activity_rois > 0
        fprintf('  Warning: %d ROIs have very high activity (>90%% high state)\n', high_activity_rois);
    end
        
        
        
       
        
    otherwise
        error('Unknown binarization method. Choose from: median, mean, kmeans, threshold, hybrid_resting, amplitude_percentile, network_state');
end

% Create output structure for binary data
binary_table = table();
binary_table.ROI = roi_names';

% Add timepoint columns
for t = 1:num_timepoints
    col_name = sprintf('TP%d', t);
    binary_table.(col_name) = binary_data(:, t);
end

% Save binary data
fprintf('Saving binary data to: %s\n', binary_file);
writetable(binary_table, binary_file);

% Create diagnostic plots for binarization
fprintf('Creating diagnostic plots for binarized data...\n');

% Create a figure showing denoised vs. binary
h_binary = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);

for i = 1:length(plot_indices)
    roi = plot_indices(i);
    subplot(length(plot_indices), 1, i);
    
    % Plot denoised with binary overlay
    yyaxis left
    plot(time_vector, denoised_data(roi, :), 'b-', 'LineWidth', 1.5);
    ylabel('Denoised Signal');
    
    yyaxis right
    stem(time_vector, binary_data(roi, :), 'r-', 'LineWidth', 1, 'MarkerSize', 4);
    ylim([-0.1, 1.1]);
    ylabel('Binary State (0/1)');
    
    title(sprintf('ROI: %s', roi_names{roi}), 'Interpreter', 'none');
    if i == length(plot_indices)
        xlabel('Time Point');
    end
    legend('Denoised Signal', 'Binary State');
    grid on;
end

% Add global title
sgtitle(['Binarization Results: ' upper(binarization_method) ' Method']);

% Save the figure
binary_plot_file = fullfile(plot_dir, [denoised_name '_binarization.png']);
saveas(h_binary, binary_plot_file);
close(h_binary);

% Create state transition visualization
h_transition = figure('Visible', 'off', 'Position', [100, 100, 1000, 600]);

% Use a subset of ROIs for visualization (max 20)
max_display_rois = min(size(binary_data, 1), 20);
if size(binary_data, 1) > max_display_rois
    display_indices = round(linspace(1, size(binary_data, 1), max_display_rois));
    display_data = binary_data(display_indices, :);
    display_names = roi_names(display_indices);
else
    display_data = binary_data;
    display_names = roi_names;
end

% Create heatmap of binary states
imagesc(display_data);
colormap([0.9 0.9 0.9; 0.2 0.4 0.8]);  % Light gray for 0, blue for 1

% Add labels and title
ylabel('Brain Region');
xlabel('Time Point');
title('Binary State Transitions Over Time');

% Add ROI names as y-tick labels
yticks(1:length(display_names));
yticklabels(display_names);

% Adjust font size if many ROIs
if length(display_names) > 10
    set(gca, 'FontSize', 8);
end

% Add colorbar
colorbar('Ticks', [0.25, 0.75], 'TickLabels', {'0 (Low)', '1 (High)'});

% Save the figure
transition_plot_file = fullfile(plot_dir, [denoised_name '_state_transitions.png']);
saveas(h_transition, transition_plot_file);
close(h_transition);

% For PBN analysis, we need to transpose the data so that:
% - Rows = time points (state vectors)
% - Columns = ROIs (nodes)
fprintf('Transposing data matrix for Python compatibility (Time x ROIs).\n');
pbn_ready_matrix = binary_data';

% --- THIS IS THE CRITICAL FIX ---
% Instead of using writetable, which adds headers and can cause format issues,
% we use writematrix. This function saves only the numerical data in a clean,
% standard comma-separated format that Python/Pandas can easily read.
fprintf('Saving PBN-ready data to: %s\n', pbn_ready_file);
writematrix(pbn_ready_matrix, pbn_ready_file);

% --- (Optional but Recommended) Save Metadata Separately ---
[pbn_dir, pbn_name, ~] = fileparts(pbn_ready_file);
metadata_file = fullfile(pbn_dir, [pbn_name, '_metadata.txt']);
fid = fopen(metadata_file, 'w');
if fid ~= -1
    fprintf(fid, 'PBN-Ready Data File: %s\n', [pbn_name, '.csv']);
    fprintf(fid, 'Original Data Source: %s\n', input_file);
    fprintf(fid, 'Processing Date: %s\n', datestr(now));
    fprintf(fid, 'Binarization Method: %s\n', binarization_method);
    fprintf(fid, 'Number of Brain Regions (Nodes): %d\n', size(binary_data, 1));
    fprintf(fid, 'Number of Time Points: %d\n', size(binary_data, 2));
    fprintf(fid, '\nBrain Region Names (in order):\n');
    for j = 1:length(roi_names)
        fprintf(fid, '%d. %s\n', j, roi_names{j});
    end
    fclose(fid);
    fprintf('Metadata with ROI names saved to: %s\n', metadata_file);
end


% Calculate state statistics
binary_stats = table();
binary_stats.ROI = roi_names';
binary_stats.PercentHighState = mean(binary_data, 2) * 100;

% Number of state transitions
state_transitions = zeros(size(binary_data, 1), 1);
for roi = 1:size(binary_data, 1)
    transitions = diff(binary_data(roi, :)) ~= 0;
    state_transitions(roi) = sum(transitions);
end
binary_stats.StateTransitions = state_transitions;

% Save statistics
stats_file = fullfile(pbn_dir, [pbn_name '_stats.csv']);
writetable(binary_stats, stats_file);
fprintf('Binary state statistics saved to: %s\n', stats_file);

% Report overall completion
total_time = toc(total_start_time);
fprintf('\n========== PREPROCESSING COMPLETED ==========\n');
fprintf('Total processing time: %.2f seconds\n', total_time);


