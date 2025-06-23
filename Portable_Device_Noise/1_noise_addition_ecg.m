function ecg_noise_generator_system()
    % COMPREHENSIVE ECG NOISE ADDITION SYSTEM
    % Simulates realistic portable ECG device noise conditions
    % Generates multiple noise types and SNR levels for robustness testing
    
    % Define paths
    original_dataset_path = 'C:\Users\henry\Downloads\ECG-Dx\Organized_Dataset';
    noisy_dataset_path = 'C:\Users\henry\Downloads\ECG-Dx\Noisy_ECG_Dataset';
    analysis_path = fullfile(noisy_dataset_path, 'Noise_Analysis');
    
    % Create output directories
    if ~exist(noisy_dataset_path, 'dir')
        mkdir(noisy_dataset_path);
    end
    if ~exist(analysis_path, 'dir')
        mkdir(analysis_path);
    end
    
    % ECG and noise parameters
    fs = 500; % Sampling frequency (Hz)
    duration_seconds = 4; % Process first 4 seconds
    target_samples = fs * duration_seconds; % 2000 samples
    
    % Define noise types and parameters for portable ECG devices
    noise_types = {
        'gaussian',        % White Gaussian noise (electronic)
        'powerline',       % 50/60 Hz powerline interference
        'baseline_wander', % Low frequency motion artifacts
        'muscle_artifact', % EMG contamination
        'motion_artifact', % High frequency motion noise
        'electrode_noise', % Contact impedance variations
        'combined'         % Realistic combination of all noise types
    };
    
    % SNR levels for research analysis (dB)
    snr_levels = [30, 20, 15, 10, 5, 0]; % High to low quality
    
    fprintf('=== ECG NOISE ADDITION SYSTEM FOR PORTABLE DEVICE SIMULATION ===\n');
    fprintf('Original dataset: %s\n', original_dataset_path);
    fprintf('Noisy dataset output: %s\n', noisy_dataset_path);
    fprintf('Processing: Lead II only (first %d seconds)\n', duration_seconds);
    fprintf('Noise types: %d different types\n', length(noise_types));
    fprintf('SNR levels: %s dB\n', mat2str(snr_levels));
    fprintf('Analysis output: %s\n\n', analysis_path);
    
    % Process datasets
    datasets = {'training', 'validation'};
    groups = {'AFIB', 'SB', 'SR'};
    
    % Initialize counters
    total_processed = 0;
    total_noisy_generated = 0;
    processing_errors = 0;
    
    % Main processing loop
    for dataset_idx = 1:length(datasets)
        dataset_name = datasets{dataset_idx};
        fprintf('Processing %s dataset...\n', dataset_name);
        
        for group_idx = 1:length(groups)
            group_name = groups{group_idx};
            
            % Define input and output directories
            input_dir = fullfile(original_dataset_path, dataset_name, group_name);
            
            if ~exist(input_dir, 'dir')
                fprintf('  Warning: Directory not found: %s\n', input_dir);
                continue;
            end
            
            % Get all .mat files (excluding metadata)
            mat_files = dir(fullfile(input_dir, '*.mat'));
            mat_files = mat_files(~strcmp({mat_files.name}, 'metadata.mat'));
            
            fprintf('  Processing %s group: %d files\n', group_name, length(mat_files));
            
            % Process each ECG file
            for file_idx = 1:length(mat_files)
                total_processed = total_processed + 1;
                
                mat_file_path = fullfile(input_dir, mat_files(file_idx).name);
                [~, base_name, ~] = fileparts(mat_files(file_idx).name);
                
                try
                    % Load and extract ECG data
                    ecg_data = load(mat_file_path);
                    ecg_signals = extract_ecg_signals(ecg_data);
                    
                    if isempty(ecg_signals) || size(ecg_signals, 1) < 2
                        fprintf('    Warning: Lead II not available in %s\n', mat_files(file_idx).name);
                        processing_errors = processing_errors + 1;
                        continue;
                    end
                    
                    % Extract Lead II and truncate to first 4 seconds
                    lead2_signal = ecg_signals(2, :);
                    if length(lead2_signal) < target_samples
                        fprintf('    Warning: Signal too short in %s\n', mat_files(file_idx).name);
                        processing_errors = processing_errors + 1;
                        continue;
                    end
                    
                    lead2_signal = lead2_signal(1:target_samples);
                    lead2_signal = double(lead2_signal);
                    
                    % Generate noisy versions for each noise type and SNR level
                    for noise_idx = 1:length(noise_types)
                        noise_type = noise_types{noise_idx};
                        
                        for snr_idx = 1:length(snr_levels)
                            snr_db = snr_levels(snr_idx);
                            
                            % Generate noisy signal
                            noisy_signal = add_portable_ecg_noise(lead2_signal, fs, noise_type, snr_db);
                            
                            % Create output directory structure
                            output_dir = fullfile(noisy_dataset_path, sprintf('SNR_%02ddB', snr_db), ...
                                                noise_type, dataset_name, group_name);
                            if ~exist(output_dir, 'dir')
                                mkdir(output_dir);
                            end
                            
                            % Save noisy signal
                            noisy_filename = sprintf('%s_NOISE_%s_SNR%02d.mat', base_name, noise_type, snr_db);
                            noisy_filepath = fullfile(output_dir, noisy_filename);
                            
                            % Save in WFDB-compatible format
                            val = noisy_signal;
                            save(noisy_filepath, 'val');
                            
                            total_noisy_generated = total_noisy_generated + 1;
                        end
                    end
                    
                    % Progress update
                    if mod(total_processed, 20) == 0
                        fprintf('    Processed %d files, generated %d noisy versions...\n', ...
                                total_processed, total_noisy_generated);
                    end
                    
                catch ME
                    fprintf('    Error processing %s: %s\n', mat_files(file_idx).name, ME.message);
                    processing_errors = processing_errors + 1;
                    continue;
                end
            end
            
            fprintf('  Completed %s group\n', group_name);
        end
        
        fprintf('Completed %s dataset\n\n', dataset_name);
    end
    
    % Final summary
    fprintf('=== NOISE GENERATION SUMMARY ===\n');
    fprintf('Original files processed: %d\n', total_processed);
    fprintf('Noisy versions generated: %d\n', total_noisy_generated);
    fprintf('Processing errors: %d\n', processing_errors);
    fprintf('Noise types: %d\n', length(noise_types));
    fprintf('SNR levels: %d\n', length(snr_levels));
    fprintf('Total combinations per file: %d\n', length(noise_types) * length(snr_levels));
    
    % Generate noise analysis and visualization
    generate_noise_analysis(noisy_dataset_path, analysis_path, fs);
    
    fprintf('\nNoisy dataset ready for scalogram conversion!\n');
    fprintf('Next step: Run convert_noisy_ecg_to_scalograms()\n');
end

function ecg_signals = extract_ecg_signals(ecg_data)
    % Extract ECG signals from WFDB .mat file format
    ecg_signals = [];
    
    % Try different common field names in WFDB format
    field_names = {'val', 'data', 'signal', 'ecg', 'y'};
    
    for i = 1:length(field_names)
        if isfield(ecg_data, field_names{i})
            signals = ecg_data.(field_names{i});
            
            % Ensure signals are in correct format (leads x samples)
            if size(signals, 1) > size(signals, 2)
                signals = signals'; % Transpose if needed
            end
            
            % Validate signal dimensions
            if size(signals, 1) >= 2 && size(signals, 2) > 100
                ecg_signals = signals;
                return;
            end
        end
    end
    
    % If no standard field found, try to find numerical arrays
    fields = fieldnames(ecg_data);
    for i = 1:length(fields)
        data = ecg_data.(fields{i});
        if isnumeric(data) && ndims(data) == 2
            if size(data, 1) > size(data, 2)
                data = data';
            end
            if size(data, 1) >= 2 && size(data, 2) > 100
                ecg_signals = data;
                return;
            end
        end
    end
end

function noisy_signal = add_portable_ecg_noise(clean_signal, fs, noise_type, snr_db)
    % Add realistic portable ECG device noise
    
    signal_length = length(clean_signal);
    t = (0:signal_length-1) / fs; % Time vector
    
    switch noise_type
        case 'gaussian'
            % White Gaussian noise (electronic noise from amplifiers)
            noise = randn(size(clean_signal));
            
        case 'powerline'
            % 50/60 Hz powerline interference (common in portable devices)
            freq_50hz = 2 * pi * 50; % 50 Hz
            freq_60hz = 2 * pi * 60; % 60 Hz
            
            % Primary interference
            noise = 0.6 * sin(freq_50hz * t) + 0.4 * sin(freq_60hz * t);
            
            % Harmonics (realistic powerline interference)
            noise = noise + 0.2 * sin(2 * freq_50hz * t) + 0.15 * sin(3 * freq_50hz * t);
            noise = noise + 0.1 * sin(2 * freq_60hz * t);
            
            % Add some randomness
            noise = noise + 0.3 * randn(size(clean_signal));
            
        case 'baseline_wander'
            % Low frequency baseline wander (motion artifacts, breathing)
            % Typical frequency: 0.05-2 Hz
            wander_freq1 = 2 * pi * 0.1; % 0.1 Hz (breathing-like)
            wander_freq2 = 2 * pi * 0.3; % 0.3 Hz (motion-like)
            wander_freq3 = 2 * pi * 0.05; % 0.05 Hz (very slow drift)
            
            noise = 2.0 * sin(wander_freq1 * t) + 1.5 * sin(wander_freq2 * t) + ...
                   3.0 * sin(wander_freq3 * t);
            
            % Add random walk component
            random_walk = cumsum(0.1 * randn(size(clean_signal))) / sqrt(signal_length);
            noise = noise + random_walk;
            
        case 'muscle_artifact'
            % EMG contamination (muscle activity)
            % Typical frequency: 20-200 Hz, burst-like
            
            % Generate burst patterns
            burst_probability = 0.3; % 30% chance of burst at any time
            burst_mask = rand(size(clean_signal)) < burst_probability;
            
            % High frequency muscle noise
            muscle_freq = 20 + 80 * rand(size(clean_signal)); % Variable frequency 20-100 Hz
            muscle_noise = sin(2 * pi * muscle_freq .* t);
            
            % Apply burst pattern
            noise = muscle_noise .* burst_mask;
            
            % Add random amplitude modulation
            amplitude_mod = 0.5 + 0.5 * rand(size(clean_signal));
            noise = noise .* amplitude_mod;
            
            % Add some white noise component
            noise = noise + 0.3 * randn(size(clean_signal));
            
        case 'motion_artifact'
            % High frequency motion artifacts from electrode movement
            % Sudden spikes and transient disturbances
            
            % Generate random spikes
            spike_probability = 0.02; % 2% chance of spike
            spike_locations = rand(size(clean_signal)) < spike_probability;
            
            % Create spikes with exponential decay
            noise = zeros(size(clean_signal));
            spike_indices = find(spike_locations);
            
            for i = 1:length(spike_indices)
                spike_idx = spike_indices(i);
                spike_amplitude = 2 + 3 * rand(); % Random amplitude
                decay_constant = 20 + 30 * rand(); % Random decay
                
                % Create decaying spike
                for j = spike_idx:min(spike_idx + decay_constant, signal_length)
                    decay_factor = exp(-(j - spike_idx) / (decay_constant / 3));
                    noise(j) = noise(j) + spike_amplitude * decay_factor * (rand() - 0.5);
                end
            end
            
            % Add high frequency component
            high_freq = 100 + 50 * rand(size(clean_signal)); % 100-150 Hz
            noise = noise + 0.5 * sin(2 * pi * high_freq .* t) .* (0.5 + 0.5 * rand(size(clean_signal)));
            
        case 'electrode_noise'
            % Contact impedance variations and electrode artifacts
            % Low frequency variations with occasional dropouts
            
            % Contact impedance variations (affects signal amplitude)
            impedance_freq = 2 * pi * 0.2; % 0.2 Hz variation
            impedance_variation = 0.1 * sin(impedance_freq * t) + 0.05 * randn(size(clean_signal));
            
            % Apply impedance effect (multiplicative)
            noisy_signal_temp = clean_signal .* (1 + impedance_variation);
            
            % Electrode dropouts (sudden signal loss)
            dropout_probability = 0.005; % 0.5% chance
            dropout_mask = rand(size(clean_signal)) < dropout_probability;
            
            % Create dropout effects
            for i = 1:length(dropout_mask)
                if dropout_mask(i)
                    dropout_length = 5 + randi(15); % 5-20 samples dropout
                    end_idx = min(i + dropout_length, length(clean_signal));
                    noisy_signal_temp(i:end_idx) = noisy_signal_temp(i:end_idx) * 0.1; % 90% signal loss
                end
            end
            
            noise = noisy_signal_temp - clean_signal;
            
        case 'combined'
            % Realistic combination of all noise types (most realistic scenario)
            
            % Weight factors for different noise types in portable devices
            gaussian_weight = 0.3;
            powerline_weight = 0.4;
            baseline_weight = 0.8;
            muscle_weight = 0.2;
            motion_weight = 0.1;
            electrode_weight = 0.3;
            
            % Generate each noise component
            noise_gaussian = gaussian_weight * randn(size(clean_signal));
            noise_powerline = powerline_weight * add_portable_ecg_noise(zeros(size(clean_signal)), fs, 'powerline', 0);
            noise_baseline = baseline_weight * add_portable_ecg_noise(zeros(size(clean_signal)), fs, 'baseline_wander', 0);
            noise_muscle = muscle_weight * add_portable_ecg_noise(zeros(size(clean_signal)), fs, 'muscle_artifact', 0);
            noise_motion = motion_weight * add_portable_ecg_noise(zeros(size(clean_signal)), fs, 'motion_artifact', 0);
            
            % For electrode noise, we need special handling since it's multiplicative
            temp_signal = clean_signal + noise_gaussian + noise_powerline + noise_baseline + noise_muscle + noise_motion;
            noisy_with_electrode = temp_signal + electrode_weight * add_portable_ecg_noise(temp_signal, fs, 'electrode_noise', 0);
            
            noise = noisy_with_electrode - clean_signal;
            
        otherwise
            error('Unknown noise type: %s', noise_type);
    end
    
    % Apply SNR constraint
    if snr_db ~= 0 % Only apply if SNR constraint is specified
        signal_power = mean(clean_signal.^2);
        noise_power = mean(noise.^2);
        
        % Calculate required noise scaling
        target_noise_power = signal_power / (10^(snr_db/10));
        noise_scaling = sqrt(target_noise_power / noise_power);
        
        noise = noise * noise_scaling;
    end
    
    % Combine signal and noise
    noisy_signal = clean_signal + noise;
    
    % Ensure signal stays within reasonable bounds (prevent clipping)
    signal_std = std(clean_signal);
    max_amplitude = 5 * signal_std;
    noisy_signal = max(-max_amplitude, min(max_amplitude, noisy_signal));
end

function generate_noise_analysis(noisy_dataset_path, analysis_path, fs)
    % Generate comprehensive noise analysis and visualization
    
    fprintf('\n=== GENERATING NOISE ANALYSIS ===\n');
    
    % Create sample noise visualizations
    create_noise_comparison_plots(analysis_path, fs);
    
    % Generate dataset statistics
    generate_noisy_dataset_statistics(noisy_dataset_path, analysis_path);
    
    % Create SNR analysis plots
    create_snr_analysis_plots(noisy_dataset_path, analysis_path);
    
    fprintf('Noise analysis completed. Results saved to: %s\n', analysis_path);
end

function create_noise_comparison_plots(analysis_path, fs)
    % Create visual comparison of different noise types
    
    % Generate a clean synthetic ECG signal for demonstration
    t = (0:1999) / fs; % 4 seconds
    clean_ecg = generate_synthetic_ecg(t);
    
    noise_types = {'gaussian', 'powerline', 'baseline_wander', 'muscle_artifact', 'motion_artifact', 'electrode_noise', 'combined'};
    snr_db = 10; % Fixed SNR for comparison
    
    % Create comprehensive noise comparison figure
    fig = figure('Position', [100, 100, 1600, 1200]);
    
    for i = 1:length(noise_types)
        subplot(4, 2, i);
        
        % Generate noisy signal
        noisy_signal = add_portable_ecg_noise(clean_ecg, fs, noise_types{i}, snr_db);
        
        % Plot comparison
        plot(t, clean_ecg, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Clean ECG');
        hold on;
        plot(t, noisy_signal, 'r-', 'LineWidth', 1, 'DisplayName', sprintf('Noisy (SNR=%ddB)', snr_db));
        
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(sprintf('%s Noise', strrep(noise_types{i}, '_', ' ')), 'FontWeight', 'bold');
        legend('Location', 'best');
        grid on; grid minor;
        xlim([0, 4]);
        hold off;
    end
    
    sgtitle('Portable ECG Device Noise Types Comparison', 'FontWeight', 'bold', 'FontSize', 16);
    
    % Save figure
    saveas(fig, fullfile(analysis_path, 'Noise_Types_Comparison.png'), 'png');
    saveas(fig, fullfile(analysis_path, 'Noise_Types_Comparison.fig'), 'fig');
    
    % Create frequency domain analysis
    create_frequency_domain_analysis(clean_ecg, noise_types, analysis_path, fs);
end

function clean_ecg = generate_synthetic_ecg(t)
    % Generate a synthetic ECG signal for demonstration
    
    % Basic ECG components (simplified)
    heart_rate = 75; % BPM
    rr_interval = 60 / heart_rate; % seconds
    
    ecg_signal = zeros(size(t));
    
    % Generate QRS complexes
    for beat_time = 0:rr_interval:max(t)
        % Find time indices for this beat
        beat_indices = find(t >= beat_time & t <= beat_time + 0.2);
        
        if ~isempty(beat_indices)
            beat_t = t(beat_indices) - beat_time;
            
            % Simplified QRS complex
            qrs = 2 * exp(-((beat_t - 0.05) / 0.02).^2) - 0.5 * exp(-((beat_t - 0.03) / 0.01).^2) + ...
                  -0.3 * exp(-((beat_t - 0.07) / 0.01).^2);
            
            ecg_signal(beat_indices) = ecg_signal(beat_indices) + qrs;
        end
    end
    
    % Add some baseline and T-wave components
    baseline_freq = 2 * pi * 1.2; % 1.2 Hz
    ecg_signal = ecg_signal + 0.2 * sin(baseline_freq * t);
    
    clean_ecg = ecg_signal;
end

function create_frequency_domain_analysis(clean_ecg, noise_types, analysis_path, fs)
    % Create frequency domain analysis of noise types
    
    fig = figure('Position', [100, 100, 1400, 900]);
    
    % Calculate FFT of clean signal
    N = length(clean_ecg);
    frequencies = (0:N-1) * fs / N;
    clean_fft = abs(fft(clean_ecg));
    
    subplot(2, 1, 1);
    semilogy(frequencies(1:N/2), clean_fft(1:N/2), 'b-', 'LineWidth', 2, 'DisplayName', 'Clean ECG');
    hold on;
    
    colors = lines(length(noise_types));
    
    for i = 1:length(noise_types)
        noisy_signal = add_portable_ecg_noise(clean_ecg, fs, noise_types{i}, 10);
        noisy_fft = abs(fft(noisy_signal));
        
        semilogy(frequencies(1:N/2), noisy_fft(1:N/2), 'Color', colors(i, :), ...
                'LineWidth', 1.5, 'DisplayName', strrep(noise_types{i}, '_', ' '));
    end
    
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    title('Frequency Domain Analysis of Noise Types', 'FontWeight', 'bold', 'FontSize', 14);
    legend('Location', 'best');
    grid on; grid minor;
    xlim([0, 250]);
    hold off;
    
    % Power spectral density comparison
    subplot(2, 1, 2);
    
    [clean_psd, f] = pwelch(clean_ecg, [], [], [], fs);
    semilogy(f, clean_psd, 'b-', 'LineWidth', 2, 'DisplayName', 'Clean ECG');
    hold on;
    
    for i = 1:length(noise_types)
        noisy_signal = add_portable_ecg_noise(clean_ecg, fs, noise_types{i}, 10);
        [noisy_psd, ~] = pwelch(noisy_signal, [], [], [], fs);
        
        semilogy(f, noisy_psd, 'Color', colors(i, :), 'LineWidth', 1.5, ...
                'DisplayName', strrep(noise_types{i}, '_', ' '));
    end
    
    xlabel('Frequency (Hz)');
    ylabel('Power Spectral Density');
    title('Power Spectral Density Comparison', 'FontWeight', 'bold', 'FontSize', 14);
    legend('Location', 'best');
    grid on; grid minor;
    xlim([0, 100]);
    hold off;
    
    sgtitle('Frequency Domain Analysis of ECG Noise Types', 'FontWeight', 'bold', 'FontSize', 16);
    
    saveas(fig, fullfile(analysis_path, 'Frequency_Domain_Analysis.png'), 'png');
    saveas(fig, fullfile(analysis_path, 'Frequency_Domain_Analysis.fig'), 'fig');
end

function generate_noisy_dataset_statistics(noisy_dataset_path, analysis_path)
    % Generate statistics about the noisy dataset
    
    fprintf('Generating dataset statistics...\n');
    
    % Count files in each category
    snr_dirs = dir(fullfile(noisy_dataset_path, 'SNR_*'));
    snr_dirs = snr_dirs([snr_dirs.isdir]);
    
    if isempty(snr_dirs)
        fprintf('No SNR directories found. Statistics generation skipped.\n');
        return;
    end
    
    % Create statistics report
    stats_file = fullfile(analysis_path, 'Noisy_Dataset_Statistics.txt');
    fid = fopen(stats_file, 'w');
    
    fprintf(fid, '=== NOISY ECG DATASET STATISTICS ===\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now));
    
    fprintf(fid, 'DATASET STRUCTURE:\n');
    fprintf(fid, 'Base Path: %s\n', noisy_dataset_path);
    fprintf(fid, 'SNR Levels: %d\n', length(snr_dirs));
    fprintf(fid, 'Directory Structure: SNR_XXdB/[noise_type]/[training|validation]/[AFIB|SB|SR]/\n\n');
    
    total_files = 0;
    
    for i = 1:length(snr_dirs)
        snr_name = snr_dirs(i).name;
        fprintf(fid, '%s:\n', snr_name);
        
        noise_dirs = dir(fullfile(noisy_dataset_path, snr_name, '*'));
        noise_dirs = noise_dirs([noise_dirs.isdir] & ~strcmp({noise_dirs.name}, '.') & ~strcmp({noise_dirs.name}, '..'));
        
        for j = 1:length(noise_dirs)
            noise_type = noise_dirs(j).name;
            
            % Count files for this noise type and SNR
            training_count = count_files_recursive(fullfile(noisy_dataset_path, snr_name, noise_type, 'training'));
            validation_count = count_files_recursive(fullfile(noisy_dataset_path, snr_name, noise_type, 'validation'));
            
            fprintf(fid, '  %s: Training=%d, Validation=%d, Total=%d\n', ...
                   noise_type, training_count, validation_count, training_count + validation_count);
            
            total_files = total_files + training_count + validation_count;
        end
        fprintf(fid, '\n');
    end
    
    fprintf(fid, 'TOTAL NOISY FILES GENERATED: %d\n', total_files);
    
    fclose(fid);
    
    fprintf('Dataset statistics saved to: %s\n', stats_file);
end

function count = count_files_recursive(dir_path)
    % Recursively count .mat files in directory
    count = 0;
    if exist(dir_path, 'dir')
        files = dir(fullfile(dir_path, '**', '*.mat'));
        count = length(files);
    end
end

function create_snr_analysis_plots(noisy_dataset_path, analysis_path)
    % Create SNR analysis plots (placeholder for when data is available)
    
    fprintf('SNR analysis plots will be generated after noise addition is complete.\n');
    
    % Create a placeholder figure showing expected SNR effects
    fig = figure('Position', [100, 100, 1200, 800]);
    
    % Theoretical SNR effects
    snr_values = [30, 20, 15, 10, 5, 0];
    expected_accuracy = [0.95, 0.92, 0.88, 0.82, 0.75, 0.65]; % Hypothetical values
    
    subplot(2, 2, 1);
    plot(snr_values, expected_accuracy, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('SNR (dB)');
    ylabel('Expected Model Accuracy');
    title('Expected Model Performance vs SNR', 'FontWeight', 'bold');
    grid on; grid minor;
    
    subplot(2, 2, 2);
    bar(snr_values, expected_accuracy);
    xlabel('SNR (dB)');
    ylabel('Expected Accuracy');
    title('Performance Degradation with Noise', 'FontWeight', 'bold');
    grid on; grid minor;
    
    subplot(2, 2, [3, 4]);
    axis off;
    
    info_text = {
        '\bf{SNR Analysis Information:}'
        ''
        '• SNR Levels: 30, 20, 15, 10, 5, 0 dB'
        '• Higher SNR = Less noise, better signal quality'
        '• Lower SNR = More noise, degraded signal quality'
        ''
        '\bf{Expected Effects:}'
        '• SNR 30 dB: Minimal noise, near-optimal performance'
        '• SNR 20 dB: Light noise, slight performance drop'
        '• SNR 15 dB: Moderate noise, noticeable degradation'
        '• SNR 10 dB: Heavy noise, significant impact'
        '• SNR 5 dB: Very heavy noise, major degradation'
        '• SNR 0 dB: Extreme noise, severe performance loss'
        ''
        '\bf{Research Applications:}'
        '• Model robustness testing'
        '• Real-world deployment preparation'
        '• Noise tolerance assessment'
        '• Algorithm comparison under adverse conditions'
    };
    
    text(0.05, 0.95, info_text, 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'FontName', 'Arial', ...
         'FontSize', 11, 'Interpreter', 'tex');
    
    sgtitle('SNR Analysis for Model Robustness Testing', 'FontWeight', 'bold', 'FontSize', 16);
    
    saveas(fig, fullfile(analysis_path, 'SNR_Analysis_Overview.png'), 'png');
    saveas(fig, fullfile(analysis_path, 'SNR_Analysis_Overview.fig'), 'fig');
end

% Main execution
fprintf('Starting ECG Noise Generation System...\n');
ecg_noise_generator_system();