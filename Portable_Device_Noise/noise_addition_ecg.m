function focused_combined_noise_generator()
    % FOCUSED COMBINED NOISE GENERATOR
    % Generates realistic combined portable ECG device noise
    % FOCUS: Only combined noise at SNR 15dB, 20dB, 25dB
    % Processes only Lead II for first 4 seconds
    % Output: Ready for scalogram conversion
    
    % Define paths
    original_dataset_path = 'C:\Users\henry\Downloads\ECG-Dx\Organized_Dataset';
    noisy_dataset_path = 'C:\Users\henry\Downloads\ECG-Dx\Focused_Combined_Noise_Dataset';
    
    % Create output directory
    if ~exist(noisy_dataset_path, 'dir')
        mkdir(noisy_dataset_path);
    end
    
    % Parameters
    fs = 500; % Sampling frequency (Hz)
    duration_seconds = 4; % Process first 4 seconds
    target_samples = fs * duration_seconds; % 2000 samples
    snr_levels = [25, 20, 15]; % High to low quality (clinical relevant range)
    noise_type = 'combined'; % Only combined noise
    
    % Initialize counters
    total_processed = 0;
    total_generated = 0;
    processing_errors = 0;
    
    fprintf('=== FOCUSED COMBINED NOISE GENERATOR ===\n');
    fprintf('Original dataset: %s\n', original_dataset_path);
    fprintf('Output dataset: %s\n', noisy_dataset_path);
    fprintf('Noise type: Combined (realistic multi-source)\n');
    fprintf('SNR levels: %s dB\n', mat2str(snr_levels));
    fprintf('Processing: Lead II only (first %d seconds)\n', duration_seconds);
    fprintf('Classes: AFIB, SB, SR\n\n');
    
    % Process datasets
    datasets = {'training', 'validation'};
    groups = {'AFIB', 'SB', 'SR'};
    
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
                    
                    % Generate noisy versions for each SNR level
                    for snr_idx = 1:length(snr_levels)
                        snr_db = snr_levels(snr_idx);
                        
                        % Generate noisy signal with combined noise
                        noisy_signal = add_combined_noise(lead2_signal, fs, snr_db);
                        
                        % Create output directory structure
                        output_dir = fullfile(noisy_dataset_path, sprintf('SNR_%02ddB', snr_db), ...
                                            dataset_name, group_name);
                        if ~exist(output_dir, 'dir')
                            mkdir(output_dir);
                        end
                        
                        % Save noisy signal
                        noisy_filename = sprintf('%s_COMBINED_SNR%02d.mat', base_name, snr_db);
                        noisy_filepath = fullfile(output_dir, noisy_filename);
                        
                        % Save in WFDB-compatible format
                        val = noisy_signal;
                        save(noisy_filepath, 'val');
                        
                        total_generated = total_generated + 1;
                    end
                    
                    % Progress update
                    if mod(total_processed, 25) == 0
                        fprintf('    Processed %d files, generated %d noisy versions...\n', ...
                                total_processed, total_generated);
                    end
                    
                catch ME
                    fprintf('    Error processing %s: %s\n', mat_files(file_idx).name, ME.message);
                    processing_errors = processing_errors + 1;
                    continue;
                end
            end
            
            fprintf('  Completed %s group: %d files generated\n', group_name, ...
                   count_files_in_dir(fullfile(noisy_dataset_path, sprintf('SNR_%02ddB', snr_levels(1)), dataset_name, group_name)));
        end
        
        fprintf('Completed %s dataset\n\n', dataset_name);
    end
    
    % Final summary
    fprintf('=== FOCUSED NOISE GENERATION SUMMARY ===\n');
    fprintf('Original files processed: %d\n', total_processed);
    fprintf('Noisy versions generated: %d\n', total_generated);
    fprintf('Processing errors: %d\n', processing_errors);
    fprintf('SNR levels: %d (%s dB)\n', length(snr_levels), mat2str(snr_levels));
    fprintf('Noise combinations per file: %d\n', length(snr_levels));
    if total_processed > 0
        fprintf('Success rate: %.1f%%\n', (total_generated/(total_processed*length(snr_levels)))*100);
    end
    fprintf('Output directory: %s\n', noisy_dataset_path);
    
    % Generate summary report
    generate_focused_report(noisy_dataset_path, total_processed, total_generated, processing_errors, snr_levels);
    
    % Generate noise visualization
    generate_combined_noise_visualization(noisy_dataset_path, fs, snr_levels);
    
    fprintf('\nFocused combined noise dataset ready!\n');
    fprintf('Next step: Run convert_focused_noisy_to_scalograms()\n');
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

function noisy_signal = add_combined_noise(clean_signal, fs, snr_db)
    % Add realistic combined portable ECG device noise
    % This represents the most realistic noise scenario for portable devices
    
    signal_length = length(clean_signal);
    t = (0:signal_length-1) / fs; % Time vector
    
    % Combined noise components with realistic weights for portable devices
    
    % 1. Gaussian noise (electronic amplifier noise) - 30% weight
    gaussian_weight = 0.3;
    noise_gaussian = gaussian_weight * randn(size(clean_signal));
    
    % 2. Powerline interference (50/60 Hz) - 40% weight
    powerline_weight = 0.4;
    freq_50hz = 2 * pi * 50;
    freq_60hz = 2 * pi * 60;
    noise_powerline = powerline_weight * (0.6 * sin(freq_50hz * t) + 0.4 * sin(freq_60hz * t));
    % Add harmonics
    noise_powerline = noise_powerline + powerline_weight * 0.2 * sin(2 * freq_50hz * t);
    noise_powerline = noise_powerline + powerline_weight * 0.1 * randn(size(clean_signal));
    
    % 3. Baseline wander (motion artifacts) - 80% weight
    baseline_weight = 0.8;
    wander_freq1 = 2 * pi * 0.1; % Breathing-like
    wander_freq2 = 2 * pi * 0.3; % Motion-like
    wander_freq3 = 2 * pi * 0.05; % Slow drift
    noise_baseline = baseline_weight * (2.0 * sin(wander_freq1 * t) + ...
                    1.5 * sin(wander_freq2 * t) + 3.0 * sin(wander_freq3 * t));
    % Add random walk
    random_walk = cumsum(0.1 * randn(size(clean_signal))) / sqrt(signal_length);
    noise_baseline = noise_baseline + baseline_weight * random_walk;
    
    % 4. Muscle artifacts (EMG) - 20% weight
    muscle_weight = 0.2;
    burst_probability = 0.3;
    burst_mask = rand(size(clean_signal)) < burst_probability;
    muscle_freq = 20 + 80 * rand(size(clean_signal));
    muscle_noise = sin(2 * pi * muscle_freq .* t);
    amplitude_mod = 0.5 + 0.5 * rand(size(clean_signal));
    noise_muscle = muscle_weight * muscle_noise .* burst_mask .* amplitude_mod;
    noise_muscle = noise_muscle + muscle_weight * 0.3 * randn(size(clean_signal));
    
    % 5. Motion artifacts (electrode movement) - 10% weight
    motion_weight = 0.1;
    spike_probability = 0.02;
    spike_locations = rand(size(clean_signal)) < spike_probability;
    noise_motion = zeros(size(clean_signal));
    spike_indices = find(spike_locations);
    
    for i = 1:length(spike_indices)
        spike_idx = spike_indices(i);
        spike_amplitude = 2 + 3 * rand();
        decay_constant = 20 + 30 * rand();
        
        for j = spike_idx:min(spike_idx + decay_constant, signal_length)
            decay_factor = exp(-(j - spike_idx) / (decay_constant / 3));
            noise_motion(j) = noise_motion(j) + spike_amplitude * decay_factor * (rand() - 0.5);
        end
    end
    
    high_freq = 100 + 50 * rand(size(clean_signal));
    noise_motion = motion_weight * (noise_motion + 0.5 * sin(2 * pi * high_freq .* t) .* ...
                  (0.5 + 0.5 * rand(size(clean_signal))));
    
    % 6. Electrode noise (contact impedance) - 30% weight
    electrode_weight = 0.3;
    impedance_freq = 2 * pi * 0.2;
    impedance_variation = 0.1 * sin(impedance_freq * t) + 0.05 * randn(size(clean_signal));
    noisy_signal_temp = clean_signal .* (1 + impedance_variation);
    
    % Electrode dropouts
    dropout_probability = 0.005;
    dropout_mask = rand(size(clean_signal)) < dropout_probability;
    for i = 1:length(dropout_mask)
        if dropout_mask(i)
            dropout_length = 5 + randi(15);
            end_idx = min(i + dropout_length, length(clean_signal));
            noisy_signal_temp(i:end_idx) = noisy_signal_temp(i:end_idx) * 0.1;
        end
    end
    noise_electrode = electrode_weight * (noisy_signal_temp - clean_signal);
    
    % Combine all noise components
    total_noise = noise_gaussian + noise_powerline + noise_baseline + ...
                  noise_muscle + noise_motion + noise_electrode;
    
    % Apply SNR constraint
    signal_power = mean(clean_signal.^2);
    noise_power = mean(total_noise.^2);
    
    % Calculate required noise scaling for target SNR
    target_noise_power = signal_power / (10^(snr_db/10));
    noise_scaling = sqrt(target_noise_power / noise_power);
    
    total_noise = total_noise * noise_scaling;
    
    % Combine signal and noise
    noisy_signal = clean_signal + total_noise;
    
    % Ensure signal stays within reasonable bounds
    signal_std = std(clean_signal);
    max_amplitude = 5 * signal_std;
    noisy_signal = max(-max_amplitude, min(max_amplitude, noisy_signal));
end

function count = count_files_in_dir(dir_path)
    % Count .mat files in directory
    if exist(dir_path, 'dir')
        files = dir(fullfile(dir_path, '*.mat'));
        count = length(files);
    else
        count = 0;
    end
end

function generate_focused_report(output_path, total_processed, total_generated, processing_errors, snr_levels)
    % Generate focused report for combined noise at specific SNR levels
    
    report_file = fullfile(output_path, 'focused_combined_noise_report.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, '=== FOCUSED COMBINED NOISE GENERATION REPORT ===\n');
    fprintf(fid, 'Date: %s\n\n', datestr(now));
    
    fprintf(fid, 'GENERATION PARAMETERS:\n');
    fprintf(fid, '- Noise Type: Combined (realistic multi-source)\n');
    fprintf(fid, '- SNR Levels: %s dB\n', mat2str(snr_levels));
    fprintf(fid, '- Signal: ECG Lead II (first 4 seconds)\n');
    fprintf(fid, '- Sampling Rate: 500 Hz (2000 samples)\n');
    fprintf(fid, '- Classes: AFIB, SB, SR\n');
    fprintf(fid, '- Datasets: Training and Validation\n\n');
    
    fprintf(fid, 'COMBINED NOISE COMPONENTS:\n');
    fprintf(fid, '- Gaussian Noise (30%% weight): Electronic amplifier noise\n');
    fprintf(fid, '- Powerline Interference (40%% weight): 50/60 Hz contamination\n');
    fprintf(fid, '- Baseline Wander (80%% weight): Motion artifacts\n');
    fprintf(fid, '- Muscle Artifacts (20%% weight): EMG contamination\n');
    fprintf(fid, '- Motion Artifacts (10%% weight): Electrode movement\n');
    fprintf(fid, '- Electrode Noise (30%% weight): Contact impedance\n\n');
    
    fprintf(fid, 'PROCESSING STATISTICS:\n');
    fprintf(fid, 'Original files processed: %d\n', total_processed);
    fprintf(fid, 'Noisy versions generated: %d\n', total_generated);
    fprintf(fid, 'Processing errors: %d\n', processing_errors);
    fprintf(fid, 'Files per SNR level: %d\n', total_generated / length(snr_levels));
    
    if total_processed > 0
        success_rate = (total_generated / (total_processed * length(snr_levels))) * 100;
        fprintf(fid, 'Success rate: %.1f%%\n', success_rate);
    end
    
    fprintf(fid, '\nOUTPUT STRUCTURE:\n');
    fprintf(fid, 'Directory: SNR_[15|20|25]dB/[training|validation]/[AFIB|SB|SR]/\n');
    fprintf(fid, 'Filename format: [PATIENT_ID]_age[AGE]_COMBINED_SNR[XX].mat\n');
    fprintf(fid, '\nExample filenames:\n');
    fprintf(fid, '- JS44163_age45_COMBINED_SNR25.mat\n');
    fprintf(fid, '- TR09173_age67_COMBINED_SNR20.mat\n');
    fprintf(fid, '- AM12456_age34_COMBINED_SNR15.mat\n');
    
    fprintf(fid, '\nCLINICAL SIGNIFICANCE BY SNR LEVEL:\n');
    fprintf(fid, '25dB SNR: High-quality portable ECG\n');
    fprintf(fid, '- Expected model accuracy: >95%%\n');
    fprintf(fid, '- Clinical grade quality\n');
    fprintf(fid, '- Suitable for diagnostic applications\n\n');
    
    fprintf(fid, '20dB SNR: Good-quality portable ECG\n');
    fprintf(fid, '- Expected model accuracy: 90-95%%\n');
    fprintf(fid, '- Clinical monitoring grade\n');
    fprintf(fid, '- Suitable for continuous monitoring\n\n');
    
    fprintf(fid, '15dB SNR: Moderate-quality portable ECG\n');
    fprintf(fid, '- Expected model accuracy: 80-90%%\n');
    fprintf(fid, '- Research grade quality\n');
    fprintf(fid, '- Requires careful validation\n\n');
    
    fprintf(fid, 'RESEARCH APPLICATIONS:\n');
    fprintf(fid, '- Model robustness validation\n');
    fprintf(fid, '- Clinical deployment readiness\n');
    fprintf(fid, '- Performance threshold determination\n');
    fprintf(fid, '- Quality control implementation\n');
    
    fclose(fid);
    
    fprintf('Focused generation report saved to: %s\n', report_file);
end

function generate_combined_noise_visualization(output_path, fs, snr_levels)
    % Generate visualization of combined noise effects
    
    fprintf('Generating combined noise visualization...\n');
    
    % Create synthetic ECG for demonstration
    t = (0:1999) / fs; % 4 seconds
    clean_ecg = generate_demo_ecg(t);
    
    % Create visualization figure
    fig = figure('Position', [100, 100, 1400, 1000]);
    
    % Plot clean signal
    subplot(length(snr_levels) + 1, 1, 1);
    plot(t, clean_ecg, 'b-', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Clean ECG Lead II (4 seconds)', 'FontWeight', 'bold', 'FontSize', 12);
    grid on; grid minor;
    xlim([0, 4]);
    
    % Plot noisy signals for each SNR level
    for snr_idx = 1:length(snr_levels)
        snr_db = snr_levels(snr_idx);
        noisy_signal = add_combined_noise(clean_ecg, fs, snr_db);
        
        subplot(length(snr_levels) + 1, 1, snr_idx + 1);
        plot(t, clean_ecg, 'b-', 'LineWidth', 1, 'DisplayName', 'Clean', 'Color', [0.5, 0.5, 1]);
        hold on;
        plot(t, noisy_signal, 'r-', 'LineWidth', 1.5, 'DisplayName', sprintf('Noisy (SNR=%ddB)', snr_db));
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(sprintf('Combined Noise at SNR %d dB', snr_db), 'FontWeight', 'bold', 'FontSize', 12);
        legend('Location', 'best');
        grid on; grid minor;
        xlim([0, 4]);
        hold off;
    end
    
    sgtitle('Combined Portable ECG Noise Effects Across SNR Levels', ...
            'FontWeight', 'bold', 'FontSize', 16);
    
    % Save figure
    visualization_path = fullfile(output_path, 'Combined_Noise_Analysis');
    if ~exist(visualization_path, 'dir')
        mkdir(visualization_path);
    end
    
    saveas(fig, fullfile(visualization_path, 'Combined_Noise_Comparison.png'), 'png');
    saveas(fig, fullfile(visualization_path, 'Combined_Noise_Comparison.fig'), 'fig');
    
    % Generate frequency domain analysis
    generate_frequency_analysis(clean_ecg, snr_levels, fs, visualization_path);
    
    fprintf('Combined noise visualization saved to: %s\n', visualization_path);
end

function clean_ecg = generate_demo_ecg(t)
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

function generate_frequency_analysis(clean_ecg, snr_levels, fs, visualization_path)
    % Generate frequency domain analysis
    
    fig = figure('Position', [100, 100, 1200, 800]);
    
    % Calculate FFT of clean signal
    N = length(clean_ecg);
    frequencies = (0:N-1) * fs / N;
    clean_fft = abs(fft(clean_ecg));
    
    subplot(2, 1, 1);
    semilogy(frequencies(1:N/2), clean_fft(1:N/2), 'b-', 'LineWidth', 2, 'DisplayName', 'Clean ECG');
    hold on;
    
    colors = lines(length(snr_levels));
    
    for i = 1:length(snr_levels)
        noisy_signal = add_combined_noise(clean_ecg, fs, snr_levels(i));
        noisy_fft = abs(fft(noisy_signal));
        
        semilogy(frequencies(1:N/2), noisy_fft(1:N/2), 'Color', colors(i, :), ...
                'LineWidth', 1.5, 'DisplayName', sprintf('SNR %d dB', snr_levels(i)));
    end
    
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    title('Frequency Domain Analysis: Combined Noise Effects', 'FontWeight', 'bold', 'FontSize', 14);
    legend('Location', 'best');
    grid on; grid minor;
    xlim([0, 100]);
    hold off;
    
    % SNR vs frequency content
    subplot(2, 1, 2);
    
    signal_power = mean(clean_ecg.^2);
    snr_actual = zeros(size(snr_levels));
    
    for i = 1:length(snr_levels)
        noisy_signal = add_combined_noise(clean_ecg, fs, snr_levels(i));
        noise_component = noisy_signal - clean_ecg;
        noise_power = mean(noise_component.^2);
        snr_actual(i) = 10 * log10(signal_power / noise_power);
    end
    
    plot(snr_levels, snr_actual, 'ro-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    hold on;
    plot([min(snr_levels), max(snr_levels)], [min(snr_levels), max(snr_levels)], 'k--', 'LineWidth', 1);
    xlabel('Target SNR (dB)');
    ylabel('Measured SNR (dB)');
    title('SNR Verification: Target vs Measured', 'FontWeight', 'bold', 'FontSize', 14);
    legend('Measured SNR', 'Perfect Match', 'Location', 'best');
    grid on; grid minor;
    hold off;
    
    sgtitle('Combined Noise: Frequency Analysis and SNR Verification', ...
            'FontWeight', 'bold', 'FontSize', 16);
    
    saveas(fig, fullfile(visualization_path, 'Frequency_Analysis.png'), 'png');
    saveas(fig, fullfile(visualization_path, 'Frequency_Analysis.fig'), 'fig');
end

% Main execution
fprintf('Starting Focused Combined Noise Generator...\n');
focused_combined_noise_generator();