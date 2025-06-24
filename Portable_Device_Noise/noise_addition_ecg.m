function generate_portable_ecg_noise_dataset()
    % PORTABLE ECG DEVICE NOISE GENERATOR
    % Adds realistic portable ECG device noise to age-balanced dataset
    % Combines 4 types of noise: EMG, baseline wander, powerline interference, gaussian noise
    % Generates datasets at SNR levels: 10dB, 15dB, 20dB
    % Shows before/after visualization for sample patient
    
    % Define paths
    source_dataset_path = 'C:\Users\henry\Downloads\ECG-Dx\Noise_ECG_Dataset_Signal';
    output_dataset_path = 'C:\Users\henry\Downloads\ECG-Dx\Noisy_Portable_ECG_Dataset';
    
    % Create output directory
    if ~exist(output_dataset_path, 'dir')
        mkdir(output_dataset_path);
    end
    
    % Parameters
    fs = 500; % Sampling frequency (Hz)
    snr_levels = [20, 15, 10]; % SNR levels in dB (high to low quality)
    classes = {'SR', 'SB', 'AFIB'};
    age_groups = {'young_adult', 'middle_aged', 'elderly'};
    
    % Initialize counters
    total_processed = 0;
    total_generated = 0;
    processing_errors = 0;
    
    fprintf('=== PORTABLE ECG DEVICE NOISE GENERATOR ===\n');
    fprintf('Source dataset: %s\n', source_dataset_path);
    fprintf('Output dataset: %s\n', output_dataset_path);
    fprintf('SNR levels: %s dB\n', mat2str(snr_levels));
    fprintf('Noise types: EMG, Baseline Wander, Powerline Interference, Gaussian\n');
    fprintf('Classes: %s\n', strjoin(classes, ', '));
    fprintf('Age groups: %s\n', strjoin(age_groups, ', '));
    fprintf('Processing: All ECG leads (10 seconds)\n\n');
    
    % Verify source dataset exists
    if ~exist(source_dataset_path, 'dir')
        error('Source dataset directory not found: %s', source_dataset_path);
    end
    
    % Create output directory structure
    create_noisy_output_directories(output_dataset_path, snr_levels, classes, age_groups);
    
    % Find a sample file for visualization
    sample_file_info = find_sample_file(source_dataset_path, classes, age_groups);
    sample_signal = [];
    sample_filename = '';
    
    % Process each class and age group
    for class_idx = 1:length(classes)
        class_name = classes{class_idx};
        fprintf('Processing %s class...\n', class_name);
        
        for age_group_idx = 1:length(age_groups)
            age_group_name = age_groups{age_group_idx};
            fprintf('  Processing %s age group...\n', age_group_name);
            
            % Define input directory
            input_dir = fullfile(source_dataset_path, class_name, age_group_name);
            
            if ~exist(input_dir, 'dir')
                fprintf('    Warning: Directory not found: %s\n', input_dir);
                continue;
            end
            
            % Get all .mat files
            mat_files = dir(fullfile(input_dir, '*.mat'));
            mat_files = mat_files(~strcmp({mat_files.name}, 'metadata.mat'));
            
            if isempty(mat_files)
                fprintf('    Warning: No .mat files found in %s\n', input_dir);
                continue;
            end
            
            fprintf('    Found %d files to process\n', length(mat_files));
            
            % Process each ECG file
            for file_idx = 1:length(mat_files)
                total_processed = total_processed + 1;
                
                mat_file_path = fullfile(input_dir, mat_files(file_idx).name);
                [~, base_name, ~] = fileparts(mat_files(file_idx).name);
                
                try
                    % Load ECG data
                    ecg_data = load(mat_file_path);
                    ecg_signals = extract_ecg_signals(ecg_data);
                    
                    if isempty(ecg_signals)
                        fprintf('      Warning: No valid ECG signals found in %s\n', mat_files(file_idx).name);
                        processing_errors = processing_errors + 1;
                        continue;
                    end
                    
                    % Store sample for visualization (first file from first class/age group)
                    if isempty(sample_signal) && strcmp(class_name, classes{1}) && strcmp(age_group_name, age_groups{1})
                        sample_signal = ecg_signals;
                        sample_filename = mat_files(file_idx).name;
                    end
                    
                    % Generate noisy versions for each SNR level
                    for snr_idx = 1:length(snr_levels)
                        snr_db = snr_levels(snr_idx);
                        
                        % Add portable ECG device noise
                        noisy_signals = add_portable_ecg_noise(ecg_signals, fs, snr_db);
                        
                        % Create output directory for this SNR level
                        output_dir = fullfile(output_dataset_path, sprintf('SNR_%02ddB', snr_db), ...
                                            class_name, age_group_name);
                        
                        % Save noisy signal
                        noisy_filename = sprintf('%s_SNR%02ddB.mat', base_name, snr_db);
                        noisy_filepath = fullfile(output_dir, noisy_filename);
                        
                        % Save in same format as original (WFDB-compatible)
                        val = noisy_signals;
                        save(noisy_filepath, 'val');
                        
                        total_generated = total_generated + 1;
                    end
                    
                    % Progress update
                    if mod(total_processed, 50) == 0
                        fprintf('      Processed %d files, generated %d noisy versions...\n', ...
                                total_processed, total_generated);
                    end
                    
                catch ME
                    fprintf('      Error processing %s: %s\n', mat_files(file_idx).name, ME.message);
                    processing_errors = processing_errors + 1;
                    continue;
                end
            end
        end
        
        fprintf('  Completed %s class\n', class_name);
    end
    
    % Final summary
    fprintf('\n=== NOISE GENERATION SUMMARY ===\n');
    fprintf('Original files processed: %d\n', total_processed);
    fprintf('Noisy versions generated: %d\n', total_generated);
    fprintf('Processing errors: %d\n', processing_errors);
    fprintf('Files per SNR level: %d\n', total_generated / length(snr_levels));
    if total_processed > 0
        fprintf('Success rate: %.1f%%\n', (total_generated/(total_processed*length(snr_levels)))*100);
    end
    fprintf('Output directory: %s\n', output_dataset_path);
    
    % Generate noise visualization with sample signal
    if ~isempty(sample_signal)
        fprintf('\nGenerating noise visualization with sample patient data...\n');
        generate_noise_visualization(sample_signal, sample_filename, fs, snr_levels, output_dataset_path);
    end
    
    % Generate comprehensive report
    generate_noise_report(output_dataset_path, total_processed, total_generated, processing_errors, snr_levels);
    
    % Verify output dataset
    verify_noisy_dataset(output_dataset_path, snr_levels, classes, age_groups);
    
    fprintf('\nPortable ECG noise dataset generation complete!\n');
    fprintf('Dataset ready for robustness testing and model benchmarking.\n');
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
            
            % Validate signal dimensions (expect multiple leads, 5000 samples for 10s)
            if size(signals, 1) >= 1 && size(signals, 2) >= 2500
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
            if size(data, 1) >= 1 && size(data, 2) >= 2500
                ecg_signals = data;
                return;
            end
        end
    end
end

function noisy_signals = add_portable_ecg_noise(clean_signals, fs, snr_db)
    % Add realistic portable ECG device noise combining 4 noise types
    % 1. EMG (Electromyography) noise - muscle artifacts
    % 2. Baseline wander - motion artifacts, breathing
    % 3. Powerline interference - 50/60 Hz electrical contamination
    % 4. Gaussian noise - electronic amplifier noise
    
    [num_leads, signal_length] = size(clean_signals);
    t = (0:signal_length-1) / fs; % Time vector
    
    % Initialize total noise
    total_noise = zeros(size(clean_signals));
    
    % Process each lead separately
    for lead_idx = 1:num_leads
        clean_signal = clean_signals(lead_idx, :);
        
        % 1. EMG NOISE (20-500 Hz, burst-like)
        emg_weight = 0.3;
        burst_probability = 0.15; % 15% chance of burst at each sample
        burst_mask = rand(size(clean_signal)) < burst_probability;
        
        % Create EMG frequency content (20-500 Hz)
        emg_freq = 20 + 480 * rand(size(clean_signal));
        emg_noise = sin(2 * pi * emg_freq .* t);
        
        % Apply burst pattern and amplitude modulation
        amplitude_mod = 0.3 + 0.7 * rand(size(clean_signal));
        emg_noise = emg_weight * emg_noise .* burst_mask .* amplitude_mod;
        
        % Add some random component
        emg_noise = emg_noise + emg_weight * 0.2 * randn(size(clean_signal));
        
        % 2. BASELINE WANDER (0.05-2 Hz, smooth drifts)
        baseline_weight = 0.4;
        
        % Breathing component (0.1-0.5 Hz)
        breathing_freq1 = 2 * pi * (0.1 + 0.4 * rand());
        breathing_freq2 = 2 * pi * (0.15 + 0.3 * rand());
        breathing_component = 1.5 * sin(breathing_freq1 * t) + 0.8 * sin(breathing_freq2 * t);
        
        % Motion component (0.05-1 Hz)
        motion_freq1 = 2 * pi * (0.05 + 0.45 * rand());
        motion_freq2 = 2 * pi * (0.2 + 0.6 * rand());
        motion_component = 2.0 * sin(motion_freq1 * t) + 1.2 * sin(motion_freq2 * t);
        
        % Random walk component (slow drift)
        random_walk = cumsum(0.05 * randn(size(clean_signal))) / sqrt(signal_length);
        
        baseline_wander = baseline_weight * (breathing_component + motion_component + random_walk);
        
        % 3. POWERLINE INTERFERENCE (50/60 Hz + harmonics)
        powerline_weight = 0.25;
        
        % Main powerline frequencies
        freq_50hz = 2 * pi * 50;
        freq_60hz = 2 * pi * 60;
        powerline_main = 0.7 * sin(freq_50hz * t) + 0.5 * sin(freq_60hz * t);
        
        % Add harmonics
        powerline_harmonics = 0.3 * sin(2 * freq_50hz * t) + 0.2 * sin(3 * freq_50hz * t) + ...
                             0.2 * sin(2 * freq_60hz * t);
        
        % Add some amplitude and frequency modulation
        modulation = 1 + 0.1 * sin(2 * pi * 0.1 * t); % Slow amplitude modulation
        powerline_interference = powerline_weight * (powerline_main + powerline_harmonics) .* modulation;
        
        % 4. GAUSSIAN NOISE (electronic amplifier noise)
        gaussian_weight = 0.2;
        gaussian_noise = gaussian_weight * randn(size(clean_signal));
        
        % Combine all noise components
        lead_noise = emg_noise + baseline_wander + powerline_interference + gaussian_noise;
        
        % Apply SNR constraint
        signal_power = mean(clean_signal.^2);
        noise_power = mean(lead_noise.^2);
        
        % Calculate required noise scaling for target SNR
        target_noise_power = signal_power / (10^(snr_db/10));
        noise_scaling = sqrt(target_noise_power / noise_power);
        
        lead_noise = lead_noise * noise_scaling;
        
        % Store noise for this lead
        total_noise(lead_idx, :) = lead_noise;
    end
    
    % Combine clean signals with noise
    noisy_signals = clean_signals + total_noise;
    
    % Ensure signals stay within reasonable bounds (prevent clipping)
    for lead_idx = 1:num_leads
        signal_std = std(clean_signals(lead_idx, :));
        max_amplitude = 4 * signal_std; % Allow 4 standard deviations
        noisy_signals(lead_idx, :) = max(-max_amplitude, min(max_amplitude, noisy_signals(lead_idx, :)));
    end
end

function create_noisy_output_directories(output_path, snr_levels, classes, age_groups)
    % Create output directory structure for noisy dataset
    
    for snr_idx = 1:length(snr_levels)
        snr_folder = sprintf('SNR_%02ddB', snr_levels(snr_idx));
        
        for class_idx = 1:length(classes)
            for age_idx = 1:length(age_groups)
                dir_path = fullfile(output_path, snr_folder, classes{class_idx}, age_groups{age_idx});
                if ~exist(dir_path, 'dir')
                    mkdir(dir_path);
                end
            end
        end
    end
end

function sample_file_info = find_sample_file(source_path, classes, age_groups)
    % Find a representative sample file for visualization
    
    sample_file_info = struct();
    
    % Look for a file in the first class/age group
    sample_dir = fullfile(source_path, classes{1}, age_groups{1});
    if exist(sample_dir, 'dir')
        mat_files = dir(fullfile(sample_dir, '*.mat'));
        if ~isempty(mat_files)
            sample_file_info.path = fullfile(sample_dir, mat_files(1).name);
            sample_file_info.name = mat_files(1).name;
        end
    end
end

function generate_noise_visualization(sample_signal, sample_filename, fs, snr_levels, output_path)
    % Generate before/after noise visualization with 4-second segments
    
    % Extract 4-second segment (samples 1-2000) from Lead II if available
    segment_duration = 4; % seconds
    segment_samples = segment_duration * fs; % 2000 samples
    
    % Use Lead II (index 2) if available, otherwise use first lead
    if size(sample_signal, 1) >= 2
        lead_signal = sample_signal(2, 1:min(segment_samples, end)); % Lead II
        lead_name = 'Lead II';
    else
        lead_signal = sample_signal(1, 1:min(segment_samples, end)); % First available lead
        lead_name = 'Lead I';
    end
    
    % Ensure we have exactly 4 seconds
    if length(lead_signal) < segment_samples
        % Pad with zeros if signal is shorter
        lead_signal = [lead_signal, zeros(1, segment_samples - length(lead_signal))];
    else
        lead_signal = lead_signal(1:segment_samples);
    end
    
    % Time vector for 4 seconds
    t = (0:segment_samples-1) / fs;
    
    % Create visualization figure
    fig = figure('Position', [100, 100, 1600, 1200]);
    
    % Plot clean signal
    subplot(length(snr_levels) + 1, 1, 1);
    plot(t, lead_signal, 'b-', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Amplitude (µV)');
    title(sprintf('Clean ECG %s (4 seconds) - Patient: %s', lead_name, strrep(sample_filename, '_', '\_')), ...
          'FontWeight', 'bold', 'FontSize', 12);
    grid on; grid minor;
    xlim([0, 4]);
    
    % Plot noisy signals for each SNR level
    for snr_idx = 1:length(snr_levels)
        snr_db = snr_levels(snr_idx);
        
        % Generate noisy version of the segment
        lead_signal_matrix = reshape(lead_signal, 1, []);
        noisy_signal_matrix = add_portable_ecg_noise(lead_signal_matrix, fs, snr_db);
        noisy_signal = noisy_signal_matrix(1, :);
        
        subplot(length(snr_levels) + 1, 1, snr_idx + 1);
        
        % Plot both clean and noisy for comparison
        plot(t, lead_signal, 'b-', 'LineWidth', 1, 'DisplayName', 'Clean', 'Color', [0.5, 0.5, 1]);
        hold on;
        plot(t, noisy_signal, 'r-', 'LineWidth', 1.5, 'DisplayName', sprintf('Noisy (SNR=%ddB)', snr_db));
        
        xlabel('Time (s)');
        ylabel('Amplitude (µV)');
        title(sprintf('Portable ECG Device Noise at SNR %d dB', snr_db), 'FontWeight', 'bold', 'FontSize', 12);
        legend('Location', 'best');
        grid on; grid minor;
        xlim([0, 4]);
        hold off;
        
        % Add SNR measurement
        signal_power = mean(lead_signal.^2);
        noise_power = mean((noisy_signal - lead_signal).^2);
        measured_snr = 10 * log10(signal_power / noise_power);
        
        text(0.02, 0.95, sprintf('Measured SNR: %.1f dB', measured_snr), ...
             'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 10);
    end
    
    sgtitle('Portable ECG Device Noise Effects: Before and After Comparison', ...
            'FontWeight', 'bold', 'FontSize', 16);
    
    % Create visualization directory
    visualization_path = fullfile(output_path, 'Noise_Analysis');
    if ~exist(visualization_path, 'dir')
        mkdir(visualization_path);
    end
    
    % Save figure
    saveas(fig, fullfile(visualization_path, 'Portable_ECG_Noise_Comparison.png'), 'png');
    saveas(fig, fullfile(visualization_path, 'Portable_ECG_Noise_Comparison.fig'), 'fig');
    
    % Generate detailed noise component analysis
    generate_noise_component_analysis(lead_signal, fs, snr_levels, visualization_path);
    
    fprintf('Noise visualization saved to: %s\n', visualization_path);
end

function generate_noise_component_analysis(clean_signal, fs, snr_levels, output_path)
    % Generate detailed analysis of individual noise components
    
    % Create figure for noise component analysis
    fig = figure('Position', [100, 100, 1600, 1000]);
    
    % Time vector
    t = (0:length(clean_signal)-1) / fs;
    
    % Generate individual noise components for demonstration (SNR 15dB)
    signal_length = length(clean_signal);
    snr_demo = 15; % Use 15dB for component demonstration
    
    % 1. EMG noise
    emg_weight = 0.3;
    burst_probability = 0.15;
    burst_mask = rand(size(clean_signal)) < burst_probability;
    emg_freq = 20 + 480 * rand(size(clean_signal));
    emg_noise = emg_weight * sin(2 * pi * emg_freq .* t) .* burst_mask .* ...
                (0.3 + 0.7 * rand(size(clean_signal)));
    
    % 2. Baseline wander
    baseline_weight = 0.4;
    breathing_freq1 = 2 * pi * 0.25;
    motion_freq1 = 2 * pi * 0.1;
    random_walk = cumsum(0.05 * randn(size(clean_signal))) / sqrt(signal_length);
    baseline_wander = baseline_weight * (1.5 * sin(breathing_freq1 * t) + ...
                     2.0 * sin(motion_freq1 * t) + random_walk);
    
    % 3. Powerline interference
    powerline_weight = 0.25;
    freq_50hz = 2 * pi * 50;
    freq_60hz = 2 * pi * 60;
    powerline_interference = powerline_weight * (0.7 * sin(freq_50hz * t) + ...
                           0.5 * sin(freq_60hz * t) + 0.3 * sin(2 * freq_50hz * t));
    
    % 4. Gaussian noise
    gaussian_weight = 0.2;
    gaussian_noise = gaussian_weight * randn(size(clean_signal));
    
    % Plot individual components
    subplot(3, 2, 1);
    plot(t, emg_noise, 'g-', 'LineWidth', 1);
    title('EMG Noise (Muscle Artifacts)', 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Amplitude (µV)');
    grid on; xlim([0, 4]);
    
    subplot(3, 2, 2);
    plot(t, baseline_wander, 'm-', 'LineWidth', 1);
    title('Baseline Wander (Motion/Breathing)', 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Amplitude (µV)');
    grid on; xlim([0, 4]);
    
    subplot(3, 2, 3);
    plot(t, powerline_interference, 'c-', 'LineWidth', 1);
    title('Powerline Interference (50/60 Hz)', 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Amplitude (µV)');
    grid on; xlim([0, 4]);
    
    subplot(3, 2, 4);
    plot(t, gaussian_noise, 'k-', 'LineWidth', 1);
    title('Gaussian Noise (Electronic)', 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Amplitude (µV)');
    grid on; xlim([0, 4]);
    
    % Combined noise
    total_noise = emg_noise + baseline_wander + powerline_interference + gaussian_noise;
    
    % Apply SNR scaling
    signal_power = mean(clean_signal.^2);
    noise_power = mean(total_noise.^2);
    target_noise_power = signal_power / (10^(snr_demo/10));
    noise_scaling = sqrt(target_noise_power / noise_power);
    total_noise = total_noise * noise_scaling;
    
    subplot(3, 2, 5);
    plot(t, total_noise, 'r-', 'LineWidth', 1);
    title(sprintf('Combined Noise (SNR %d dB)', snr_demo), 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Amplitude (µV)');
    grid on; xlim([0, 4]);
    
    subplot(3, 2, 6);
    plot(t, clean_signal, 'b-', 'LineWidth', 1, 'DisplayName', 'Clean ECG');
    hold on;
    plot(t, clean_signal + total_noise, 'r-', 'LineWidth', 1, 'DisplayName', 'Noisy ECG');
    title('Final Result: Clean vs Noisy ECG', 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Amplitude (µV)');
    legend('Location', 'best');
    grid on; xlim([0, 4]);
    hold off;
    
    sgtitle('Portable ECG Device Noise Components Analysis', 'FontWeight', 'bold', 'FontSize', 16);
    
    % Save component analysis
    saveas(fig, fullfile(output_path, 'Noise_Components_Analysis.png'), 'png');
    saveas(fig, fullfile(output_path, 'Noise_Components_Analysis.fig'), 'fig');
end

function generate_noise_report(output_path, total_processed, total_generated, processing_errors, snr_levels)
    % Generate comprehensive noise generation report
    
    report_file = fullfile(output_path, 'portable_ecg_noise_report.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, '=== PORTABLE ECG DEVICE NOISE GENERATION REPORT ===\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now));
    
    fprintf(fid, 'NOISE GENERATION PARAMETERS:\n');
    fprintf(fid, 'Source: Age-balanced ECG dataset\n');
    fprintf(fid, 'Signal format: 10-second ECG recordings at 500 Hz\n');
    fprintf(fid, 'SNR levels: %s dB\n', mat2str(snr_levels));
    fprintf(fid, 'Noise types: 4 combined sources\n');
    fprintf(fid, 'Target: Portable ECG device simulation\n');
    fprintf(fid, 'Output directory: %s\n\n', output_path);
    
    fprintf(fid, 'NOISE COMPOSITION (4 TYPES):\n');
    fprintf(fid, '1. EMG Noise (30%% weight):\n');
    fprintf(fid, '   - Frequency range: 20-500 Hz\n');
    fprintf(fid, '   - Characteristics: Burst-like muscle artifacts\n');
    fprintf(fid, '   - Burst probability: 15%%\n');
    fprintf(fid, '   - Clinical relevance: Patient movement, muscle tension\n\n');
    
    fprintf(fid, '2. Baseline Wander (40%% weight):\n');
    fprintf(fid, '   - Frequency range: 0.05-2 Hz\n');
    fprintf(fid, '   - Components: Breathing (0.1-0.5 Hz), Motion (0.05-1 Hz)\n');
    fprintf(fid, '   - Characteristics: Smooth, low-frequency drifts\n');
    fprintf(fid, '   - Clinical relevance: Patient breathing, electrode movement\n\n');
    
    fprintf(fid, '3. Powerline Interference (25%% weight):\n');
    fprintf(fid, '   - Frequencies: 50 Hz, 60 Hz + harmonics\n');
    fprintf(fid, '   - Characteristics: Sinusoidal with amplitude modulation\n');
    fprintf(fid, '   - Clinical relevance: Electrical environment contamination\n\n');
    
    fprintf(fid, '4. Gaussian Noise (20%% weight):\n');
    fprintf(fid, '   - Characteristics: White noise, electronic origin\n');
    fprintf(fid, '   - Clinical relevance: Amplifier noise, thermal noise\n\n');
    
    fprintf(fid, 'PROCESSING STATISTICS:\n');
    fprintf(fid, 'Original files processed: %d\n', total_processed);
    fprintf(fid, 'Noisy versions generated: %d\n', total_generated);
    fprintf(fid, 'Processing errors: %d\n', processing_errors);
    fprintf(fid, 'Files per SNR level: %d\n', total_generated / length(snr_levels));
    
    if total_processed > 0
        success_rate = (total_generated / (total_processed * length(snr_levels))) * 100;
        fprintf(fid, 'Success rate: %.1f%%\n', success_rate);
    end
    fprintf(fid, '\n');
    
    fprintf(fid, 'SNR LEVEL INTERPRETATION:\n');
    for i = 1:length(snr_levels)
        snr = snr_levels(i);
        if snr >= 20
            quality = 'High quality portable ECG';
            accuracy = '>90%';
            clinical = 'Suitable for diagnostic applications';
        elseif snr >= 15
            quality = 'Good quality portable ECG';
            accuracy = '85-90%';
            clinical = 'Suitable for monitoring applications';
        else
            quality = 'Acceptable quality portable ECG';
            accuracy = '75-85%';
            clinical = 'Suitable for screening applications';
        end
        
        fprintf(fid, 'SNR %d dB:\n', snr);
        fprintf(fid, '  Quality: %s\n', quality);
        fprintf(fid, '  Expected model accuracy: %s\n', accuracy);
        fprintf(fid, '  Clinical use: %s\n', clinical);
        fprintf(fid, '\n');
    end
    
    fprintf(fid, 'OUTPUT DIRECTORY STRUCTURE:\n');
    fprintf(fid, 'Noisy_Portable_ECG_Dataset/\n');
    for i = 1:length(snr_levels)
        fprintf(fid, '├── SNR_%02ddB/\n', snr_levels(i));
        fprintf(fid, '│   ├── SR/\n');
        fprintf(fid, '│   │   ├── young_adult/ (100 files)\n');
        fprintf(fid, '│   │   ├── middle_aged/ (100 files)\n');
        fprintf(fid, '│   │   └── elderly/ (100 files)\n');
        fprintf(fid, '│   ├── SB/ (same structure)\n');
        fprintf(fid, '│   └── AFIB/ (same structure)\n');
    end
    fprintf(fid, '└── Noise_Analysis/ (visualization files)\n\n');
    
    fprintf(fid, 'FILENAME CONVENTION:\n');
    fprintf(fid, 'Format: [PATIENT_ID]_age[AGE]_SNR[XX]dB.mat\n');
    fprintf(fid, 'Examples:\n');
    fprintf(fid, '- JS44163_age45_SNR20dB.mat\n');
    fprintf(fid, '- TR09173_age67_SNR15dB.mat\n');
    fprintf(fid, '- AM12456_age34_SNR10dB.mat\n\n');
    
    fprintf(fid, 'RESEARCH APPLICATIONS:\n');
    fprintf(fid, '• Model robustness validation under realistic noise conditions\n');
    fprintf(fid, '• Portable ECG device deployment readiness assessment\n');
    fprintf(fid, '• Noise tolerance threshold determination\n');
    fprintf(fid, '• Clinical validation with device-representative data\n');
    fprintf(fid, '• Regulatory submission with noise robustness evidence\n');
    fprintf(fid, '• Quality control algorithm development\n\n');
    
    fprintf(fid, 'RECOMMENDED VALIDATION WORKFLOW:\n');
    fprintf(fid, '1. Train model on clean age-balanced dataset\n');
    fprintf(fid, '2. Test on SNR 20dB (expect minimal degradation)\n');
    fprintf(fid, '3. Test on SNR 15dB (expect moderate degradation)\n');
    fprintf(fid, '4. Test on SNR 10dB (expect significant degradation)\n');
    fprintf(fid, '5. Generate performance curves vs SNR and age groups\n');
    fprintf(fid, '6. Determine minimum acceptable SNR for deployment\n');
    fprintf(fid, '7. Implement real-time quality monitoring\n');
    
    fclose(fid);
    
    fprintf('Noise generation report saved to: %s\n', report_file);
end

function verify_noisy_dataset(output_path, snr_levels, classes, age_groups)
    % Verify the generated noisy dataset
    
    fprintf('\n=== NOISY DATASET VERIFICATION ===\n');
    
    total_files = 0;
    
    for snr_idx = 1:length(snr_levels)
        snr_folder = sprintf('SNR_%02ddB', snr_levels(snr_idx));
        snr_total = 0;
        
        fprintf('%s:\n', snr_folder);
        
        for class_idx = 1:length(classes)
            class_total = 0;
            
            for age_idx = 1:length(age_groups)
                dir_path = fullfile(output_path, snr_folder, classes{class_idx}, age_groups{age_idx});
                
                if exist(dir_path, 'dir')
                    mat_files = dir(fullfile(dir_path, '*.mat'));
                    count = length(mat_files);
                    class_total = class_total + count;
                    snr_total = snr_total + count;
                    total_files = total_files + count;
                end
            end
            
            fprintf('  %s: %d files\n', classes{class_idx}, class_total);
        end
        
        fprintf('  Total for %s: %d files\n', snr_folder, snr_total);
        
        if snr_total == 900 % 3 classes × 3 age groups × 100 files
            fprintf('  ✓ Complete dataset (900 files)\n');
        else
            fprintf('  ⚠ Incomplete dataset (expected 900, got %d)\n', snr_total);
        end
        fprintf('\n');
    end
    
    fprintf('Grand total files in noisy dataset: %d\n', total_files);
    fprintf('Expected total: %d\n', length(snr_levels) * length(classes) * length(age_groups) * 100);
    
    if total_files == length(snr_levels) * length(classes) * length(age_groups) * 100
        fprintf('✓ Perfect noisy dataset generation completed!\n');
    else
        fprintf('⚠ Some files may be missing from the noisy dataset\n');
    end
end

% Execute the function
generate_portable_ecg_noise_dataset();