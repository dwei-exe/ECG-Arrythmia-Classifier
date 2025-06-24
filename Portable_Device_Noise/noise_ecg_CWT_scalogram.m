function generate_ecg_lead2_scalogram_dataset()
    % ECG LEAD II SCALOGRAM GENERATOR WITH CWT
    % Converts noisy ECG datasets to scalogram images using Continuous Wavelet Transform
    % Focus: Lead II only, first 4 seconds, 227x227 RGB scalograms
    % Enhanced processing with robust error handling and validation
    
    % Define paths
    source_dataset_path = 'C:\Users\henry\Downloads\ECG-Dx\Noisy_Portable_ECG_Dataset';
    output_dataset_path = 'C:\Users\henry\Downloads\ECG-Dx\Lead2_Scalogram_Dataset';
    
    % Create output directory
    if ~exist(output_dataset_path, 'dir')
        mkdir(output_dataset_path);
    end
    
    % Processing parameters
    fs = 500; % Sampling frequency (Hz)
    duration_seconds = 4; % Extract first 4 seconds
    target_samples = fs * duration_seconds; % 2000 samples
    target_size = [227, 227]; % Target scalogram image size
    voices_per_octave = 12; % CWT parameter
    lead_index = 2; % Lead II (second lead)
    
    % Dataset structure
    snr_levels = [20, 15, 10]; % SNR levels to process
    classes = {'SR', 'SB', 'AFIB'};
    age_groups = {'young_adult', 'middle_aged', 'elderly'};
    
    % Initialize counters
    total_processed = 0;
    total_converted = 0;
    conversion_errors = 0;
    missing_lead2 = 0;
    short_signals = 0;
    
    fprintf('=== ECG LEAD II SCALOGRAM GENERATOR ===\n');
    fprintf('Source dataset: %s\n', source_dataset_path);
    fprintf('Output dataset: %s\n', output_dataset_path);
    fprintf('Processing: Lead II only (first %d seconds = %d samples)\n', duration_seconds, target_samples);
    fprintf('Target image size: %dx%d pixels\n', target_size(1), target_size(2));
    fprintf('CWT parameters: Analytic Morlet, %d voices per octave\n', voices_per_octave);
    fprintf('Colormap: Jet (128 colors)\n');
    fprintf('SNR levels: %s dB\n', mat2str(snr_levels));
    fprintf('Classes: %s\n', strjoin(classes, ', '));
    fprintf('Age groups: %s\n\n', strjoin(age_groups, ', '));
    
    % Verify source dataset exists
    if ~exist(source_dataset_path, 'dir')
        error('Source dataset directory not found: %s', source_dataset_path);
    end
    
    % Create output directory structure
    create_scalogram_output_directories(output_dataset_path, snr_levels, classes, age_groups);
    
    % Process each SNR level
    for snr_idx = 1:length(snr_levels)
        snr_level = snr_levels(snr_idx);
        snr_folder = sprintf('SNR_%02ddB', snr_level);
        fprintf('Processing %s...\n', snr_folder);
        
        snr_source_path = fullfile(source_dataset_path, snr_folder);
        snr_output_path = fullfile(output_dataset_path, snr_folder);
        
        if ~exist(snr_source_path, 'dir')
            fprintf('  Warning: SNR directory not found: %s\n', snr_source_path);
            continue;
        end
        
        % Process each class
        for class_idx = 1:length(classes)
            class_name = classes{class_idx};
            fprintf('  Processing %s class...\n', class_name);
            
            % Process each age group
            for age_idx = 1:length(age_groups)
                age_group = age_groups{age_idx};
                fprintf('    Processing %s age group...\n', age_group);
                
                % Define input and output directories
                input_dir = fullfile(snr_source_path, class_name, age_group);
                output_dir = fullfile(snr_output_path, class_name, age_group);
                
                if ~exist(input_dir, 'dir')
                    fprintf('      Warning: Directory not found: %s\n', input_dir);
                    continue;
                end
                
                % Get all .mat files
                mat_files = dir(fullfile(input_dir, '*.mat'));
                mat_files = mat_files(~strcmp({mat_files.name}, 'metadata.mat'));
                
                if isempty(mat_files)
                    fprintf('      Warning: No .mat files found in %s\n', input_dir);
                    continue;
                end
                
                fprintf('      Found %d files to process\n', length(mat_files));
                
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
                            fprintf('        Warning: No valid ECG signals found in %s\n', mat_files(file_idx).name);
                            conversion_errors = conversion_errors + 1;
                            continue;
                        end
                        
                        % Enhanced validation
                        [num_leads, signal_length] = size(ecg_signals);
                        
                        % Check if Lead II is available
                        if num_leads < lead_index
                            fprintf('        Warning: Lead II not available in %s (only %d leads)\n', ...
                                    mat_files(file_idx).name, num_leads);
                            missing_lead2 = missing_lead2 + 1;
                            conversion_errors = conversion_errors + 1;
                            continue;
                        end
                        
                        % Check signal length
                        if signal_length < target_samples
                            fprintf('        Warning: Signal too short in %s (%d < %d samples)\n', ...
                                    mat_files(file_idx).name, signal_length, target_samples);
                            short_signals = short_signals + 1;
                            conversion_errors = conversion_errors + 1;
                            continue;
                        end
                        
                        % Extract Lead II and first 4 seconds
                        lead2_signal = ecg_signals(lead_index, 1:target_samples);
                        lead2_signal = double(lead2_signal); % Ensure double precision
                        
                        % Generate scalogram using enhanced CWT processing
                        scalogram_img = generate_enhanced_scalogram(lead2_signal, fs, voices_per_octave, target_size);
                        
                        if isempty(scalogram_img)
                            fprintf('        Warning: Failed to generate scalogram for %s\n', mat_files(file_idx).name);
                            conversion_errors = conversion_errors + 1;
                            continue;
                        end
                        
                        % Create output filename preserving age information
                        % Format: [PATIENT_ID]_age[AGE]_Lead2_4sec.png
                        output_filename = create_scalogram_filename(base_name);
                        output_filepath = fullfile(output_dir, output_filename);
                        
                        % Save scalogram image
                        imwrite(scalogram_img, output_filepath);
                        total_converted = total_converted + 1;
                        
                        % Progress update
                        if mod(total_processed, 100) == 0
                            fprintf('        Processed %d files, converted %d scalograms...\n', ...
                                    total_processed, total_converted);
                        end
                        
                    catch ME
                        fprintf('        Error processing %s: %s\n', mat_files(file_idx).name, ME.message);
                        conversion_errors = conversion_errors + 1;
                        continue;
                    end
                end
                
                % Count generated scalograms in this group
                generated_count = count_png_files(output_dir);
                fprintf('      Completed %s: %d scalograms generated\n', age_group, generated_count);
            end
        end
        
        fprintf('  Completed %s\n', snr_folder);
    end
    
    % Final summary
    fprintf('\n=== SCALOGRAM GENERATION SUMMARY ===\n');
    fprintf('Total ECG files processed: %d\n', total_processed);
    fprintf('Scalograms successfully generated: %d\n', total_converted);
    fprintf('Conversion errors: %d\n', conversion_errors);
    fprintf('  - Missing Lead II: %d\n', missing_lead2);
    fprintf('  - Short signals: %d\n', short_signals);
    fprintf('  - Other errors: %d\n', conversion_errors - missing_lead2 - short_signals);
    
    if total_processed > 0
        fprintf('Success rate: %.1f%%\n', (total_converted/total_processed)*100);
    end
    fprintf('Output directory: %s\n', output_dataset_path);
    
    % Generate comprehensive report
    generate_scalogram_report(output_dataset_path, total_processed, total_converted, conversion_errors, ...
                             missing_lead2, short_signals, snr_levels, target_size, voices_per_octave);
    
    % Create sample visualization
    generate_sample_scalogram_visualization(output_dataset_path, snr_levels, classes);
    
    % Verify output dataset
    verify_scalogram_dataset(output_dataset_path, snr_levels, classes, age_groups);
    
    fprintf('\nLead II scalogram dataset generation complete!\n');
    fprintf('Dataset ready for CNN training and evaluation.\n');
end

function ecg_signals = extract_ecg_signals(ecg_data)
    % Enhanced ECG signal extraction with better validation
    ecg_signals = [];
    
    % Try different common field names in WFDB format
    field_names = {'val', 'data', 'signal', 'ecg', 'y'};
    
    for i = 1:length(field_names)
        if isfield(ecg_data, field_names{i})
            signals = ecg_data.(field_names{i});
            
            % Ensure signals are numeric
            if ~isnumeric(signals)
                continue;
            end
            
            % Handle different orientations
            if size(signals, 1) > size(signals, 2)
                signals = signals'; % Transpose if needed
            end
            
            % Enhanced validation: require at least 2 leads and 2000 samples
            if size(signals, 1) >= 2 && size(signals, 2) >= 2000
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
            % Handle orientation
            if size(data, 1) > size(data, 2)
                data = data';
            end
            % Enhanced validation
            if size(data, 1) >= 2 && size(data, 2) >= 2000
                ecg_signals = data;
                return;
            end
        end
    end
end

function scalogram_img = generate_enhanced_scalogram(signal, fs, voices_per_octave, target_size)
    % Enhanced scalogram generation with robust error handling
    scalogram_img = [];
    
    try
        % Step 1: Enhanced signal preprocessing
        signal = double(signal);
        
        % Remove DC component
        signal = signal - mean(signal);
        
        % Enhanced normalization with division-by-zero protection
        signal_std = std(signal);
        if signal_std > eps
            signal = signal / signal_std;
        else
            % Handle constant signal case
            fprintf('          Warning: Signal has zero standard deviation, using original signal\n');
        end
        
        % Step 2: Apply Continuous Wavelet Transform
        % Use analytic Morlet wavelet with specified voices per octave
        try
            [wt, frequencies] = cwt(signal, 'amor', fs, 'VoicesPerOctave', voices_per_octave);
        catch
            % Fallback for older MATLAB versions
            try
                [wt, frequencies] = cwt(signal, 'amor', fs);
            catch ME
                fprintf('          Error in CWT: %s\n', ME.message);
                return;
            end
        end
        
        % Step 3: Generate scalogram (magnitude calculation)
        scalogram = abs(wt);
        
        % Check for valid scalogram
        if isempty(scalogram) || all(scalogram(:) == 0)
            fprintf('          Warning: Empty or zero scalogram generated\n');
            return;
        end
        
        % Step 4: Apply logarithmic scaling for better visualization
        scalogram = log10(scalogram + eps); % Add eps to avoid log(0)
        
        % Step 5: Robust normalization to [0, 1] range
        scalogram_min = min(scalogram(:));
        scalogram_max = max(scalogram(:));
        
        if scalogram_max > scalogram_min
            scalogram = (scalogram - scalogram_min) / (scalogram_max - scalogram_min);
        else
            % Handle constant scalogram case
            scalogram = zeros(size(scalogram));
        end
        
        % Step 6: Apply jet colormap with 128 colors
        jet_colormap = jet(128);
        
        % Convert scalogram to indexed image (1-128 range)
        scalogram_indexed = round(scalogram * 127) + 1;
        scalogram_indexed = max(1, min(128, scalogram_indexed)); % Clamp values
        
        % Convert indexed image to RGB
        scalogram_rgb = ind2rgb(scalogram_indexed, jet_colormap);
        
        % Step 7: Resize to target dimensions using bilinear interpolation
        scalogram_img = imresize(scalogram_rgb, target_size, 'bilinear');
        
        % Step 8: Convert to uint8 for image saving
        scalogram_img = uint8(scalogram_img * 255);
        
    catch ME
        fprintf('          Error in scalogram generation: %s\n', ME.message);
        scalogram_img = [];
    end
end

function output_filename = create_scalogram_filename(base_name)
    % Create scalogram filename preserving age information
    % Input: JS44163_age45_SNR20dB -> Output: JS44163_age45_Lead2_4sec.png
    
    % Remove SNR information but preserve age
    age_match = regexp(base_name, '(.+_age\d+)', 'tokens');
    
    if ~isempty(age_match)
        patient_age_part = age_match{1}{1};
        output_filename = sprintf('%s_Lead2_4sec.png', patient_age_part);
    else
        % Fallback if age pattern not found
        output_filename = sprintf('%s_Lead2_4sec.png', base_name);
    end
end

function create_scalogram_output_directories(output_path, snr_levels, classes, age_groups)
    % Create organized output directory structure for scalograms
    
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

function count = count_png_files(dir_path)
    % Count PNG files in directory
    if exist(dir_path, 'dir')
        png_files = dir(fullfile(dir_path, '*.png'));
        count = length(png_files);
    else
        count = 0;
    end
end

function generate_scalogram_report(output_path, total_processed, total_converted, conversion_errors, ...
                                  missing_lead2, short_signals, snr_levels, target_size, voices_per_octave)
    % Generate comprehensive scalogram generation report
    
    report_file = fullfile(output_path, 'lead2_scalogram_generation_report.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, '=== ECG LEAD II SCALOGRAM GENERATION REPORT ===\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now));
    
    fprintf(fid, 'PROCESSING PARAMETERS:\n');
    fprintf(fid, 'Source: Noisy portable ECG dataset\n');
    fprintf(fid, 'Focus: Lead II only (2nd lead)\n');
    fprintf(fid, 'Duration: First 4 seconds (2000 samples at 500 Hz)\n');
    fprintf(fid, 'Target image size: %dx%d pixels\n', target_size(1), target_size(2));
    fprintf(fid, 'Wavelet: Analytic Morlet (amor)\n');
    fprintf(fid, 'Voices per octave: %d\n', voices_per_octave);
    fprintf(fid, 'Colormap: Jet (128 colors)\n');
    fprintf(fid, 'Interpolation: Bilinear\n');
    fprintf(fid, 'SNR levels processed: %s dB\n', mat2str(snr_levels));
    fprintf(fid, 'Output directory: %s\n\n', output_path);
    
    fprintf(fid, 'ENHANCED PROCESSING PIPELINE:\n');
    fprintf(fid, '1. Signal Preprocessing:\n');
    fprintf(fid, '   - DC component removal (mean subtraction)\n');
    fprintf(fid, '   - Robust normalization with division-by-zero protection\n');
    fprintf(fid, '   - Double precision conversion\n\n');
    
    fprintf(fid, '2. CWT Application:\n');
    fprintf(fid, '   - Analytic Morlet wavelet\n');
    fprintf(fid, '   - %d voices per octave for optimal frequency resolution\n', voices_per_octave);
    fprintf(fid, '   - Automatic frequency range selection\n\n');
    
    fprintf(fid, '3. Scalogram Generation:\n');
    fprintf(fid, '   - Magnitude calculation from complex CWT coefficients\n');
    fprintf(fid, '   - Logarithmic scaling for enhanced visualization\n');
    fprintf(fid, '   - Robust normalization to [0,1] range\n\n');
    
    fprintf(fid, '4. Colormap Application:\n');
    fprintf(fid, '   - Jet colormap with 128 discrete colors\n');
    fprintf(fid, '   - Index mapping with value clamping\n');
    fprintf(fid, '   - RGB conversion for image format\n\n');
    
    fprintf(fid, '5. Image Resizing:\n');
    fprintf(fid, '   - Bilinear interpolation to %dx%d pixels\n', target_size(1), target_size(2));
    fprintf(fid, '   - Uint8 conversion for efficient storage\n\n');
    
    fprintf(fid, 'PROCESSING STATISTICS:\n');
    fprintf(fid, 'Total ECG files processed: %d\n', total_processed);
    fprintf(fid, 'Scalograms successfully generated: %d\n', total_converted);
    fprintf(fid, 'Total conversion errors: %d\n', conversion_errors);
    fprintf(fid, '\nError Breakdown:\n');
    fprintf(fid, '- Missing Lead II: %d (%.1f%%)\n', missing_lead2, (missing_lead2/total_processed)*100);
    fprintf(fid, '- Short signals (<2000 samples): %d (%.1f%%)\n', short_signals, (short_signals/total_processed)*100);
    fprintf(fid, '- Other errors: %d (%.1f%%)\n', conversion_errors - missing_lead2 - short_signals, ...
           ((conversion_errors - missing_lead2 - short_signals)/total_processed)*100);
    
    if total_processed > 0
        success_rate = (total_converted / total_processed) * 100;
        fprintf(fid, '\nSuccess rate: %.1f%%\n', success_rate);
    end
    
    fprintf(fid, '\nOUTPUT STRUCTURE:\n');
    fprintf(fid, 'Lead2_Scalogram_Dataset/\n');
    for i = 1:length(snr_levels)
        fprintf(fid, '├── SNR_%02ddB/\n', snr_levels(i));
        fprintf(fid, '│   ├── SR/\n');
        fprintf(fid, '│   │   ├── young_adult/ (scalogram .png files)\n');
        fprintf(fid, '│   │   ├── middle_aged/ (scalogram .png files)\n');
        fprintf(fid, '│   │   └── elderly/ (scalogram .png files)\n');
        fprintf(fid, '│   ├── SB/ (same structure)\n');
        fprintf(fid, '│   └── AFIB/ (same structure)\n');
    end
    fprintf(fid, '└── Sample_Visualizations/ (example scalograms)\n\n');
    
    fprintf(fid, 'FILENAME CONVENTION:\n');
    fprintf(fid, 'Format: [PATIENT_ID]_age[AGE]_Lead2_4sec.png\n');
    fprintf(fid, 'Examples:\n');
    fprintf(fid, '- JS44163_age45_Lead2_4sec.png\n');
    fprintf(fid, '- TR09173_age67_Lead2_4sec.png\n');
    fprintf(fid, '- AM12456_age34_Lead2_4sec.png\n\n');
    
    fprintf(fid, 'CLINICAL SIGNIFICANCE:\n');
    fprintf(fid, 'Lead II Focus:\n');
    fprintf(fid, '- Most diagnostically important lead for rhythm analysis\n');
    fprintf(fid, '- Optimal P-wave and QRS complex visualization\n');
    fprintf(fid, '- Standard lead for arrhythmia classification\n');
    fprintf(fid, '- Consistent electrode placement across patients\n\n');
    
    fprintf(fid, '4-Second Duration:\n');
    fprintf(fid, '- Captures 4-5 complete cardiac cycles at normal heart rates\n');
    fprintf(fid, '- Sufficient for rhythm pattern recognition\n');
    fprintf(fid, '- Reduces computational requirements\n');
    fprintf(fid, '- Standardizes input across all samples\n\n');
    
    fprintf(fid, 'RESEARCH APPLICATIONS:\n');
    fprintf(fid, '• CNN-based ECG classification model training\n');
    fprintf(fid, '• Noise robustness validation across SNR levels\n');
    fprintf(fid, '• Age-stratified performance analysis\n');
    fprintf(fid, '• Transfer learning for portable ECG devices\n');
    fprintf(fid, '• Comparative analysis of time-frequency representations\n');
    fprintf(fid, '• Clinical deployment readiness assessment\n\n');
    
    fprintf(fid, 'RECOMMENDED CNN ARCHITECTURE CONSIDERATIONS:\n');
    fprintf(fid, '• Input shape: 227×227×3 (RGB scalogram images)\n');
    fprintf(fid, '• Pre-trained backbones: ResNet, EfficientNet, Vision Transformer\n');
    fprintf(fid, '• Data augmentation: Rotation, scaling, color jittering\n');
    fprintf(fid, '• Class balancing: 100 samples per age group ensures balance\n');
    fprintf(fid, '• Validation strategy: Age-stratified cross-validation\n');
    fprintf(fid, '• Performance metrics: Sensitivity, specificity, F1-score per class\n\n');
    
    fprintf(fid, 'SCALOGRAM INTERPRETATION:\n');
    fprintf(fid, '• X-axis: Time (0-4 seconds)\n');
    fprintf(fid, '• Y-axis: Frequency (0-250 Hz based on Nyquist)\n');
    fprintf(fid, '• Color intensity: Wavelet coefficient magnitude\n');
    fprintf(fid, '• Red/Yellow: High energy regions (QRS complexes, P-waves)\n');
    fprintf(fid, '• Blue/Purple: Low energy regions (baseline, noise)\n');
    fprintf(fid, '• Patterns: Rhythmic structures indicate normal/abnormal rhythms\n');
    
    fclose(fid);
    
    fprintf('Scalogram generation report saved to: %s\n', report_file);
end

function generate_sample_scalogram_visualization(output_path, snr_levels, classes)
    % Generate sample scalogram visualization showing different SNR levels and classes
    
    fprintf('Generating sample scalogram visualization...\n');
    
    % Create figure
    fig = figure('Position', [100, 100, 1600, 1000]);
    
    % Find sample files for visualization
    sample_count = 0;
    max_samples = length(snr_levels) * length(classes); % 3×3 = 9 samples
    
    subplot_idx = 0;
    
    for snr_idx = 1:length(snr_levels)
        snr_folder = sprintf('SNR_%02ddB', snr_levels(snr_idx));
        
        for class_idx = 1:length(classes)
            subplot_idx = subplot_idx + 1;
            
            % Look for a sample file in young_adult age group
            sample_dir = fullfile(output_path, snr_folder, classes{class_idx}, 'young_adult');
            
            if exist(sample_dir, 'dir')
                png_files = dir(fullfile(sample_dir, '*.png'));
                
                if ~isempty(png_files)
                    % Load and display first available scalogram
                    img_path = fullfile(sample_dir, png_files(1).name);
                    img = imread(img_path);
                    
                    subplot(length(snr_levels), length(classes), subplot_idx);
                    imshow(img);
                    
                    % Extract patient info from filename
                    [~, filename, ~] = fileparts(png_files(1).name);
                    patient_info = strrep(filename, '_', ' ');
                    
                    title(sprintf('%s - %s\n%s', snr_folder, classes{class_idx}, patient_info), ...
                          'FontSize', 10, 'Interpreter', 'none');
                    
                    sample_count = sample_count + 1;
                end
            end
        end
    end
    
    if sample_count > 0
        sgtitle('Sample Lead II Scalograms: SNR Levels vs ECG Classes', ...
                'FontWeight', 'bold', 'FontSize', 16);
        
        % Add axis labels
        annotation('textbox', [0.02, 0.5, 0.03, 0.1], 'String', 'SNR Level', ...
                  'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'none', ...
                  'Rotation', 90, 'HorizontalAlignment', 'center');
        
        annotation('textbox', [0.5, 0.02, 0.1, 0.03], 'String', 'ECG Class', ...
                  'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'none', ...
                  'HorizontalAlignment', 'center');
        
        % Create sample visualization directory
        sample_viz_path = fullfile(output_path, 'Sample_Visualizations');
        if ~exist(sample_viz_path, 'dir')
            mkdir(sample_viz_path);
        end
        
        % Save figure
        saveas(fig, fullfile(sample_viz_path, 'Sample_Scalograms_Overview.png'), 'png');
        saveas(fig, fullfile(sample_viz_path, 'Sample_Scalograms_Overview.fig'), 'fig');
        
        fprintf('Sample scalogram visualization saved to: %s\n', sample_viz_path);
    else
        fprintf('No sample scalograms found for visualization.\n');
        close(fig);
    end
end

function verify_scalogram_dataset(output_path, snr_levels, classes, age_groups)
    % Verify the generated scalogram dataset
    
    fprintf('\n=== SCALOGRAM DATASET VERIFICATION ===\n');
    
    total_scalograms = 0;
    expected_per_group = 100; % Expected scalograms per age group
    
    for snr_idx = 1:length(snr_levels)
        snr_folder = sprintf('SNR_%02ddB', snr_levels(snr_idx));
        snr_total = 0;
        
        fprintf('%s:\n', snr_folder);
        
        for class_idx = 1:length(classes)
            class_total = 0;
            fprintf('  %s:\n', classes{class_idx});
            
            for age_idx = 1:length(age_groups)
                dir_path = fullfile(output_path, snr_folder, classes{class_idx}, age_groups{age_idx});
                count = count_png_files(dir_path);
                
                class_total = class_total + count;
                snr_total = snr_total + count;
                total_scalograms = total_scalograms + count;
                
                status = '✓';
                if count < expected_per_group
                    status = '⚠';
                end
                
                fprintf('    %s: %d scalograms %s\n', age_groups{age_idx}, count, status);
            end
            
            fprintf('    Total for %s: %d scalograms\n', classes{class_idx}, class_total);
        end
        
        fprintf('  Total for %s: %d scalograms\n', snr_folder, snr_total);
        
        expected_snr_total = length(classes) * length(age_groups) * expected_per_group;
        if snr_total == expected_snr_total
            fprintf('  ✓ Complete SNR dataset (%d scalograms)\n', expected_snr_total);
        else
            fprintf('  ⚠ Incomplete SNR dataset (expected %d, got %d)\n', expected_snr_total, snr_total);
        end
        fprintf('\n');
    end
    
    % Overall summary
    expected_total = length(snr_levels) * length(classes) * length(age_groups) * expected_per_group;
    
    fprintf('OVERALL SUMMARY:\n');
    fprintf('Total scalograms generated: %d\n', total_scalograms);
    fprintf('Expected total: %d\n', expected_total);
    fprintf('Completion rate: %.1f%%\n', (total_scalograms/expected_total)*100);
    
    if total_scalograms == expected_total
        fprintf('✓ Perfect scalogram dataset generation completed!\n');
    else
        fprintf('⚠ Some scalograms may be missing from the dataset\n');
    end
    
    % File size analysis
    if total_scalograms > 0
        fprintf('\nDataset characteristics:\n');
        fprintf('• Image format: 227×227 RGB PNG\n');
        fprintf('• Estimated dataset size: ~%.1f MB\n', total_scalograms * 0.1); % Rough estimate
        fprintf('• Ready for CNN training and validation\n');
    end
end

% Execute the function
generate_ecg_lead2_scalogram_dataset();