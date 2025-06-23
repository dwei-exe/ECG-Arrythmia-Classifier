function convert_ecg_lead2_to_scalograms()
    % Convert ECG signals to scalogram images using Continuous Wavelet Transform
    % FOCUS: Only processes first 4 seconds of Lead II
    % Uses analytic Morlet wavelet with 12 voices per octave
    % Output: 227x227 RGB images with jet colormap
    
    % Define paths
    dataset_path = 'C:\Users\henry\Downloads\ECG-Dx\Organized_Dataset';
    output_path = 'C:\Users\henry\Downloads\ECG-Dx\Lead2_Scalogram_Dataset';
    
    % Create output directory
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end
    
    % ECG parameters
    fs = 500; % Sampling frequency (Hz)
    target_size = [227, 227]; % Target image size
    voices_per_octave = 12; % Number of voices per octave
    duration_seconds = 4; % Process only first 4 seconds
    target_samples = fs * duration_seconds; % 2000 samples
    
    % Initialize counters
    total_processed = 0;
    total_converted = 0;
    conversion_errors = 0;
    
    fprintf('=== ECG LEAD II TO SCALOGRAM CONVERSION (4 SECONDS) ===\n');
    fprintf('Dataset path: %s\n', dataset_path);
    fprintf('Output path: %s\n', output_path);
    fprintf('Target image size: %dx%d pixels\n', target_size(1), target_size(2));
    fprintf('Processing: Lead II only (first %d seconds = %d samples)\n', duration_seconds, target_samples);
    fprintf('Wavelet: Analytic Morlet (amor)\n');
    fprintf('Voices per octave: %d\n', voices_per_octave);
    fprintf('Colormap: Jet (128 colors)\n\n');
    
    % Process both training and validation sets
    datasets = {'training', 'validation'};
    groups = {'AFIB', 'SB', 'SR'}; % Updated to only include your 3 specified folders
    
    for dataset_idx = 1:length(datasets)
        dataset_name = datasets{dataset_idx};
        fprintf('Processing %s dataset...\n', dataset_name);
        
        for group_idx = 1:length(groups)
            group_name = groups{group_idx};
            
            % Define input and output directories
            input_dir = fullfile(dataset_path, dataset_name, group_name);
            output_dir = fullfile(output_path, dataset_name, group_name);
            
            if ~exist(input_dir, 'dir')
                fprintf('  Warning: Directory not found: %s\n', input_dir);
                continue;
            end
            
            % Create output directory
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
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
                    % Load ECG data
                    ecg_data = load(mat_file_path);
                    
                    % Extract ECG signals (handle different WFDB formats)
                    ecg_signals = extract_ecg_signals(ecg_data);
                    
                    if isempty(ecg_signals)
                        fprintf('    Warning: No valid ECG signals found in %s\n', mat_files(file_idx).name);
                        conversion_errors = conversion_errors + 1;
                        continue;
                    end
                    
                    % Extract Lead II (second lead) and truncate to first 4 seconds
                    if size(ecg_signals, 1) < 2
                        fprintf('    Warning: Lead II not found in %s (only %d leads)\n', ...
                                mat_files(file_idx).name, size(ecg_signals, 1));
                        conversion_errors = conversion_errors + 1;
                        continue;
                    end
                    
                    lead2_signal = ecg_signals(2, :); % Extract Lead II
                    
                    % Truncate to first 4 seconds (2000 samples)
                    if length(lead2_signal) < target_samples
                        fprintf('    Warning: Signal too short in %s (%d samples < %d required)\n', ...
                                mat_files(file_idx).name, length(lead2_signal), target_samples);
                        conversion_errors = conversion_errors + 1;
                        continue;
                    end
                    
                    lead2_signal = lead2_signal(1:target_samples);
                    
                    % Generate scalogram for Lead II
                    scalogram_img = generate_lead2_scalogram(lead2_signal, fs, voices_per_octave, target_size);
                    
                    % Save scalogram image with patient name preserved
                    output_filename = sprintf('%s_Lead2_4sec.png', base_name);
                    output_filepath = fullfile(output_dir, output_filename);
                    
                    imwrite(scalogram_img, output_filepath);
                    
                    total_converted = total_converted + 1;
                    
                    % Progress update
                    if mod(total_processed, 50) == 0
                        fprintf('    Processed %d files...\n', total_processed);
                    end
                    
                catch ME
                    fprintf('    Error processing %s: %s\n', mat_files(file_idx).name, ME.message);
                    conversion_errors = conversion_errors + 1;
                    continue;
                end
            end
            
            fprintf('  Completed %s group: %d files converted\n', group_name, ...
                   count_converted_files(output_dir));
        end
        
        fprintf('Completed %s dataset\n\n', dataset_name);
    end
    
    % Final summary
    fprintf('=== CONVERSION SUMMARY ===\n');
    fprintf('Total files processed: %d\n', total_processed);
    fprintf('Successfully converted: %d\n', total_converted);
    fprintf('Conversion errors: %d\n', conversion_errors);
    fprintf('Success rate: %.1f%%\n', (total_converted/total_processed)*100);
    fprintf('Output directory: %s\n', output_path);
    
    % Generate conversion report
    generate_lead2_conversion_report(output_path, total_processed, total_converted, conversion_errors, duration_seconds);
    
    % Generate dataset statistics
    analyze_lead2_dataset();
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
            
            % Validate signal dimensions (expect at least 2 leads for Lead II)
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

function scalogram_img = generate_lead2_scalogram(signal, fs, voices_per_octave, target_size)
    % Generate scalogram for Lead II using Continuous Wavelet Transform
    
    % Preprocess signal
    signal = double(signal);
    signal = signal - mean(signal); % Remove DC component
    
    % Robust normalization
    signal_std = std(signal);
    if signal_std > eps
        signal = signal / signal_std;  % Normalize
    else
        signal = signal; % Keep as is if std is too small
    end
    
    % Apply CWT with analytic Morlet wavelet
    try
        [wt, frequencies] = cwt(signal, 'amor', fs, 'VoicesPerOctave', voices_per_octave);
    catch
        % Fallback for older MATLAB versions
        [wt, frequencies] = cwt(signal, 'amor', fs);
    end
    
    % Convert to scalogram (magnitude)
    scalogram = abs(wt);
    
    % Apply logarithmic scaling for better visualization
    scalogram = log10(scalogram + eps);
    
    % Robust normalization to [0, 1] range
    scalogram_min = min(scalogram(:));
    scalogram_max = max(scalogram(:));
    if scalogram_max > scalogram_min
        scalogram = (scalogram - scalogram_min) / (scalogram_max - scalogram_min);
    else
        scalogram = zeros(size(scalogram)); % Handle constant signal case
    end
    
    % Apply jet colormap with 128 colors
    jet_colormap = jet(128);
    
    % Convert scalogram to RGB image
    scalogram_indexed = round(scalogram * 127) + 1; % Map to 1-128 range
    scalogram_indexed = max(1, min(128, scalogram_indexed)); % Clamp values
    scalogram_rgb = ind2rgb(scalogram_indexed, jet_colormap);
    
    % Resize to target dimensions using bicubic interpolation for better quality
    scalogram_img = imresize(scalogram_rgb, target_size, 'bicubic');
    
    % Convert to uint8 for image saving
    scalogram_img = uint8(scalogram_img * 255);
end

function count = count_converted_files(output_dir)
    % Count successfully converted files in output directory
    if exist(output_dir, 'dir')
        files = dir(fullfile(output_dir, '*_Lead2_4sec.png'));
        count = length(files);
    else
        count = 0;
    end
end

function generate_lead2_conversion_report(output_path, total_processed, total_converted, conversion_errors, duration_seconds)
    % Generate comprehensive conversion report
    
    report_file = fullfile(output_path, 'lead2_scalogram_conversion_report.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, '=== ECG LEAD II TO SCALOGRAM CONVERSION REPORT ===\n');
    fprintf(fid, 'Date: %s\n\n', datestr(now));
    
    fprintf(fid, 'CONVERSION PARAMETERS:\n');
    fprintf(fid, '- ECG Lead: Lead II only\n');
    fprintf(fid, '- Duration: First %d seconds\n', duration_seconds);
    fprintf(fid, '- Samples processed: %d (at 500 Hz)\n', duration_seconds * 500);
    fprintf(fid, '- Wavelet: Analytic Morlet (amor)\n');
    fprintf(fid, '- Voices per octave: 12\n');
    fprintf(fid, '- Target image size: 227x227 pixels\n');
    fprintf(fid, '- Color format: RGB\n');
    fprintf(fid, '- Colormap: Jet (128 colors)\n');
    fprintf(fid, '- Interpolation: Bicubic\n');
    fprintf(fid, '- Sampling frequency: 500 Hz\n\n');
    
    fprintf(fid, 'DATASET STRUCTURE:\n');
    fprintf(fid, '- Groups processed: AFIB, SB, SR\n');
    fprintf(fid, '- Sets: training, validation\n');
    fprintf(fid, '- Total combinations: 6 folders\n\n');
    
    fprintf(fid, 'CONVERSION STATISTICS:\n');
    fprintf(fid, 'Total ECG files processed: %d\n', total_processed);
    fprintf(fid, 'Successfully converted: %d\n', total_converted);
    fprintf(fid, 'Conversion errors: %d\n', conversion_errors);
    
    if total_processed > 0
        success_rate = (total_converted / total_processed) * 100;
        fprintf(fid, 'Success rate: %.1f%%\n', success_rate);
    end
    
    fprintf(fid, '\nOUTPUT STRUCTURE:\n');
    fprintf(fid, 'Each ECG file generates:\n');
    fprintf(fid, '- Single Lead II scalogram: [PATIENT_ID]_age[AGE]_Lead2_4sec.png\n');
    fprintf(fid, '\nExample filenames:\n');
    fprintf(fid, '- JS44163_age45_Lead2_4sec.png\n');
    fprintf(fid, '- TR09173_age67_Lead2_4sec.png\n');
    
    fprintf(fid, '\nDIRECTORY STRUCTURE:\n');
    fprintf(fid, 'training/[AFIB|SB|SR]/[lead2_scalogram_images]\n');
    fprintf(fid, 'validation/[AFIB|SB|SR]/[lead2_scalogram_images]\n');
    
    fprintf(fid, '\nERROR ANALYSIS:\n');
    if conversion_errors > 0
        fprintf(fid, 'Common error causes:\n');
        fprintf(fid, '- Missing Lead II in ECG data\n');
        fprintf(fid, '- Signal length < %d samples\n', duration_seconds * 500);
        fprintf(fid, '- Invalid file format\n');
        fprintf(fid, '- Corrupted data files\n');
    else
        fprintf(fid, 'No conversion errors encountered.\n');
    end
    
    fclose(fid);
    
    fprintf('Conversion report saved to: %s\n', report_file);
end

function analyze_lead2_dataset()
    % Analyze the generated Lead II scalogram dataset
    
    dataset_path = 'C:\Users\henry\Downloads\ECG-Dx\Lead2_Scalogram_Dataset';
    
    fprintf('\n=== LEAD II SCALOGRAM DATASET ANALYSIS ===\n');
    
    datasets = {'training', 'validation'};
    groups = {'AFIB', 'SB', 'SR'};
    
    total_images = 0;
    group_counts = struct();
    
    for dataset_idx = 1:length(datasets)
        dataset_name = datasets{dataset_idx};
        fprintf('\n%s Dataset:\n', upper(dataset_name));
        
        for group_idx = 1:length(groups)
            group_name = groups{group_idx};
            group_path = fullfile(dataset_path, dataset_name, group_name);
            
            if exist(group_path, 'dir')
                png_files = dir(fullfile(group_path, '*_Lead2_4sec.png'));
                num_files = length(png_files);
                
                fprintf('  %s: %d Lead II scalograms\n', group_name, num_files);
                
                total_images = total_images + num_files;
                
                % Store counts for balance analysis
                field_name = sprintf('%s_%s', dataset_name, group_name);
                group_counts.(field_name) = num_files;
            else
                fprintf('  %s: Directory not found\n', group_name);
            end
        end
    end
    
    fprintf('\nTotal Lead II scalogram images: %d\n', total_images);
    
    % Dataset balance analysis
    fprintf('\n=== DATASET BALANCE ANALYSIS ===\n');
    training_total = 0;
    validation_total = 0;
    
    for group_idx = 1:length(groups)
        group_name = groups{group_idx};
        train_field = sprintf('training_%s', group_name);
        val_field = sprintf('validation_%s', group_name);
        
        train_count = 0;
        val_count = 0;
        
        if isfield(group_counts, train_field)
            train_count = group_counts.(train_field);
            training_total = training_total + train_count;
        end
        
        if isfield(group_counts, val_field)
            val_count = group_counts.(val_field);
            validation_total = validation_total + val_count;
        end
        
        total_group = train_count + val_count;
        if total_group > 0
            train_pct = (train_count / total_group) * 100;
            val_pct = (val_count / total_group) * 100;
            fprintf('%s: Training=%d (%.1f%%), Validation=%d (%.1f%%), Total=%d\n', ...
                    group_name, train_count, train_pct, val_count, val_pct, total_group);
        else
            fprintf('%s: No data found\n', group_name);
        end
    end
    
    fprintf('\nOverall: Training=%d, Validation=%d\n', training_total, validation_total);
    if training_total + validation_total > 0
        train_percentage = (training_total / (training_total + validation_total)) * 100;
        fprintf('Train/Validation split: %.1f%% / %.1f%%\n', train_percentage, 100 - train_percentage);
    end
end

% Main execution
fprintf('Starting ECG Lead II to Scalogram conversion...\n');
convert_ecg_lead2_to_scalograms();