function convert_combined_25db_to_scalograms()
    % FOCUSED SCALOGRAM CONVERTER - COMBINED 25dB ONLY
    % Converts combined 25dB noisy ECG signals to scalogram images
    % Lead II, 4 seconds, 227x227 RGB scalograms with jet colormap
    % Patient names preserved in filenames
    
    % Define paths
    noisy_dataset_path = 'C:\Users\henry\Downloads\ECG-Dx\Combined_25dB_Dataset';
    scalogram_output_path = 'C:\Users\henry\Downloads\ECG-Dx\Combined_25dB_Scalograms';
    
    % Create output directory
    if ~exist(scalogram_output_path, 'dir')
        mkdir(scalogram_output_path);
    end
    
    % ECG parameters
    fs = 500; % Sampling frequency (Hz)
    target_size = [227, 227]; % Target image size
    voices_per_octave = 12; % Number of voices per octave
    duration_seconds = 4; % 4 seconds of data
    target_samples = fs * duration_seconds; % 2000 samples
    
    % Initialize counters
    total_processed = 0;
    total_converted = 0;
    conversion_errors = 0;
    
    fprintf('=== COMBINED 25dB SCALOGRAM CONVERTER ===\n');
    fprintf('Input dataset: %s\n', noisy_dataset_path);
    fprintf('Output dataset: %s\n', scalogram_output_path);
    fprintf('Target image size: %dx%d pixels\n', target_size(1), target_size(2));
    fprintf('Processing: Combined 25dB noisy Lead II (4 seconds)\n');
    fprintf('Wavelet: Analytic Morlet (amor)\n');
    fprintf('Voices per octave: %d\n', voices_per_octave);
    fprintf('Colormap: Jet (128 colors)\n\n');
    
    % Process training and validation datasets
    datasets = {'training', 'validation'};
    groups = {'AFIB', 'SB', 'SR'};
    
    for dataset_idx = 1:length(datasets)
        dataset_name = datasets{dataset_idx};
        fprintf('Processing %s dataset...\n', dataset_name);
        
        for group_idx = 1:length(groups)
            group_name = groups{group_idx};
            
            % Define input and output directories
            input_dir = fullfile(noisy_dataset_path, dataset_name, group_name);
            output_dir = fullfile(scalogram_output_path, dataset_name, group_name);
            
            if ~exist(input_dir, 'dir')
                fprintf('  Warning: Directory not found: %s\n', input_dir);
                continue;
            end
            
            % Create output directory
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end
            
            % Get all combined 25dB .mat files
            mat_files = dir(fullfile(input_dir, '*_COMBINED_25dB.mat'));
            
            if isempty(mat_files)
                fprintf('  Warning: No combined 25dB files found in %s\n', input_dir);
                continue;
            end
            
            fprintf('  Processing %s group: %d files\n', group_name, length(mat_files));
            
            % Process each noisy ECG file
            for file_idx = 1:length(mat_files)
                total_processed = total_processed + 1;
                
                mat_file_path = fullfile(input_dir, mat_files(file_idx).name);
                [~, base_name, ~] = fileparts(mat_files(file_idx).name);
                
                try
                    % Load noisy ECG data
                    ecg_data = load(mat_file_path);
                    
                    % Extract signal (should be stored as 'val')
                    if isfield(ecg_data, 'val')
                        noisy_signal = ecg_data.val;
                    else
                        fprintf('      Warning: No "val" field in %s\n', mat_files(file_idx).name);
                        conversion_errors = conversion_errors + 1;
                        continue;
                    end
                    
                    % Ensure signal is the right length
                    if length(noisy_signal) ~= target_samples
                        fprintf('      Warning: Signal length mismatch in %s (%d vs %d)\n', ...
                                mat_files(file_idx).name, length(noisy_signal), target_samples);
                        conversion_errors = conversion_errors + 1;
                        continue;
                    end
                    
                    % Generate scalogram for combined 25dB noisy Lead II
                    scalogram_img = generate_combined_25db_scalogram(noisy_signal, fs, voices_per_octave, target_size);
                    
                    % Create output filename preserving patient info
                    % Convert from: [PATIENT_ID]_age[AGE]_COMBINED_25dB.mat
                    % To: [PATIENT_ID]_age[AGE]_NOISE_COMBINED_25dB_Lead2_4sec.png
                    patient_part = strrep(base_name, '_COMBINED_25dB', '');
                    output_filename = sprintf('%s_NOISE_COMBINED_25dB_Lead2_4sec.png', patient_part);
                    output_filepath = fullfile(output_dir, output_filename);
                    
                    % Save scalogram image
                    imwrite(scalogram_img, output_filepath);
                    
                    total_converted = total_converted + 1;
                    
                    % Progress update
                    if mod(total_processed, 25) == 0
                        fprintf('      Processed %d files, converted %d...\n', total_processed, total_converted);
                    end
                    
                catch ME
                    fprintf('      Error processing %s: %s\n', mat_files(file_idx).name, ME.message);
                    conversion_errors = conversion_errors + 1;
                    continue;
                end
            end
            
            % Show completion for this group
            converted_count = count_converted_scalograms(output_dir);
            fprintf('  Completed %s group: %d scalograms generated\n', group_name, converted_count);
        end
        
        fprintf('Completed %s dataset\n\n', dataset_name);
    end
    
    % Final summary
    fprintf('=== SCALOGRAM CONVERSION SUMMARY ===\n');
    fprintf('Total noisy files processed: %d\n', total_processed);
    fprintf('Successfully converted: %d\n', total_converted);
    fprintf('Conversion errors: %d\n', conversion_errors);
    if total_processed > 0
        fprintf('Success rate: %.1f%%\n', (total_converted/total_processed)*100);
    end
    fprintf('Scalogram output directory: %s\n', scalogram_output_path);
    
    % Generate conversion report
    generate_combined_25db_conversion_report(scalogram_output_path, total_processed, total_converted, conversion_errors);
    
    % Analyze the generated dataset
    analyze_combined_25db_dataset(scalogram_output_path);
    
    % Generate sample visualization
    create_combined_25db_visualization(scalogram_output_path);
    
    fprintf('\nCombined 25dB scalogram dataset ready for model testing!\n');
end

function scalogram_img = generate_combined_25db_scalogram(signal, fs, voices_per_octave, target_size)
    % Generate scalogram for combined 25dB noisy Lead II using CWT
    
    % Preprocess signal
    signal = double(signal);
    signal = signal - mean(signal); % Remove DC component
    
    % Robust normalization (important for noisy signals)
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
    
    % Resize to target dimensions using bicubic interpolation
    scalogram_img = imresize(scalogram_rgb, target_size, 'bicubic');
    
    % Convert to uint8 for image saving
    scalogram_img = uint8(scalogram_img * 255);
end

function count = count_converted_scalograms(output_dir)
    % Count successfully converted scalogram files
    if exist(output_dir, 'dir')
        files = dir(fullfile(output_dir, '*_NOISE_COMBINED_25dB_Lead2_4sec.png'));
        count = length(files);
    else
        count = 0;
    end
end

function generate_combined_25db_conversion_report(scalogram_output_path, total_processed, total_converted, conversion_errors)
    % Generate comprehensive conversion report
    
    report_file = fullfile(scalogram_output_path, 'combined_25dB_scalogram_report.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, '=== COMBINED 25dB SCALOGRAM CONVERSION REPORT ===\n');
    fprintf(fid, 'Date: %s\n\n', datestr(now));
    
    fprintf(fid, 'CONVERSION PARAMETERS:\n');
    fprintf(fid, '- Source: Combined 25dB noisy ECG Lead II signals\n');
    fprintf(fid, '- Duration: 4 seconds (2000 samples @ 500 Hz)\n');
    fprintf(fid, '- Noise Type: Combined (realistic multi-source)\n');
    fprintf(fid, '- SNR Level: 25 dB (high-quality portable conditions)\n');
    fprintf(fid, '- Wavelet: Analytic Morlet (amor)\n');
    fprintf(fid, '- Voices per octave: 12\n');
    fprintf(fid, '- Target image size: 227x227 pixels\n');
    fprintf(fid, '- Color format: RGB\n');
    fprintf(fid, '- Colormap: Jet (128 colors)\n');
    fprintf(fid, '- Interpolation: Bicubic\n\n');
    
    fprintf(fid, 'DATASET COVERAGE:\n');
    fprintf(fid, '- ECG Classes: AFIB, SB, SR\n');
    fprintf(fid, '- Datasets: Training and Validation\n');
    fprintf(fid, '- Total combinations: 6 folders\n\n');
    
    fprintf(fid, 'CONVERSION STATISTICS:\n');
    fprintf(fid, 'Total noisy files processed: %d\n', total_processed);
    fprintf(fid, 'Successfully converted: %d\n', total_converted);
    fprintf(fid, 'Conversion errors: %d\n', conversion_errors);
    
    if total_processed > 0
        success_rate = (total_converted / total_processed) * 100;
        fprintf(fid, 'Success rate: %.1f%%\n', success_rate);
    end
    
    fprintf(fid, '\nOUTPUT STRUCTURE:\n');
    fprintf(fid, 'Directory structure: [training|validation]/[AFIB|SB|SR]/\n');
    fprintf(fid, 'Filename format: [PATIENT_ID]_age[AGE]_NOISE_COMBINED_25dB_Lead2_4sec.png\n');
    
    fprintf(fid, '\nEXAMPLE FILENAMES:\n');
    fprintf(fid, '- JS44163_age45_NOISE_COMBINED_25dB_Lead2_4sec.png\n');
    fprintf(fid, '- TR09173_age67_NOISE_COMBINED_25dB_Lead2_4sec.png\n');
    fprintf(fid, '- AF82501_age23_NOISE_COMBINED_25dB_Lead2_4sec.png\n');
    
    fprintf(fid, '\nCLINICAL SIGNIFICANCE:\n');
    fprintf(fid, '25dB SNR represents:\n');
    fprintf(fid, '- High-quality portable ECG recording conditions\n');
    fprintf(fid, '- Minimal but realistic noise contamination\n');
    fprintf(fid, '- Suitable for clinical-grade diagnostic applications\n');
    fprintf(fid, '- Expected model performance: >90%% accuracy\n');
    fprintf(fid, '- Deployment-ready signal quality\n');
    
    fprintf(fid, '\nMODEL TESTING RECOMMENDATIONS:\n');
    fprintf(fid, '1. Test model performance on generated scalograms\n');
    fprintf(fid, '2. Compare with clean data baseline performance\n');
    fprintf(fid, '3. Measure accuracy drop due to 25dB noise\n');
    fprintf(fid, '4. Validate robustness for portable deployment\n');
    fprintf(fid, '5. Use results to set quality control thresholds\n');
    
    if conversion_errors > 0
        fprintf(fid, '\nERROR ANALYSIS:\n');
        fprintf(fid, 'Common error causes:\n');
        fprintf(fid, '- Missing "val" field in noisy signal files\n');
        fprintf(fid, '- Signal length mismatches (expected 2000 samples)\n');
        fprintf(fid, '- Corrupted noise data files\n');
        fprintf(fid, '- Memory issues during CWT computation\n');
    else
        fprintf(fid, '\nNo conversion errors encountered - Perfect success rate!\n');
    end
    
    fclose(fid);
    
    fprintf('Conversion report saved to: %s\n', report_file);
end

function analyze_combined_25db_dataset(scalogram_output_path)
    % Analyze the generated combined 25dB scalogram dataset
    
    fprintf('\n=== COMBINED 25dB DATASET ANALYSIS ===\n');
    
    datasets = {'training', 'validation'};
    groups = {'AFIB', 'SB', 'SR'};
    
    total_scalograms = 0;
    class_counts = zeros(1, length(groups));
    dataset_counts = zeros(1, length(datasets));
    
    fprintf('Dataset Distribution:\n');
    
    for dataset_idx = 1:length(datasets)
        dataset_name = datasets{dataset_idx};
        dataset_total = 0;
        
        fprintf('\n%s Dataset:\n', upper(dataset_name));
        
        for group_idx = 1:length(groups)
            group_name = groups{group_idx};
            group_path = fullfile(scalogram_output_path, dataset_name, group_name);
            
            if exist(group_path, 'dir')
                png_files = dir(fullfile(group_path, '*_NOISE_COMBINED_25dB_Lead2_4sec.png'));
                count = length(png_files);
                
                fprintf('  %s: %d scalograms\n', group_name, count);
                
                class_counts(group_idx) = class_counts(group_idx) + count;
                dataset_total = dataset_total + count;
            else
                fprintf('  %s: Directory not found\n', group_name);
            end
        end
        
        dataset_counts(dataset_idx) = dataset_total;
        total_scalograms = total_scalograms + dataset_total;
        fprintf('  %s Total: %d scalograms\n', dataset_name, dataset_total);
    end
    
    fprintf('\n=== SUMMARY STATISTICS ===\n');
    fprintf('Total Combined 25dB scalograms: %d\n', total_scalograms);
    
    % Class distribution
    fprintf('\nClass Distribution:\n');
    for group_idx = 1:length(groups)
        percentage = (class_counts(group_idx) / total_scalograms) * 100;
        fprintf('  %s: %d (%.1f%%)\n', groups{group_idx}, class_counts(group_idx), percentage);
    end
    
    % Dataset split
    fprintf('\nDataset Split:\n');
    for dataset_idx = 1:length(datasets)
        percentage = (dataset_counts(dataset_idx) / total_scalograms) * 100;
        fprintf('  %s: %d (%.1f%%)\n', datasets{dataset_idx}, dataset_counts(dataset_idx), percentage);
    end
    
    % Balance analysis
    if total_scalograms > 0
        max_class = max(class_counts);
        min_class = min(class_counts(class_counts > 0));
        imbalance_ratio = max_class / min_class;
        
        fprintf('\nBalance Analysis:\n');
        fprintf('  Imbalance ratio: %.2f\n', imbalance_ratio);
        if imbalance_ratio < 2
            fprintf('  Assessment: Well balanced\n');
        elseif imbalance_ratio < 5
            fprintf('  Assessment: Moderately imbalanced\n');
        else
            fprintf('  Assessment: Highly imbalanced\n');
        end
    end
end

function create_combined_25db_visualization(scalogram_output_path)
    % Create visualization of generated combined 25dB scalograms
    
    fprintf('\nGenerating sample visualization...\n');
    
    datasets = {'training', 'validation'};
    groups = {'AFIB', 'SB', 'SR'};
    
    % Create sample comparison figure
    fig = figure('Position', [100, 100, 1200, 800]);
    
    subplot_idx = 1;
    
    for dataset_idx = 1:length(datasets)
        for group_idx = 1:length(groups)
            group_path = fullfile(scalogram_output_path, datasets{dataset_idx}, groups{group_idx});
            
            if exist(group_path, 'dir')
                files = dir(fullfile(group_path, '*_NOISE_COMBINED_25dB_Lead2_4sec.png'));
                
                if ~isempty(files)
                    % Select first available sample
                    sample_file = fullfile(group_path, files(1).name);
                    
                    subplot(2, 3, subplot_idx);
                    img = imread(sample_file);
                    imshow(img);
                    
                    % Extract patient info from filename for title
                    [~, filename, ~] = fileparts(files(1).name);
                    patient_part = strrep(filename, '_NOISE_COMBINED_25dB_Lead2_4sec', '');
                    title_str = sprintf('%s - %s\n%s (25dB)', datasets{dataset_idx}, groups{group_idx}, ...
                               strrep(patient_part, '_', ' '));
                    title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
                    
                    % Add labels for first row
                    if dataset_idx == 1
                        xlabel('Time (4 seconds)', 'FontSize', 9);
                    end
                    if group_idx == 1
                        ylabel('Frequency', 'FontSize', 9);
                    end
                end
            end
            subplot_idx = subplot_idx + 1;
        end
    end
    
    sgtitle('Combined 25dB Noisy ECG Scalograms - Sample from Each Class', ...
            'FontWeight', 'bold', 'FontSize', 14);
    
    % Save visualization
    vis_path = fullfile(scalogram_output_path, 'Combined_25dB_Sample_Visualization.png');
    saveas(fig, vis_path, 'png');
    saveas(fig, strrep(vis_path, '.png', '.fig'), 'fig');
    
    % Add explanatory text
    figure('Position', [100, 100, 800, 600]);
    axis off;
    
    explanation_text = {
        '\bf{\fontsize{16}Combined 25dB Noisy ECG Scalogram Dataset}'
        ''
        '\bf{Dataset Specifications:}'
        '• Noise Type: Combined (realistic multi-source portable ECG noise)'
        '• SNR Level: 25 dB (high-quality portable conditions)'
        '• Signal Duration: 4 seconds (2000 samples @ 500 Hz)'
        '• Image Size: 227×227 RGB scalograms'
        '• Colormap: Jet (128 colors)'
        '• Transform: Continuous Wavelet Transform (Analytic Morlet)'
        ''
        '\bf{Noise Components in Combined Signal:}'
        '• Electronic amplifier noise (Gaussian)'
        '• Powerline interference (50/60 Hz)'
        '• Motion artifacts (baseline wander)'
        '• Muscle artifacts (EMG contamination)'
        '• Electrode movement effects'
        '• Contact impedance variations'
        ''
        '\bf{Clinical Significance:}'
        '• Represents high-quality portable ECG conditions'
        '• Suitable for clinical-grade diagnostic applications'
        '• Expected model performance: >90% accuracy'
        '• Validates model robustness for real-world deployment'
        ''
        '\bf{Research Applications:}'
        '• Model robustness validation under realistic noise'
        '• Performance baseline for portable ECG deployment'
        '• Quality control threshold determination'
        '• Clinical readiness assessment'
        ''
        '\bf{Next Steps:}'
        '• Test trained model on these noisy scalograms'
        '• Compare performance with clean data baseline'
        '• Measure robustness under 25dB noise conditions'
        '• Validate deployment readiness for portable devices'
    };
    
    text(0.05, 0.95, explanation_text, 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'FontName', 'Arial', ...
         'FontSize', 11, 'Interpreter', 'tex');
    
    % Save explanation
    exp_path = fullfile(scalogram_output_path, 'Combined_25dB_Dataset_Information.png');
    saveas(gcf, exp_path, 'png');
    saveas(gcf, strrep(exp_path, '.png', '.fig'), 'fig');
    
    fprintf('Sample visualization saved to: %s\n', scalogram_output_path);
    close all;
end

% Main execution
fprintf('Starting Combined 25dB Scalogram Converter...\n');
convert_combined_25db_to_scalograms();