function publication_visualization_utility()
    % Publication-ready visualization utility for Lead II ECG scalograms
    % Provides comprehensive analysis and visualization tools for research publications
    
    fprintf('=== PUBLICATION-READY VISUALIZATION UTILITY ===\n');
    fprintf('Available functions:\n');
    fprintf('1. create_sample_figure() - Create sample scalograms across groups\n');
    fprintf('2. create_comparison_figure() - Compare scalograms between conditions\n');
    fprintf('3. analyze_dataset_statistics() - Generate comprehensive statistics\n');
    fprintf('4. create_methodology_figure() - Show CWT methodology\n');
    fprintf('5. export_publication_figures() - Export all figures for publication\n');
    fprintf('6. create_age_analysis_figure() - Age-based analysis\n');
    fprintf('7. create_frequency_analysis() - Frequency domain analysis\n\n');
    
    % Set default paths
    global DATASET_PATH OUTPUT_PATH
    DATASET_PATH = 'C:\Users\henry\Downloads\ECG-Dx\Lead2_Scalogram_Dataset';
    OUTPUT_PATH = 'C:\Users\henry\Downloads\ECG-Dx\Publication_Figures';
    
    if ~exist(OUTPUT_PATH, 'dir')
        mkdir(OUTPUT_PATH);
    end
    
    % Run comprehensive analysis
    fprintf('Generating all publication figures...\n');
    create_sample_figure();
    create_comparison_figure();
    analyze_dataset_statistics();
    create_methodology_figure();
    create_age_analysis_figure();
    create_frequency_analysis();
    export_publication_figures();
    
    fprintf('All publication figures generated in: %s\n', OUTPUT_PATH);
end

function create_sample_figure()
    % Create publication-quality figure showing sample scalograms from each group
    
    global DATASET_PATH OUTPUT_PATH
    
    groups = {'SB', 'AFIB', 'GSVT', 'SR'};
    group_names = {'Sinus Bradycardia', 'Atrial Fibrillation', 'GSVT', 'Sinus Rhythm'};
    
    % Set publication parameters
    fig = figure('Position', [100, 100, 1600, 1000], 'Color', 'white');
    set(fig, 'PaperPositionMode', 'auto', 'PaperUnits', 'inches');
    
    samples_per_group = 3;
    subplot_idx = 1;
    
    for group_idx = 1:length(groups)
        group_name = groups{group_idx};
        group_path = fullfile(DATASET_PATH, 'training', group_name);
        
        if exist(group_path, 'dir')
            png_files = dir(fullfile(group_path, '*_Lead2_4sec.png'));
            
            % Select representative samples
            if length(png_files) >= samples_per_group
                sample_indices = round(linspace(1, length(png_files), samples_per_group));
                
                for sample_idx = 1:samples_per_group
                    subplot(length(groups), samples_per_group, subplot_idx);
                    
                    img_path = fullfile(group_path, png_files(sample_indices(sample_idx)).name);
                    img = imread(img_path);
                    
                    imagesc(img);
                    axis image off;
                    
                    % Add time and frequency axes for first column
                    if sample_idx == 1
                        % Add time axis (0-4 seconds)
                        time_ticks = [1, 57, 113, 170, 227]; % Approximate positions
                        time_labels = {'0', '1', '2', '3', '4'};
                        
                        % Add frequency axis (approximate values for CWT)
                        freq_ticks = [1, 30, 60, 90, 120, 150, 180, 210, 227];
                        freq_labels = {'250', '100', '50', '25', '12', '6', '3', '1.5', '1'};
                        
                        set(gca, 'XTick', time_ticks, 'XTickLabel', time_labels, ...
                                'YTick', freq_ticks, 'YTickLabel', freq_labels);
                        xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
                        ylabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
                        
                        % Add group label
                        text(-30, 113, group_names{group_idx}, 'Rotation', 90, ...
                             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                             'FontSize', 14, 'FontWeight', 'bold');
                    end
                    
                    % Add patient ID as title for top row
                    if group_idx == 1
                        [~, filename, ~] = fileparts(png_files(sample_indices(sample_idx)).name);
                        patient_id = extractBefore(filename, '_Lead2_4sec');
                        title(patient_id, 'FontSize', 10, 'Interpreter', 'none');
                    end
                    
                    subplot_idx = subplot_idx + 1;
                end
            end
        end
    end
    
    % Add colorbar
    colorbar('Position', [0.92, 0.15, 0.02, 0.7], 'FontSize', 12);
    
    % Add main title
    sgtitle('Lead II ECG Scalograms by Cardiac Condition (First 4 seconds)', ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    % Save figure
    saveas(fig, fullfile(OUTPUT_PATH, 'Figure1_Sample_Scalograms.png'), 'png');
    saveas(fig, fullfile(OUTPUT_PATH, 'Figure1_Sample_Scalograms.pdf'), 'pdf');
    
    fprintf('Generated Figure 1: Sample scalograms\n');
end

function create_comparison_figure()
    % Create side-by-side comparison of different cardiac conditions
    
    global DATASET_PATH OUTPUT_PATH
    
    groups = {'SR', 'AFIB', 'GSVT', 'SB'};
    group_names = {'Normal Sinus Rhythm', 'Atrial Fibrillation', 'GSVT', 'Sinus Bradycardia'};
    colors = [0.2, 0.6, 0.8; 0.8, 0.2, 0.2; 0.8, 0.6, 0.2; 0.4, 0.8, 0.4];
    
    fig = figure('Position', [100, 100, 1800, 600], 'Color', 'white');
    
    for group_idx = 1:length(groups)
        group_name = groups{group_idx};
        group_path = fullfile(DATASET_PATH, 'training', group_name);
        
        if exist(group_path, 'dir')
            png_files = dir(fullfile(group_path, '*_Lead2_4sec.png'));
            
            if ~isempty(png_files)
                % Select middle file for consistency
                middle_idx = round(length(png_files) / 2);
                img_path = fullfile(group_path, png_files(middle_idx).name);
                img = imread(img_path);
                
                subplot(1, length(groups), group_idx);
                imagesc(img);
                axis image;
                
                % Add proper axes
                time_ticks = [1, 57, 113, 170, 227];
                time_labels = {'0', '1', '2', '3', '4'};
                freq_ticks = [1, 57, 113, 170, 227];
                freq_labels = {'250', '50', '12', '3', '1'};
                
                set(gca, 'XTick', time_ticks, 'XTickLabel', time_labels, ...
                        'YTick', freq_ticks, 'YTickLabel', freq_labels, ...
                        'FontSize', 11);
                
                xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
                if group_idx == 1
                    ylabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
                end
                
                title(group_names{group_idx}, 'FontSize', 13, 'FontWeight', 'bold', ...
                      'Color', colors(group_idx, :));
                
                % Add border with group color
                hold on;
                rectangle('Position', [0.5, 0.5, 227, 227], 'EdgeColor', colors(group_idx, :), ...
                         'LineWidth', 3);
                hold off;
            end
        end
    end
    
    % Add unified colorbar
    colorbar('Position', [0.93, 0.2, 0.015, 0.6], 'FontSize', 12);
    
    sgtitle('Comparative Analysis: Lead II Scalograms Across Cardiac Conditions', ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    % Save figure
    saveas(fig, fullfile(OUTPUT_PATH, 'Figure2_Condition_Comparison.png'), 'png');
    saveas(fig, fullfile(OUTPUT_PATH, 'Figure2_Condition_Comparison.pdf'), 'pdf');
    
    fprintf('Generated Figure 2: Condition comparison\n');
end

function stats = analyze_dataset_statistics()
    % Generate comprehensive dataset statistics for publication
    
    global DATASET_PATH OUTPUT_PATH
    
    groups = {'SB', 'AFIB', 'GSVT', 'SR'};
    datasets = {'training', 'validation'};
    
    stats = struct();
    
    fprintf('\n=== DATASET STATISTICS FOR PUBLICATION ===\n');
    
    total_patients = 0;
    group_totals = zeros(1, length(groups));
    
    % Collect statistics
    for dataset_idx = 1:length(datasets)
        dataset_name = datasets{dataset_idx};
        
        for group_idx = 1:length(groups)
            group_name = groups{group_idx};
            group_path = fullfile(DATASET_PATH, dataset_name, group_name);
            
            if exist(group_path, 'dir')
                png_files = dir(fullfile(group_path, '*_Lead2_4sec.png'));
                count = length(png_files);
                
                field_name = sprintf('%s_%s', dataset_name, group_name);
                stats.(field_name) = count;
                group_totals(group_idx) = group_totals(group_idx) + count;
                total_patients = total_patients + count;
                
                % Extract age information
                ages = [];
                for file_idx = 1:length(png_files)
                    filename = png_files(file_idx).name;
                    age_match = regexp(filename, 'age(\d+)', 'tokens');
                    if ~isempty(age_match)
                        ages(end+1) = str2double(age_match{1}{1});
                    end
                end
                
                if ~isempty(ages)
                    age_field = sprintf('%s_%s_ages', dataset_name, group_name);
                    stats.(age_field) = ages;
                end
            end
        end
    end
    
    % Create statistics table figure
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');
    
    % Create summary table
    subplot(2, 2, 1);
    table_data = zeros(length(groups), 3); % [Training, Validation, Total]
    
    for group_idx = 1:length(groups)
        group_name = groups{group_idx};
        train_field = sprintf('training_%s', group_name);
        val_field = sprintf('validation_%s', group_name);
        
        train_count = 0;
        val_count = 0;
        
        if isfield(stats, train_field)
            train_count = stats.(train_field);
        end
        if isfield(stats, val_field)
            val_count = stats.(val_field);
        end
        
        table_data(group_idx, :) = [train_count, val_count, train_count + val_count];
    end
    
    % Display as table
    axis off;
    col_labels = {'Training', 'Validation', 'Total'};
    row_labels = {'SB', 'AFIB', 'GSVT', 'SR'};
    
    table_handle = uitable('Parent', gca, 'Data', table_data, ...
                          'ColumnName', col_labels, 'RowName', row_labels, ...
                          'Position', [0.1, 0.1, 0.8, 0.8], ...
                          'FontSize', 12);
    title('Dataset Distribution by Condition', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Age distribution analysis
    subplot(2, 2, 2);
    all_ages = [];
    group_ages = cell(length(groups), 1);
    
    for group_idx = 1:length(groups)
        group_name = groups{group_idx};
        group_all_ages = [];
        
        for dataset_idx = 1:length(datasets)
            dataset_name = datasets{dataset_idx};
            age_field = sprintf('%s_%s_ages', dataset_name, group_name);
            
            if isfield(stats, age_field)
                ages = stats.(age_field);
                group_all_ages = [group_all_ages, ages];
                all_ages = [all_ages, ages];
            end
        end
        
        group_ages{group_idx} = group_all_ages;
    end
    
    % Box plot of ages by group
    if ~isempty(all_ages)
        boxplot([group_ages{:}], [repmat(1, 1, length(group_ages{1})), ...
                                 repmat(2, 1, length(group_ages{2})), ...
                                 repmat(3, 1, length(group_ages{3})), ...
                                 repmat(4, 1, length(group_ages{4}))], ...
                'Labels', groups);
        title('Age Distribution by Condition', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Cardiac Condition', 'FontSize', 12);
        ylabel('Age (years)', 'FontSize', 12);
        grid on;
    end
    
    % Dataset balance pie chart
    subplot(2, 2, 3);
    pie(group_totals, groups);
    title('Dataset Balance Across Conditions', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Summary statistics text
    subplot(2, 2, 4);
    axis off;
    
    summary_text = {
        sprintf('Total Patients: %d', total_patients)
        sprintf('Training: %d (%.1f%%)', sum(table_data(:,1)), sum(table_data(:,1))/total_patients*100)
        sprintf('Validation: %d (%.1f%%)', sum(table_data(:,2)), sum(table_data(:,2))/total_patients*100)
        ''
        'Condition Distribution:'
        sprintf('  SB: %d (%.1f%%)', group_totals(1), group_totals(1)/total_patients*100)
        sprintf('  AFIB: %d (%.1f%%)', group_totals(2), group_totals(2)/total_patients*100)
        sprintf('  GSVT: %d (%.1f%%)', group_totals(3), group_totals(3)/total_patients*100)
        sprintf('  SR: %d (%.1f%%)', group_totals(4), group_totals(4)/total_patients*100)
    };
    
    if ~isempty(all_ages)
        summary_text{end+1} = '';
        summary_text{end+1} = sprintf('Age Range: %d - %d years', min(all_ages), max(all_ages));
        summary_text{end+1} = sprintf('Mean Age: %.1f ± %.1f years', mean(all_ages), std(all_ages));
    end
    
    text(0.1, 0.9, summary_text, 'FontSize', 12, 'VerticalAlignment', 'top');
    
    sgtitle('Dataset Statistics Summary', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save figure
    saveas(fig, fullfile(OUTPUT_PATH, 'Figure3_Dataset_Statistics.png'), 'png');
    saveas(fig, fullfile(OUTPUT_PATH, 'Figure3_Dataset_Statistics.pdf'), 'pdf');
    
    % Save statistics to file
    stats_file = fullfile(OUTPUT_PATH, 'dataset_statistics.txt');
    fid = fopen(stats_file, 'w');
    
    fprintf(fid, '=== LEAD II SCALOGRAM DATASET STATISTICS ===\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now));
    
    fprintf(fid, 'OVERALL STATISTICS:\n');
    fprintf(fid, 'Total patients: %d\n', total_patients);
    fprintf(fid, 'Training samples: %d (%.1f%%)\n', sum(table_data(:,1)), sum(table_data(:,1))/total_patients*100);
    fprintf(fid, 'Validation samples: %d (%.1f%%)\n\n', sum(table_data(:,2)), sum(table_data(:,2))/total_patients*100);
    
    fprintf(fid, 'CONDITION BREAKDOWN:\n');
    for group_idx = 1:length(groups)
        fprintf(fid, '%s: Training=%d, Validation=%d, Total=%d (%.1f%%)\n', ...
                groups{group_idx}, table_data(group_idx,1), table_data(group_idx,2), ...
                table_data(group_idx,3), table_data(group_idx,3)/total_patients*100);
    end
    
    if ~isempty(all_ages)
        fprintf(fid, '\nAGE STATISTICS:\n');
        fprintf(fid, 'Age range: %d - %d years\n', min(all_ages), max(all_ages));
        fprintf(fid, 'Mean age: %.1f ± %.1f years\n', mean(all_ages), std(all_ages));
        fprintf(fid, 'Median age: %.1f years\n', median(all_ages));
    end
    
    fclose(fid);
    
    fprintf('Generated Figure 3: Dataset statistics\n');
    fprintf('Statistics saved to: %s\n', stats_file);
    
    return
end

function create_methodology_figure()
    % Create figure explaining the CWT methodology
    
    global OUTPUT_PATH
    
    % Generate sample ECG signal (synthetic)
    fs = 500;
    t = 0:1/fs:4-1/fs;
    
    % Create synthetic ECG-like signal
    ecg_signal = 0.5 * sin(2*pi*1.2*t) + ... % Heart rate component
                 0.3 * sin(2*pi*15*t) + ...  % P-wave component
                 0.8 * sin(2*pi*25*t) + ...  % QRS component
                 0.2 * sin(2*pi*5*t) + ...   % T-wave component
                 0.1 * randn(size(t));        % Noise
    
    fig = figure('Position', [100, 100, 1600, 1000], 'Color', 'white');
    
    % Original signal
    subplot(3, 2, [1, 2]);
    plot(t, ecg_signal, 'b-', 'LineWidth', 2);
    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Amplitude (mV)', 'FontSize', 12, 'FontWeight', 'bold');
    title('A) Original Lead II ECG Signal (4 seconds)', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim([0, 4]);
    
    % CWT scalogram
    subplot(3, 2, [3, 4]);
    [wt, frequencies] = cwt(ecg_signal, 'amor', fs, 'VoicesPerOctave', 12);
    scalogram = abs(wt);
    scalogram_log = log10(scalogram + eps);
    
    imagesc(t, frequencies, scalogram_log);
    set(gca, 'YDir', 'normal', 'YScale', 'log');
    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    title('B) Continuous Wavelet Transform Scalogram', 'FontSize', 14, 'FontWeight', 'bold');
    colorbar;
    
    % Wavelet example
    subplot(3, 2, 5);
    [psi, x] = morlet(-8, 8, 1000);
    plot(x, real(psi), 'r-', 'LineWidth', 2);
    hold on;
    plot(x, imag(psi), 'b-', 'LineWidth', 2);
    plot(x, abs(psi), 'k--', 'LineWidth', 2);
    xlabel('Time', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
    title('C) Analytic Morlet Wavelet', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Real', 'Imaginary', 'Magnitude', 'Location', 'best');
    grid on;
    
    % Final processed image
    subplot(3, 2, 6);
    % Simulate the final processing
    scalogram_normalized = (scalogram_log - min(scalogram_log(:))) / ...
                          (max(scalogram_log(:)) - min(scalogram_log(:)));
    scalogram_resized = imresize(scalogram_normalized, [227, 227]);
    
    imagesc(scalogram_resized);
    axis image;
    xlabel('Time (227 pixels)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Frequency (227 pixels)', 'FontSize', 12, 'FontWeight', 'bold');
    title('D) Final 227×227 RGB Scalogram', 'FontSize', 14, 'FontWeight', 'bold');
    colorbar;
    
    sgtitle('ECG to Scalogram Conversion Methodology', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save figure
    saveas(fig, fullfile(OUTPUT_PATH, 'Figure4_Methodology.png'), 'png');
    saveas(fig, fullfile(OUTPUT_PATH, 'Figure4_Methodology.pdf'), 'pdf');
    
    fprintf('Generated Figure 4: Methodology\n');
end

function create_age_analysis_figure()
    % Create age-based analysis figure
    
    global DATASET_PATH OUTPUT_PATH
    
    groups = {'SB', 'AFIB', 'GSVT', 'SR'};
    datasets = {'training', 'validation'};
    age_bins = [0:10:100]; % Age bins: 0-10, 10-20, ..., 90-100
    
    % Collect age data
    age_data = struct();
    for group_idx = 1:length(groups)
        group_name = groups{group_idx};
        all_ages = [];
        
        for dataset_idx = 1:length(datasets)
            dataset_name = datasets{dataset_idx};
            group_path = fullfile(DATASET_PATH, dataset_name, group_name);
            
            if exist(group_path, 'dir')
                png_files = dir(fullfile(group_path, '*_Lead2_4sec.png'));
                
                for file_idx = 1:length(png_files)
                    filename = png_files(file_idx).name;
                    age_match = regexp(filename, 'age(\d+)', 'tokens');
                    if ~isempty(age_match)
                        age = str2double(age_match{1}{1});
                        if age >= 0 && age <= 100
                            all_ages(end+1) = age;
                        end
                    end
                end
            end
        end
        
        age_data.(group_name) = all_ages;
    end
    
    fig = figure('Position', [100, 100, 1600, 1000], 'Color', 'white');
    
    % Age distribution histogram
    subplot(2, 3, [1, 2]);
    colors = [0.8, 0.2, 0.2; 0.2, 0.6, 0.8; 0.8, 0.6, 0.2; 0.4, 0.8, 0.4];
    
    hold on;
    for group_idx = 1:length(groups)
        group_name = groups{group_idx};
        ages = age_data.(group_name);
        if ~isempty(ages)
            histogram(ages, age_bins, 'FaceColor', colors(group_idx, :), ...
                     'FaceAlpha', 0.7, 'DisplayName', group_name);
        end
    end
    
    xlabel('Age (years)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Number of Patients', 'FontSize', 12, 'FontWeight', 'bold');
    title('Age Distribution by Cardiac Condition', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'northeast');
    grid on;
    
    % Box plot comparison
    subplot(2, 3, 3);
    group_ages_array = [];
    group_labels = [];
    
    for group_idx = 1:length(groups)
        group_name = groups{group_idx};
        ages = age_data.(group_name);
        group_ages_array = [group_ages_array, ages];
        group_labels = [group_labels, repmat(group_idx, 1, length(ages))];
    end
    
    if ~isempty(group_ages_array)
        boxplot(group_ages_array, group_labels, 'Labels', groups);
        ylabel('Age (years)', 'FontSize', 12, 'FontWeight', 'bold');
        title('Age Distribution Comparison', 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
    end
    
    % Sample scalograms by age groups
    age_groups = {'Young (≤40)', 'Middle (41-65)', 'Elderly (>65)'};
    subplot_positions = [2, 3, 4; 2, 3, 5; 2, 3, 6];
    
    for age_group_idx = 1:length(age_groups)
        subplot(subplot_positions(age_group_idx, 1), ...
                subplot_positions(age_group_idx, 2), ...
                subplot_positions(age_group_idx, 3));
        
        % Find representative sample
        found_sample = false;
        
        for group_idx = 1:length(groups)
            if found_sample, break; end
            
            group_name = groups{group_idx};
            group_path = fullfile(DATASET_PATH, 'training', group_name);
            
            if exist(group_path, 'dir')
                png_files = dir(fullfile(group_path, '*_Lead2_4sec.png'));
                
                for file_idx = 1:length(png_files)
                    filename = png_files(file_idx).name;
                    age_match = regexp(filename, 'age(\d+)', 'tokens');
                    
                    if ~isempty(age_match)
                        age = str2double(age_match{1}{1});
                        
                        age_in_range = false;
                        if age_group_idx == 1 && age <= 40
                            age_in_range = true;
                        elseif age_group_idx == 2 && age > 40 && age <= 65
                            age_in_range = true;
                        elseif age_group_idx == 3 && age > 65
                            age_in_range = true;
                        end
                        
                        if age_in_range
                            img_path = fullfile(group_path, filename);
                            img = imread(img_path);
                            imagesc(img);
                            axis image off;
                            title(sprintf('%s (Age %d, %s)', age_groups{age_group_idx}, age, group_name), ...
                                  'FontSize', 12, 'FontWeight', 'bold');
                            found_sample = true;
                            break;
                        end
                    end
                end
            end
        end
        
        if ~found_sample
            axis off;
            text(0.5, 0.5, 'No sample found', 'HorizontalAlignment', 'center');
        end
    end
    
    sgtitle('Age-Based Analysis of Lead II Scalograms', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save figure
    saveas(fig, fullfile(OUTPUT_PATH, 'Figure5_Age_Analysis.png'), 'png');
    saveas(fig, fullfile(OUTPUT_PATH, 'Figure5_Age_Analysis.pdf'), 'pdf');
    
    fprintf('Generated Figure 5: Age analysis\n');
end

function create_frequency_analysis()
    % Create frequency domain analysis
    
    global OUTPUT_PATH
    
    % Generate example frequency analysis
    fig = figure('Position', [100, 100, 1400, 800], 'Color', 'white');
    
    % CWT frequency scales
    subplot(2, 2, 1);
    fs = 500;
    voices_per_octave = 12;
    
    % Generate frequency vector for CWT
    freq_limits = [1, fs/2];
    num_octaves = log2(freq_limits(2)/freq_limits(1));
    num_voices = num_octaves * voices_per_octave;
    frequencies = freq_limits(1) * 2.^((0:num_voices-1) / voices_per_octave);
    
    semilogx(frequencies, 1:length(frequencies), 'b-', 'LineWidth', 2);
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Scale Index', 'FontSize', 12, 'FontWeight', 'bold');
    title('CWT Frequency Distribution', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    
    % Typical ECG frequency bands
    subplot(2, 2, 2);
    freq_bands = [0.05, 0.15; 0.15, 0.25; 0.25, 40; 40, 100; 100, 250];
    band_names = {'Baseline Wander', 'Low Freq Noise', 'QRS Complex', 'Muscle Noise', 'High Freq Noise'};
    band_colors = [0.7, 0.7, 0.7; 0.8, 0.6, 0.4; 0.2, 0.8, 0.2; 0.8, 0.4, 0.4; 0.4, 0.4, 0.8];
    
    for band_idx = 1:size(freq_bands, 1)
        freq_range = freq_bands(band_idx, :);
        bar_height = band_idx;
        
        barh(bar_height, freq_range(2) - freq_range(1), 'BarWidth', 0.8, ...
             'FaceColor', band_colors(band_idx, :), 'EdgeColor', 'k');
        hold on;
    end
    
    set(gca, 'YTick', 1:length(band_names), 'YTickLabel', band_names);
    xlabel('Frequency Range (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    title('ECG Frequency Bands', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim([0, 250]);
    
    % Wavelet resolution
    subplot(2, 2, 3);
    central_freq = 0.8; % For Morlet wavelet
    time_bandwidth = 60; % Morlet parameter
    
    test_freqs = logspace(0, 2, 50); % 1 to 100 Hz
    time_resolution = time_bandwidth ./ (2 * pi * test_freqs);
    freq_resolution = test_freqs / voices_per_octave;
    
    yyaxis left;
    loglog(test_freqs, time_resolution, 'b-', 'LineWidth', 2);
    ylabel('Time Resolution (s)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');
    
    yyaxis right;
    loglog(test_freqs, freq_resolution, 'r-', 'LineWidth', 2);
    ylabel('Frequency Resolution (Hz)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
    
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Time-Frequency Resolution Trade-off', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    
    % Processing pipeline summary
    subplot(2, 2, 4);
    axis off;
    
    pipeline_text = {
        'PROCESSING PIPELINE SUMMARY:'
        ''
        '1. Signal Preprocessing:'
        '   • DC removal: signal = signal - mean(signal)'
        '   • Normalization: signal = signal / std(signal)'
        ''
        '2. CWT Parameters:'
        '   • Wavelet: Analytic Morlet (amor)'
        '   • Voices per octave: 12'
        '   • Frequency range: 1-250 Hz'
        ''
        '3. Scalogram Generation:'
        '   • Magnitude: |CWT coefficients|'
        '   • Log scaling: log₁₀(magnitude + ε)'
        '   • Normalization: [0, 1] range'
        ''
        '4. Image Processing:'
        '   • Colormap: Jet (128 colors)'
        '   • Resize: 227×227 pixels'
        '   • Format: RGB uint8'
    };
    
    text(0.05, 0.95, pipeline_text, 'FontSize', 11, 'VerticalAlignment', 'top', ...
         'FontName', 'FixedWidth');
    
    sgtitle('Frequency Domain Analysis and Processing Pipeline', ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    % Save figure
    saveas(fig, fullfile(OUTPUT_PATH, 'Figure6_Frequency_Analysis.png'), 'png');
    saveas(fig, fullfile(OUTPUT_PATH, 'Figure6_Frequency_Analysis.pdf'), 'pdf');
    
    fprintf('Generated Figure 6: Frequency analysis\n');
end

function export_publication_figures()
    % Export all figures in publication-ready formats
    
    global OUTPUT_PATH
    
    % Publication settings
    set(0, 'DefaultFigureRenderer', 'painters'); % For vector graphics
    
    fprintf('\nExporting publication-ready figures...\n');
    
    % Create a summary document
    summary_file = fullfile(OUTPUT_PATH, 'Figure_Summary.txt');
    fid = fopen(summary_file, 'w');
    
    fprintf(fid, '=== PUBLICATION FIGURE SUMMARY ===\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now));
    
    fprintf(fid, 'AVAILABLE FIGURES:\n\n');
    
    fprintf(fid, 'Figure 1: Sample Scalograms\n');
    fprintf(fid, '- Shows representative scalograms from each cardiac condition\n');
    fprintf(fid, '- 3 samples per condition with proper time/frequency axes\n');
    fprintf(fid, '- Files: Figure1_Sample_Scalograms.png/.pdf\n\n');
    
    fprintf(fid, 'Figure 2: Condition Comparison\n');
    fprintf(fid, '- Side-by-side comparison of all four conditions\n');
    fprintf(fid, '- Color-coded borders for easy identification\n');
    fprintf(fid, '- Files: Figure2_Condition_Comparison.png/.pdf\n\n');
    
    fprintf(fid, 'Figure 3: Dataset Statistics\n');
    fprintf(fid, '- Comprehensive dataset overview with tables and charts\n');
    fprintf(fid, '- Age distribution analysis by condition\n');
    fprintf(fid, '- Files: Figure3_Dataset_Statistics.png/.pdf\n\n');
    
    fprintf(fid, 'Figure 4: Methodology\n');
    fprintf(fid, '- Complete CWT processing pipeline visualization\n');
    fprintf(fid, '- Shows original signal to final scalogram conversion\n');
    fprintf(fid, '- Files: Figure4_Methodology.png/.pdf\n\n');
    
    fprintf(fid, 'Figure 5: Age Analysis\n');
    fprintf(fid, '- Age-based distribution and sample analysis\n');
    fprintf(fid, '- Comparison across age groups\n');
    fprintf(fid, '- Files: Figure5_Age_Analysis.png/.pdf\n\n');
    
    fprintf(fid, 'Figure 6: Frequency Analysis\n');
    fprintf(fid, '- Technical details of frequency domain processing\n');
    fprintf(fid, '- CWT parameter effects and ECG frequency bands\n');
    fprintf(fid, '- Files: Figure6_Frequency_Analysis.png/.pdf\n\n');
    
    fprintf(fid, 'ADDITIONAL FILES:\n');
    fprintf(fid, '- dataset_statistics.txt: Detailed numerical statistics\n');
    fprintf(fid, '- Figure_Summary.txt: This summary document\n\n');
    
    fprintf(fid, 'RECOMMENDED USAGE:\n');
    fprintf(fid, '- Use PDF versions for publication (vector graphics)\n');
    fprintf(fid, '- Use PNG versions for presentations\n');
    fprintf(fid, '- All figures are publication-ready with proper fonts and sizing\n');
    
    fclose(fid);
    
    fprintf('All figures exported to: %s\n', OUTPUT_PATH);
    fprintf('Figure summary saved to: %s\n', summary_file);
end

% Helper function for Morlet wavelet (if not available)
function [psi, x] = morlet(lb, ub, n)
    % Generate Morlet wavelet
    if nargin < 3, n = 1000; end
    
    x = linspace(lb, ub, n);
    sigma = 1;
    psi = (1/sqrt(sigma*sqrt(pi))) * exp(1i*5*x) .* exp(-x.^2/(2*sigma^2));
end

% Run the utility
publication_visualization_utility();