function create_age_balanced_lead2_scalogram_dataset()
    % CREATE AGE BALANCED LEAD II SCALOGRAM DATASET
    % Creates age-balanced dataset from existing Lead II scalogram images
    % Balances across young_adult (18-40), middle_aged (41-65), elderly (66+)
    % for each class (SR, SB, AFIB)
    
    % Define paths
    source_dataset_path = 'C:\Users\henry\Downloads\ECG-Dx\Lead2_Scalogram_Dataset';
    output_dataset_path = 'C:\Users\henry\Downloads\ECG-Dx\Age_Balanced_Lead2_Scalogram_Dataset';
    
    % Create output directory
    if ~exist(output_dataset_path, 'dir')
        mkdir(output_dataset_path);
    end
    
    % Age group definitions (same as used in other scripts)
    age_groups = struct();
    age_groups.young_adult = [0, 40];    % 0-40 years
    age_groups.middle_aged = [41, 65];    % 41-65 years
    age_groups.elderly = [66, 100];       % 66+ years
    age_group_names = {'young_adult', 'middle_aged', 'elderly'};
    
    % Dataset parameters
    datasets = {'training', 'validation'};
    classes = {'SR', 'SB', 'AFIB'};
    
    % Initialize counters and statistics
    total_processed = 0;
    total_copied = 0;
    processing_errors = 0;
    age_balance_stats = struct();
    
    fprintf('=== AGE BALANCED LEAD II SCALOGRAM DATASET CREATOR ===\n');
    fprintf('Source dataset: %s\n', source_dataset_path);
    fprintf('Output dataset: %s\n', output_dataset_path);
    fprintf('Classes: %s\n', strjoin(classes, ', '));
    fprintf('Age groups: young_adult (18-40), middle_aged (41-65), elderly (66+)\n');
    fprintf('Expected files per class: Training=6476, Validation=1619\n\n');
    
    % Verify source dataset exists
    if ~exist(source_dataset_path, 'dir')
        error('Source dataset directory not found: %s', source_dataset_path);
    end
    
    % Create output directory structure
    create_output_directories(output_dataset_path, classes);
    
    % Process each class separately for age balancing
    for class_idx = 1:length(classes)
        class_name = classes{class_idx};
        fprintf('Processing %s class with age balancing...\n', class_name);
        
        % STEP 1: Collect all files from training and validation for this class
        all_files = [];
        
        for dataset_idx = 1:length(datasets)
            dataset_name = datasets{dataset_idx};
            input_dir = fullfile(source_dataset_path, dataset_name, class_name);
            
            if exist(input_dir, 'dir')
                png_files = dir(fullfile(input_dir, '*.png'));
                
                fprintf('  Found %d files in %s/%s\n', length(png_files), dataset_name, class_name);
                
                % Add full path and dataset info to each file
                for f = 1:length(png_files)
                    file_info = struct();
                    file_info.name = png_files(f).name;
                    file_info.path = fullfile(input_dir, png_files(f).name);
                    file_info.dataset = dataset_name;
                    all_files = [all_files, file_info];
                    total_processed = total_processed + 1;
                end
            else
                fprintf('  Warning: Directory not found: %s\n', input_dir);
            end
        end
        
        if isempty(all_files)
            fprintf('  Warning: No files found for %s class\n', class_name);
            continue;
        end
        
        fprintf('  Total files collected for %s: %d\n', class_name, length(all_files));
        
        % STEP 2: Extract ages and group by age categories
        age_grouped_files = struct();
        for age_group_idx = 1:length(age_group_names)
            age_grouped_files.(age_group_names{age_group_idx}) = [];
        end
        
        ages_extracted = [];
        valid_files = [];
        
        for f = 1:length(all_files)
            % Extract age from filename using regex
            filename = all_files(f).name;
            age_match = regexp(filename, '_age(\d+)_', 'tokens');
            
            if ~isempty(age_match)
                age = str2double(age_match{1}{1});
                ages_extracted = [ages_extracted, age];
                valid_files = [valid_files, all_files(f)];
                
                % Assign to age group
                assigned = false;
                for age_group_idx = 1:length(age_group_names)
                    age_group_name = age_group_names{age_group_idx};
                    age_range = age_groups.(age_group_name);
                    
                    if age >= age_range(1) && age <= age_range(2)
                        age_grouped_files.(age_group_name) = [age_grouped_files.(age_group_name), all_files(f)];
                        assigned = true;
                        break;
                    end
                end
                
                if ~assigned
                    fprintf('    Warning: Age %d not assigned to any group for %s\n', age, filename);
                    processing_errors = processing_errors + 1;
                end
            else
                fprintf('    Warning: Could not extract age from filename: %s\n', filename);
                processing_errors = processing_errors + 1;
            end
        end
        
        % STEP 3: Display age distribution
        fprintf('  Age distribution for %s:\n', class_name);
        age_counts = zeros(1, length(age_group_names));
        
        for age_group_idx = 1:length(age_group_names)
            age_group_name = age_group_names{age_group_idx};
            count = length(age_grouped_files.(age_group_name));
            age_counts(age_group_idx) = count;
            age_range = age_groups.(age_group_name);
            
            fprintf('    %s (%d-%d years): %d files\n', age_group_name, age_range(1), age_range(2), count);
        end
        
        % Calculate age statistics
        if ~isempty(ages_extracted)
            fprintf('    Age statistics: mean=%.1f, std=%.1f, range=%d-%d years\n', ...
                   mean(ages_extracted), std(ages_extracted), min(ages_extracted), max(ages_extracted));
        end
        
        % STEP 4: Create balanced subset
        min_count = min(age_counts(age_counts > 0)); % Minimum non-zero count
        
        if min_count == 0
            fprintf('    Error: One or more age groups have no samples for %s\n', class_name);
            continue;
        end
        
        fprintf('    Creating balanced subset with %d samples per age group\n', min_count);
        fprintf('    Total balanced samples for %s: %d\n', class_name, min_count * length(age_group_names));
        
        balanced_files = [];
        rng(42); % Set seed for reproducibility
        
        for age_group_idx = 1:length(age_group_names)
            age_group_name = age_group_names{age_group_idx};
            group_files = age_grouped_files.(age_group_name);
            
            if length(group_files) >= min_count
                % Randomly select min_count files from this age group
                selected_indices = randperm(length(group_files), min_count);
                selected_files = group_files(selected_indices);
                balanced_files = [balanced_files, selected_files];
                
                fprintf('    Selected %d files from %s age group\n', min_count, age_group_name);
            else
                fprintf('    Warning: Insufficient files in %s age group (%d < %d)\n', ...
                       age_group_name, length(group_files), min_count);
            end
        end
        
        % STEP 5: Copy balanced files to output directory
        output_dir = fullfile(output_dataset_path, class_name);
        
        for file_idx = 1:length(balanced_files)
            file_info = balanced_files(file_idx);
            source_path = file_info.path;
            
            % Create output filename with age group info
            [~, base_name, ext] = fileparts(file_info.name);
            
            % Extract age for age group labeling
            age_match = regexp(file_info.name, '_age(\d+)_', 'tokens');
            if ~isempty(age_match)
                age = str2double(age_match{1}{1});
                
                % Determine age group
                age_group_label = '';
                for age_group_idx = 1:length(age_group_names)
                    age_group_name = age_group_names{age_group_idx};
                    age_range = age_groups.(age_group_name);
                    
                    if age >= age_range(1) && age <= age_range(2)
                        age_group_label = age_group_name;
                        break;
                    end
                end
                
                % Create new filename with age group info
                output_filename = sprintf('%s_%s%s', base_name, age_group_label, ext);
            else
                output_filename = file_info.name;
            end
            
            output_path = fullfile(output_dir, output_filename);
            
            try
                copyfile(source_path, output_path);
                total_copied = total_copied + 1;
            catch ME
                fprintf('    Error copying %s: %s\n', file_info.name, ME.message);
                processing_errors = processing_errors + 1;
            end
        end
        
        % Store age balance statistics
        stats_key = class_name;
        age_balance_stats.(stats_key) = struct();
        age_balance_stats.(stats_key).original_counts = age_counts;
        age_balance_stats.(stats_key).balanced_count = min_count;
        age_balance_stats.(stats_key).total_ages = ages_extracted;
        age_balance_stats.(stats_key).total_balanced = length(balanced_files);
        
        fprintf('  Completed %s: %d age-balanced files copied\n\n', class_name, length(balanced_files));
    end
    
    % Final summary
    fprintf('=== AGE BALANCED DATASET CREATION SUMMARY ===\n');
    fprintf('Total original files processed: %d\n', total_processed);
    fprintf('Total balanced files copied: %d\n', total_copied);
    fprintf('Processing errors: %d\n', processing_errors);
    if total_processed > 0
        fprintf('Success rate: %.1f%%\n', (total_copied/total_processed)*100);
    end
    fprintf('Output directory: %s\n', output_dataset_path);
    
    % Display detailed age balancing summary
    display_age_balancing_summary(age_balance_stats, age_groups);
    
    % Generate comprehensive report
    generate_age_balance_report(output_dataset_path, age_balance_stats, age_groups, ...
                               total_processed, total_copied, processing_errors);
    
    % Generate visualization
    generate_age_balance_visualization(output_dataset_path, age_balance_stats, age_groups);
    
    % Verify output dataset
    verify_age_balanced_dataset(output_dataset_path, classes);
    
    fprintf('\nAge-balanced Lead II scalogram dataset creation complete!\n');
    fprintf('Dataset ready for age-unbiased model training and evaluation.\n');
end

function create_output_directories(output_path, classes)
    % Create output directory structure
    
    for i = 1:length(classes)
        class_dir = fullfile(output_path, classes{i});
        if ~exist(class_dir, 'dir')
            mkdir(class_dir);
        end
    end
end

function display_age_balancing_summary(age_balance_stats, age_groups)
    % Display detailed age balancing summary
    
    fprintf('\n=== DETAILED AGE BALANCING SUMMARY ===\n');
    
    class_names = fieldnames(age_balance_stats);
    age_group_names = fieldnames(age_groups);
    
    % Create summary table
    fprintf('%-8s %-12s %-12s %-12s %-12s %-12s\n', 'Class', 'Young Adult', 'Middle Aged', 'Elderly', 'Balanced', 'Total');
    fprintf('%-8s %-12s %-12s %-12s %-12s %-12s\n', '-----', '-----------', '-----------', '-------', '--------', '-----');
    
    total_balanced_all = 0;
    
    for i = 1:length(class_names)
        class_name = class_names{i};
        stats = age_balance_stats.(class_name);
        
        fprintf('%-8s %-12d %-12d %-12d %-12d %-12d\n', class_name, ...
               stats.original_counts(1), stats.original_counts(2), stats.original_counts(3), ...
               stats.balanced_count, stats.total_balanced);
        
        total_balanced_all = total_balanced_all + stats.total_balanced;
    end
    
    fprintf('%-8s %-12s %-12s %-12s %-12s %-12d\n', 'TOTAL', '', '', '', '', total_balanced_all);
    
    % Age group definitions reminder
    fprintf('\nAge Group Definitions:\n');
    for i = 1:length(age_group_names)
        age_group_name = age_group_names{i};
        age_range = age_groups.(age_group_name);
        fprintf('  %s: %d-%d years\n', age_group_name, age_range(1), age_range(2));
    end
    
    % Balance quality metrics
    fprintf('\nBalance Quality Metrics:\n');
    for i = 1:length(class_names)
        class_name = class_names{i};
        stats = age_balance_stats.(class_name);
        
        % Calculate coefficient of variation (CV) for original vs balanced
        original_cv = std(stats.original_counts) / mean(stats.original_counts);
        balanced_cv = 0; % Perfect balance
        
        fprintf('  %s: Original CV=%.3f, Balanced CV=%.3f (improvement: %.1f%%)\n', ...
               class_name, original_cv, balanced_cv, (1-balanced_cv/original_cv)*100);
    end
end

function generate_age_balance_report(output_path, age_balance_stats, age_groups, ...
                                   total_processed, total_copied, processing_errors)
    % Generate comprehensive age balance report
    
    report_file = fullfile(output_path, 'age_balance_report.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, '=== AGE BALANCED LEAD II SCALOGRAM DATASET REPORT ===\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now));
    
    fprintf(fid, 'DATASET OVERVIEW:\n');
    fprintf(fid, 'Source: Lead II scalogram images from ECG-Dx dataset\n');
    fprintf(fid, 'Classes: SR (Sinus Rhythm), SB (Sinus Bradycardia), AFIB (Atrial Fibrillation)\n');
    fprintf(fid, 'Image format: 227x227 RGB scalograms (4-second Lead II ECG)\n');
    fprintf(fid, 'Age balancing: Equal representation across age groups\n');
    fprintf(fid, 'Output directory: %s\n\n', output_path);
    
    fprintf(fid, 'AGE GROUP DEFINITIONS:\n');
    age_group_names = fieldnames(age_groups);
    for i = 1:length(age_group_names)
        age_group_name = age_group_names{i};
        age_range = age_groups.(age_group_name);
        fprintf(fid, '- %s: %d-%d years\n', age_group_name, age_range(1), age_range(2));
    end
    fprintf(fid, '\n');
    
    fprintf(fid, 'PROCESSING STATISTICS:\n');
    fprintf(fid, 'Total original files processed: %d\n', total_processed);
    fprintf(fid, 'Total balanced files created: %d\n', total_copied);
    fprintf(fid, 'Processing errors: %d\n', processing_errors);
    if total_processed > 0
        fprintf(fid, 'Success rate: %.1f%%\n', (total_copied/total_processed)*100);
    end
    fprintf(fid, '\n');
    
    fprintf(fid, 'AGE BALANCING RESULTS BY CLASS:\n');
    class_names = fieldnames(age_balance_stats);
    
    for i = 1:length(class_names)
        class_name = class_names{i};
        stats = age_balance_stats.(class_name);
        
        fprintf(fid, '%s Class:\n', class_name);
        fprintf(fid, '  Original distribution:\n');
        fprintf(fid, '    Young Adult (18-40): %d files\n', stats.original_counts(1));
        fprintf(fid, '    Middle Aged (41-65): %d files\n', stats.original_counts(2));
        fprintf(fid, '    Elderly (66+): %d files\n', stats.original_counts(3));
        fprintf(fid, '  Balanced dataset: %d files per age group\n', stats.balanced_count);
        fprintf(fid, '  Total balanced files: %d\n', stats.total_balanced);
        
        if ~isempty(stats.total_ages)
            fprintf(fid, '  Age statistics: mean=%.1f±%.1f years, range=%d-%d\n', ...
                   mean(stats.total_ages), std(stats.total_ages), ...
                   min(stats.total_ages), max(stats.total_ages));
        end
        
        % Balance improvement
        original_cv = std(stats.original_counts) / mean(stats.original_counts);
        fprintf(fid, '  Balance improvement: CV reduced from %.3f to 0.000 (perfect balance)\n', original_cv);
        fprintf(fid, '\n');
    end
    
    fprintf(fid, 'METHODOLOGY:\n');
    fprintf(fid, '1. Collected all files from original training and validation sets\n');
    fprintf(fid, '2. Extracted age information from filenames using regex pattern "_age(\\d+)_"\n');
    fprintf(fid, '3. Grouped files into age categories (young_adult, middle_aged, elderly)\n');
    fprintf(fid, '4. Identified minimum count across age groups for each class\n');
    fprintf(fid, '5. Randomly sampled equal numbers from each age group (seed=42)\n');
    fprintf(fid, '6. Created balanced dataset with equal age representation\n');
    fprintf(fid, '7. Added age group labels to output filenames for traceability\n\n');
    
    fprintf(fid, 'FILENAME CONVENTION:\n');
    fprintf(fid, 'Format: [PATIENT_ID]_age[AGE]_Lead2_4sec_[AGE_GROUP].png\n');
    fprintf(fid, 'Examples:\n');
    fprintf(fid, '- JS44163_age25_Lead2_4sec_young_adult.png\n');
    fprintf(fid, '- TR09173_age55_Lead2_4sec_middle_aged.png\n');
    fprintf(fid, '- AM12456_age75_Lead2_4sec_elderly.png\n\n');
    
    fprintf(fid, 'ADVANTAGES OF AGE BALANCING:\n');
    fprintf(fid, '• Eliminates age bias in model training and evaluation\n');
    fprintf(fid, '• Enables age-stratified performance analysis\n');
    fprintf(fid, '• Supports regulatory demographic balance requirements\n');
    fprintf(fid, '• Facilitates clinical validation across age groups\n');
    fprintf(fid, '• Improves model generalizability to all populations\n');
    fprintf(fid, '• Enables age-specific performance monitoring\n\n');
    
    fprintf(fid, 'RESEARCH APPLICATIONS:\n');
    fprintf(fid, '• Age-unbiased ECG classification model development\n');
    fprintf(fid, '• Demographic-specific performance validation\n');
    fprintf(fid, '• Clinical deployment readiness assessment\n');
    fprintf(fid, '• Regulatory submission with demographic compliance\n');
    fprintf(fid, '• Age-inclusive AI system validation\n');
    fprintf(fid, '• Evidence-based clinical decision support\n\n');
    
    fprintf(fid, 'EXPECTED CLINICAL PERFORMANCE BY AGE GROUP:\n');
    fprintf(fid, 'Young Adults (18-40 years):\n');
    fprintf(fid, '• Expected high model accuracy due to cleaner physiological signals\n');
    fprintf(fid, '• Lower baseline heart rate variability\n');
    fprintf(fid, '• Fewer comorbidities affecting ECG patterns\n\n');
    
    fprintf(fid, 'Middle Aged (41-65 years):\n');
    fprintf(fid, '• Moderate model performance with potential age-related changes\n');
    fprintf(fid, '• Increased prevalence of cardiovascular risk factors\n');
    fprintf(fid, '• May require age-specific decision thresholds\n\n');
    
    fprintf(fid, 'Elderly (66+ years):\n');
    fprintf(fid, '• May show reduced model performance due to physiological complexity\n');
    fprintf(fid, '• Higher prevalence of multiple cardiac conditions\n');
    fprintf(fid, '• Requires careful clinical interpretation and validation\n\n');
    
    fprintf(fid, 'VALIDATION RECOMMENDATIONS:\n');
    fprintf(fid, '1. Train models using age-balanced dataset\n');
    fprintf(fid, '2. Evaluate performance within and across age groups\n');
    fprintf(fid, '3. Report age-stratified sensitivity and specificity\n');
    fprintf(fid, '4. Validate with independent age-balanced test sets\n');
    fprintf(fid, '5. Document age-specific performance thresholds\n');
    fprintf(fid, '6. Implement age-aware quality monitoring in deployment\n\n');
    
    fprintf(fid, 'REGULATORY CONSIDERATIONS:\n');
    fprintf(fid, '• Demonstrates commitment to demographic inclusion in AI development\n');
    fprintf(fid, '• Supports FDA guidance on AI/ML-based medical devices\n');
    fprintf(fid, '• Enables evidence-based claims about population coverage\n');
    fprintf(fid, '• Facilitates post-market surveillance across age groups\n');
    fprintf(fid, '• Supports health equity requirements in clinical deployment\n');
    
    fclose(fid);
    
    fprintf('Age balance report saved to: %s\n', report_file);
end

function generate_age_balance_visualization(output_path, age_balance_stats, age_groups)
    % Generate visualization of age balancing results
    
    fprintf('Generating age balance visualization...\n');
    
    % Create figure
    fig = figure('Position', [100, 100, 1400, 1000]);
    
    class_names = fieldnames(age_balance_stats);
    age_group_names = fieldnames(age_groups);
    
    % Subplot 1: Original distribution
    subplot(2, 3, 1);
    
    original_data = zeros(length(class_names), length(age_group_names));
    for i = 1:length(class_names)
        stats = age_balance_stats.(class_names{i});
        original_data(i, :) = stats.original_counts;
    end
    
    b1 = bar(original_data, 'grouped');
    b1(1).DisplayName = 'Young Adult (18-40)';
    b1(2).DisplayName = 'Middle Aged (41-65)';
    b1(3).DisplayName = 'Elderly (66+)';
    
    set(gca, 'XTick', 1:length(class_names), 'XTickLabel', class_names);
    xlabel('ECG Class');
    ylabel('Number of Files');
    title('Original Age Distribution', 'FontWeight', 'bold', 'FontSize', 14);
    legend('Location', 'best');
    grid on; grid minor;
    
    % Subplot 2: Balanced distribution
    subplot(2, 3, 2);
    
    balanced_data = zeros(length(class_names), length(age_group_names));
    for i = 1:length(class_names)
        stats = age_balance_stats.(class_names{i});
        balanced_data(i, :) = stats.balanced_count;
    end
    
    b2 = bar(balanced_data, 'grouped');
    colors = [0.3, 0.7, 0.3; 0.3, 0.7, 0.3; 0.3, 0.7, 0.3];
    for j = 1:length(b2)
        b2(j).FaceColor = colors(j, :);
    end
    
    set(gca, 'XTick', 1:length(class_names), 'XTickLabel', class_names);
    xlabel('ECG Class');
    ylabel('Number of Files');
    title('Age-Balanced Distribution', 'FontWeight', 'bold', 'FontSize', 14);
    grid on; grid minor;
    
    % Subplot 3: Balance improvement (CV reduction)
    subplot(2, 3, 3);
    
    cv_original = zeros(1, length(class_names));
    cv_balanced = zeros(1, length(class_names));
    
    for i = 1:length(class_names)
        stats = age_balance_stats.(class_names{i});
        cv_original(i) = std(stats.original_counts) / mean(stats.original_counts);
        cv_balanced(i) = 0; % Perfect balance
    end
    
    x_pos = 1:length(class_names);
    bar_width = 0.35;
    
    bar(x_pos - bar_width/2, cv_original, bar_width, 'FaceColor', [0.8, 0.3, 0.3], 'DisplayName', 'Original');
    hold on;
    bar(x_pos + bar_width/2, cv_balanced, bar_width, 'FaceColor', [0.3, 0.8, 0.3], 'DisplayName', 'Balanced');
    
    set(gca, 'XTick', x_pos, 'XTickLabel', class_names);
    xlabel('ECG Class');
    ylabel('Coefficient of Variation');
    title('Balance Improvement', 'FontWeight', 'bold', 'FontSize', 14);
    legend('Location', 'best');
    grid on; grid minor;
    hold off;
    
    % Subplot 4: Age statistics by class
    subplot(2, 3, 4);
    
    age_means = zeros(1, length(class_names));
    age_stds = zeros(1, length(class_names));
    
    for i = 1:length(class_names)
        stats = age_balance_stats.(class_names{i});
        if ~isempty(stats.total_ages)
            age_means(i) = mean(stats.total_ages);
            age_stds(i) = std(stats.total_ages);
        end
    end
    
    errorbar(1:length(class_names), age_means, age_stds, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    set(gca, 'XTick', 1:length(class_names), 'XTickLabel', class_names);
    xlabel('ECG Class');
    ylabel('Age (years)');
    title('Age Statistics by Class', 'FontWeight', 'bold', 'FontSize', 14);
    grid on; grid minor;
    
    % Subplot 5: Sample size summary
    subplot(2, 3, [5, 6]);
    
    summary_text = {
        '\bf{Age Balancing Summary}'
        ''
        '\bf{Original Dataset:}'
    };
    
    for i = 1:length(class_names)
        stats = age_balance_stats.(class_names{i});
        summary_text{end+1} = sprintf('%s: Total=%d (Young=%d, Middle=%d, Elderly=%d)', ...
                                     class_names{i}, sum(stats.original_counts), ...
                                     stats.original_counts(1), stats.original_counts(2), stats.original_counts(3));
    end
    
    summary_text{end+1} = '';
    summary_text{end+1} = '\bf{Balanced Dataset:}';
    
    for i = 1:length(class_names)
        stats = age_balance_stats.(class_names{i});
        summary_text{end+1} = sprintf('%s: %d per age group (%d total)', ...
                                     class_names{i}, stats.balanced_count, stats.total_balanced);
    end
    
    summary_text{end+1} = '';
    summary_text{end+1} = '\bf{Benefits of Age Balancing:}';
    summary_text{end+1} = '• Eliminates age bias in model training';
    summary_text{end+1} = '• Enables age-stratified performance analysis';
    summary_text{end+1} = '• Supports regulatory demographic requirements';
    summary_text{end+1} = '• Facilitates clinical validation across ages';
    summary_text{end+1} = '• Improves generalizability to all populations';
    summary_text{end+1} = '';
    summary_text{end+1} = '\bf{Age Group Definitions:}';
    
    for i = 1:length(age_group_names)
        age_group_name = age_group_names{i};
        age_range = age_groups.(age_group_name);
        summary_text{end+1} = sprintf('• %s: %d-%d years', age_group_name, age_range(1), age_range(2));
    end
    
    axis off;
    text(0.05, 0.95, summary_text, 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'FontName', 'Arial', ...
         'FontSize', 10, 'Interpreter', 'tex');
    
    sgtitle('Age-Balanced Lead II Scalogram Dataset Analysis', ...
            'FontWeight', 'bold', 'FontSize', 16);
    
    % Save figure
    figures_path = fullfile(output_path, 'Age_Balance_Analysis');
    if ~exist(figures_path, 'dir')
        mkdir(figures_path);
    end
    
    saveas(fig, fullfile(figures_path, 'Age_Balance_Summary.png'), 'png');
    saveas(fig, fullfile(figures_path, 'Age_Balance_Summary.fig'), 'fig');
    
    fprintf('Age balance visualization saved to: %s\n', figures_path);
end

function verify_age_balanced_dataset(output_path, classes)
    % Verify the created age-balanced dataset
    
    fprintf('\n=== DATASET VERIFICATION ===\n');
    
    total_files = 0;
    
    for i = 1:length(classes)
        class_name = classes{i};
        class_dir = fullfile(output_path, class_name);
        
        if exist(class_dir, 'dir')
            png_files = dir(fullfile(class_dir, '*.png'));
            count = length(png_files);
            total_files = total_files + count;
            
            fprintf('%s: %d files\n', class_name, count);
            
            % Check age group distribution in filenames
            young_count = length(dir(fullfile(class_dir, '*young_adult.png')));
            middle_count = length(dir(fullfile(class_dir, '*middle_aged.png')));
            elderly_count = length(dir(fullfile(class_dir, '*elderly.png')));
            
            fprintf('  Age groups: Young=%d, Middle=%d, Elderly=%d\n', ...
                   young_count, middle_count, elderly_count);
            
            if young_count == middle_count && middle_count == elderly_count
                fprintf('  ✓ Perfect age balance achieved\n');
            else
                fprintf('  ⚠ Age imbalance detected\n');
            end
        else
            fprintf('%s: Directory not found\n', class_name);
        end
    end
    
    fprintf('\nTotal files in age-balanced dataset: %d\n', total_files);
    fprintf('Average files per class: %.1f\n', total_files / length(classes));
end

% Execute the function
create_age_balanced_lead2_scalogram_dataset();