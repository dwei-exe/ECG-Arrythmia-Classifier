function organize_ecg_database()
    % Organize ECG database into clinical groups with age labeling
    % Creates balanced training/validation datasets
    
    % Define paths
    source_path = 'C:\Users\henry\Downloads\ECG-Dx\Raw Download\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords';
    output_path = 'C:\Users\henry\Downloads\ECG-Dx\Organized_Dataset';
    
    % SNOMED CT codes for each group (from CSV analysis)
    snomed_groups = containers.Map();
    snomed_groups('SB') = [426177001];  % Sinus Bradycardia
    snomed_groups('AFIB') = [164889003, 164890007];  % Atrial Fibrillation + Flutter
    snomed_groups('SR') = [426783006];  % Sinus Rhythm
    
    fprintf('=== ECG DATABASE ORGANIZATION ===\n');
    fprintf('Source path: %s\n', source_path);
    fprintf('Output path: %s\n', output_path);
    fprintf('Target groups: SB, AFIB, SR\n\n');
    
    % Create output directories
    create_output_directories(output_path);
    
    % Initialize counters
    stats = struct();
    stats.total_processed = 0;
    stats.SB = 0;
    stats.AFIB = 0;
    stats.SR = 0;
    stats.unclassified = 0;
    stats.errors = 0;
    
    % Patient data storage
    patients = struct();
    patients.SB = {};
    patients.AFIB = {};
    patients.SR = {};
    
    % Process all folders and subfolders
    fprintf('Scanning database and extracting patient information...\n');
    
    % Get all first-level directories (should be 46 folders)
    first_level_dirs = dir(source_path);
    first_level_dirs = first_level_dirs([first_level_dirs.isdir] & ~ismember({first_level_dirs.name}, {'.', '..'}));
    
    fprintf('Found %d first-level directories\n', length(first_level_dirs));
    
    for i = 1:length(first_level_dirs)
        first_level_path = fullfile(source_path, first_level_dirs(i).name);
        
        % Get second-level directories (should be 10 subfolders each)
        second_level_dirs = dir(first_level_path);
        second_level_dirs = second_level_dirs([second_level_dirs.isdir] & ~ismember({second_level_dirs.name}, {'.', '..'}));
        
        for j = 1:length(second_level_dirs)
            second_level_path = fullfile(first_level_path, second_level_dirs(j).name);
            
            % Process all .hea files in this subfolder
            hea_files = dir(fullfile(second_level_path, '*.hea'));
            
            fprintf('Processing %s/%s: %d patients\n', first_level_dirs(i).name, second_level_dirs(j).name, length(hea_files));
            
            for k = 1:length(hea_files)
                try
                    % Parse header file
                    hea_file_path = fullfile(second_level_path, hea_files(k).name);
                    patient_info = parse_header_file(hea_file_path);
                    
                    if isempty(patient_info)
                        stats.errors = stats.errors + 1;
                        continue;
                    end
                    
                    stats.total_processed = stats.total_processed + 1;
                    
                    % Classify patient based on diagnosis codes
                    group = classify_patient(patient_info.dx_codes, snomed_groups);
                    
                    if ~isempty(group)
                        % Add patient to appropriate group
                        patient_info.source_folder = second_level_path;
                        patient_info.base_name = hea_files(k).name(1:end-4); % Remove .hea extension
                        
                        patients.(group){end+1} = patient_info;
                        stats.(group) = stats.(group) + 1;
                    else
                        stats.unclassified = stats.unclassified + 1;
                    end
                    
                catch ME
                    fprintf('Error processing %s: %s\n', hea_files(k).name, ME.message);
                    stats.errors = stats.errors + 1;
                end
            end
        end
        
        % Progress update
        if mod(i, 5) == 0
            fprintf('Completed %d/%d first-level directories\n', i, length(first_level_dirs));
        end
    end
    
    % Display classification statistics
    display_statistics(stats);
    
    % Create balanced dataset
    fprintf('\n=== CREATING BALANCED DATASET ===\n');
    create_balanced_dataset(patients, output_path, stats);
    
    % Generate final report
    generate_organization_report(output_path, stats, patients);
    
    fprintf('\nOrganization complete! Check the output directory: %s\n', output_path);
end

function create_output_directories(output_path)
    % Create the organized dataset directory structure
    
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end
    
    datasets = {'training', 'validation'};
    groups = {'SB', 'AFIB', 'SR'};
    
    for i = 1:length(datasets)
        for j = 1:length(groups)
            dir_path = fullfile(output_path, datasets{i}, groups{j});
            if ~exist(dir_path, 'dir')
                mkdir(dir_path);
            end
        end
    end
end

function patient_info = parse_header_file(hea_file_path)
    % Parse .hea file to extract patient information
    
    patient_info = [];
    
    try
        % Read header file
        fid = fopen(hea_file_path, 'r');
        if fid == -1
            return;
        end
        
        content = textscan(fid, '%s', 'Delimiter', '\n');
        fclose(fid);
        
        lines = content{1};
        
        % Initialize patient info structure
        patient_info = struct();
        patient_info.age = [];
        patient_info.sex = '';
        patient_info.dx_codes = [];
        
        % Parse each line
        for i = 1:length(lines)
            line = lines{i};
            
            if startsWith(line, '#Age:')
                % Extract age
                age_str = strtrim(strrep(line, '#Age:', ''));
                patient_info.age = str2double(age_str);
                
            elseif startsWith(line, '#Sex:')
                % Extract sex
                patient_info.sex = strtrim(strrep(line, '#Sex:', ''));
                
            elseif startsWith(line, '#Dx:')
                % Extract diagnosis codes
                dx_str = strtrim(strrep(line, '#Dx:', ''));
                if ~isempty(dx_str)
                    % Split by comma and convert to numbers
                    dx_codes_str = strsplit(dx_str, ',');
                    dx_codes = [];
                    for j = 1:length(dx_codes_str)
                        code = str2double(strtrim(dx_codes_str{j}));
                        if ~isnan(code)
                            dx_codes(end+1) = code;
                        end
                    end
                    patient_info.dx_codes = dx_codes;
                end
            end
        end
        
        % Validate required fields
        if isempty(patient_info.age) || isnan(patient_info.age) || isempty(patient_info.dx_codes)
            patient_info = [];
        end
        
    catch ME
        patient_info = [];
    end
end

function group = classify_patient(dx_codes, snomed_groups)
    % Classify patient into one of the three groups based on diagnosis codes
    
    group = '';
    
    if isempty(dx_codes)
        return;
    end
    
    % Check each group for matching codes
    group_names = keys(snomed_groups);
    
    for i = 1:length(group_names)
        group_name = group_names{i};
        target_codes = snomed_groups(group_name);
        
        % Check if any diagnosis code matches this group
        if any(ismember(dx_codes, target_codes))
            group = group_name;
            return;
        end
    end
end

function display_statistics(stats)
    % Display classification statistics
    
    fprintf('\n=== CLASSIFICATION STATISTICS ===\n');
    fprintf('Total files processed: %d\n', stats.total_processed);
    fprintf('Successfully classified:\n');
    fprintf('  SB (Sinus Bradycardia): %d\n', stats.SB);
    fprintf('  AFIB (Atrial Fibrillation/Flutter): %d\n', stats.AFIB);
    fprintf('  SR (Sinus Rhythm): %d\n', stats.SR);
    fprintf('Unclassified: %d\n', stats.unclassified);
    fprintf('Errors: %d\n', stats.errors);
    
    total_classified = stats.SB + stats.AFIB + stats.SR;
    if total_classified > 0
        fprintf('\nDistribution:\n');
        fprintf('  SB: %.1f%%\n', (stats.SB / total_classified) * 100);
        fprintf('  AFIB: %.1f%%\n', (stats.AFIB / total_classified) * 100);
        fprintf('  SR: %.1f%%\n', (stats.SR / total_classified) * 100);
    end
end

function create_balanced_dataset(patients, output_path, stats)
    % Create balanced training/validation datasets
    
    % Find the minimum count among the three groups
    group_counts = [stats.SB, stats.AFIB, stats.SR];
    min_count = min(group_counts);
    
    fprintf('Minimum group size: %d patients\n', min_count);
    fprintf('Creating balanced dataset with %d patients per group\n', min_count);
    
    % Training/validation split
    train_ratio = 0.8;
    train_count = floor(min_count * train_ratio);
    val_count = min_count - train_count;
    
    fprintf('Training set: %d patients per group\n', train_count);
    fprintf('Validation set: %d patients per group\n', val_count);
    
    group_names = {'SB', 'AFIB', 'SR'};
    
    for i = 1:length(group_names)
        group_name = group_names{i};
        group_patients = patients.(group_name);
        
        fprintf('\nProcessing %s group (%d patients available)...\n', group_name, length(group_patients));
        
        % Randomly shuffle patients
        rng(42); % Set seed for reproducibility
        shuffled_indices = randperm(length(group_patients));
        
        % Select balanced subset
        selected_patients = group_patients(shuffled_indices(1:min_count));
        
        % Split into training and validation
        train_patients = selected_patients(1:train_count);
        val_patients = selected_patients(train_count+1:end);
        
        % Copy training files
        copy_patient_files(train_patients, output_path, 'training', group_name);
        
        % Copy validation files
        copy_patient_files(val_patients, output_path, 'validation', group_name);
    end
end

function copy_patient_files(patients, output_path, dataset_type, group_name)
    % Copy patient files to organized directory with age labeling
    
    target_dir = fullfile(output_path, dataset_type, group_name);
    
    for i = 1:length(patients)
        patient = patients{i};
        
        % Create age-labeled filename
        age_label = sprintf('age%d', patient.age);
        new_base_name = sprintf('%s_%s', patient.base_name, age_label);
        
        % Source files
        source_hea = fullfile(patient.source_folder, [patient.base_name '.hea']);
        source_mat = fullfile(patient.source_folder, [patient.base_name '.mat']);
        
        % Target files
        target_hea = fullfile(target_dir, [new_base_name '.hea']);
        target_mat = fullfile(target_dir, [new_base_name '.mat']);
        
        % Copy files
        try
            copyfile(source_hea, target_hea);
            copyfile(source_mat, target_mat);
        catch ME
            fprintf('Error copying files for %s: %s\n', patient.base_name, ME.message);
        end
    end
    
    fprintf('  Copied %d patients to %s/%s\n', length(patients), dataset_type, group_name);
end

function generate_organization_report(output_path, stats, patients)
    % Generate comprehensive organization report
    
    report_file = fullfile(output_path, 'organization_report.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, '=== ECG DATABASE ORGANIZATION REPORT ===\n');
    fprintf(fid, 'Date: %s\n\n', datestr(now));
    
    fprintf(fid, 'ORIGINAL DATABASE STATISTICS:\n');
    fprintf(fid, 'Total files processed: %d\n', stats.total_processed);
    fprintf(fid, 'SB (Sinus Bradycardia): %d\n', stats.SB);
    fprintf(fid, 'AFIB (Atrial Fibrillation/Flutter): %d\n', stats.AFIB);
    fprintf(fid, 'SR (Sinus Rhythm): %d\n', stats.SR);
    fprintf(fid, 'Unclassified: %d\n', stats.unclassified);
    fprintf(fid, 'Errors: %d\n\n', stats.errors);
    
    fprintf(fid, 'SNOMED CT CODE MAPPING:\n');
    fprintf(fid, 'SB: 426177001 (Sinus Bradycardia)\n');
    fprintf(fid, 'AFIB: 164889003 (Atrial Fibrillation), 164890007 (Atrial Flutter)\n');
    fprintf(fid, 'SR: 426783006 (Sinus Rhythm)\n\n');
    
    % Balanced dataset info
    group_counts = [stats.SB, stats.AFIB, stats.SR];
    min_count = min(group_counts);
    train_count = floor(min_count * 0.8);
    val_count = min_count - train_count;
    
    fprintf(fid, 'BALANCED DATASET:\n');
    fprintf(fid, 'Patients per group: %d\n', min_count);
    fprintf(fid, 'Training set: %d patients per group (%d total)\n', train_count, train_count * 3);
    fprintf(fid, 'Validation set: %d patients per group (%d total)\n', val_count, val_count * 3);
    fprintf(fid, 'Total organized patients: %d\n\n', min_count * 3);
    
    fprintf(fid, 'DIRECTORY STRUCTURE:\n');
    fprintf(fid, 'training/\n');
    fprintf(fid, '  SB/     - %d patients\n', train_count);
    fprintf(fid, '  AFIB/   - %d patients\n', train_count);
    fprintf(fid, '  SR/     - %d patients\n', train_count);
    fprintf(fid, 'validation/\n');
    fprintf(fid, '  SB/     - %d patients\n', val_count);
    fprintf(fid, '  AFIB/   - %d patients\n', val_count);
    fprintf(fid, '  SR/     - %d patients\n', val_count);
    
    fprintf(fid, '\nFILE NAMING CONVENTION:\n');
    fprintf(fid, 'Format: [ORIGINAL_ID]_age[AGE].[ext]\n');
    fprintf(fid, 'Example: JS44176_age62.mat, JS44176_age62.hea\n');
    
    % Age distribution analysis
    fprintf(fid, '\nAGE DISTRIBUTION ANALYSIS:\n');
    group_names = {'SB', 'AFIB', 'SR'};
    for i = 1:length(group_names)
        group_name = group_names{i};
        group_patients = patients.(group_name);
        
        if ~isempty(group_patients)
            ages = cellfun(@(p) p.age, group_patients);
            fprintf(fid, '%s: Mean age = %.1f ± %.1f (range: %d-%d)\n', ...
                    group_name, mean(ages), std(ages), min(ages), max(ages));
        end
    end
    
    fclose(fid);
    
    fprintf('Organization report saved to: %s\n', report_file);
end

function analyze_age_distributions(patients)
    % Analyze and visualize age distributions across groups
    
    fprintf('\n=== AGE DISTRIBUTION ANALYSIS ===\n');
    
    group_names = {'SB', 'AFIB', 'SR'};
    colors = {'blue', 'red', 'green'};
    
    figure('Position', [100, 100, 1200, 400]);
    
    for i = 1:length(group_names)
        group_name = group_names{i};
        group_patients = patients.(group_name);
        
        if ~isempty(group_patients)
            ages = cellfun(@(p) p.age, group_patients);
            
            fprintf('%s Group:\n', group_name);
            fprintf('  Patients: %d\n', length(ages));
            fprintf('  Age range: %d - %d years\n', min(ages), max(ages));
            fprintf('  Mean age: %.1f ± %.1f years\n', mean(ages), std(ages));
            fprintf('  Median age: %.1f years\n', median(ages));
            
            % Age group analysis
            age_groups = {[0, 30], [31, 50], [51, 70], [71, 100]};
            age_group_names = {'≤30', '31-50', '51-70', '>70'};
            
            fprintf('  Age distribution:\n');
            for j = 1:length(age_groups)
                count = sum(ages >= age_groups{j}(1) & ages <= age_groups{j}(2));
                percentage = (count / length(ages)) * 100;
                fprintf('    %s years: %d (%.1f%%)\n', age_group_names{j}, count, percentage);
            end
            fprintf('\n');
            
            % Plot histogram
            subplot(1, 3, i);
            histogram(ages, 20, 'FaceColor', colors{i}, 'EdgeColor', 'black', 'FaceAlpha', 0.7);
            title(sprintf('%s Group (n=%d)', group_name, length(ages)));
            xlabel('Age (years)');
            ylabel('Frequency');
            grid on;
        end
    end
    
    sgtitle('Age Distributions Across ECG Groups', 'FontSize', 14, 'FontWeight', 'bold');
end

% Helper function to verify the organization
function verify_organization(output_path)
    % Verify that the organization was successful
    
    fprintf('\n=== VERIFICATION ===\n');
    
    datasets = {'training', 'validation'};
    groups = {'SB', 'AFIB', 'SR'};
    
    for i = 1:length(datasets)
        fprintf('%s dataset:\n', datasets{i});
        
        for j = 1:length(groups)
            group_dir = fullfile(output_path, datasets{i}, groups{j});
            
            if exist(group_dir, 'dir')
                mat_files = dir(fullfile(group_dir, '*.mat'));
                hea_files = dir(fullfile(group_dir, '*.hea'));
                
                fprintf('  %s: %d .mat files, %d .hea files\n', groups{j}, length(mat_files), length(hea_files));
                
                % Check if file counts match
                if length(mat_files) ~= length(hea_files)
                    fprintf('    WARNING: Mismatch in file counts!\n');
                end
            else
                fprintf('  %s: Directory not found\n', groups{j});
            end
        end
    end
end

% Run the organization
organize_ecg_database();