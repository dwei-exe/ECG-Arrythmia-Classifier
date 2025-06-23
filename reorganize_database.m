function reorganize_ecg_database()
    % ECG Database Reorganization Script
    % Reorganizes ECG data into clinical groups and creates balanced datasets
    
    % Define paths
    base_path = 'C:\Users\henry\Downloads\ECG-Dx\Raw Download\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords';
    output_path = 'C:\Users\henry\Downloads\ECG-Dx\Organized_Dataset';
    
    % Create output directories
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end
    
    % Define SNOMED CT code mappings based on clinical groupings
    group_codes = containers.Map();
    
    % SB: Sinus Bradycardia
    group_codes('SB') = [426177001];
    
    % AFIB: Atrial Fibrillation + Atrial Flutter  
    group_codes('AFIB') = [164889003, 164890007];
    
    % GSVT: Supraventricular Tachycardia, Atrial Tachycardia, AVRT
    group_codes('GSVT') = [426761007, 713422000, 233897008];
    
    % SR: Sinus Rhythm + Sinus Irregularity
    group_codes('SR') = [426783006, 427393009];
    
    % Initialize data structures
    groups = {'SB', 'AFIB', 'GSVT', 'SR'};
    patient_data = containers.Map();
    for i = 1:length(groups)
        patient_data(groups{i}) = {};
    end
    
    % Process all ECG files
    fprintf('Starting ECG database reorganization...\n');
    fprintf('Scanning directory: %s\n', base_path);
    
    total_processed = 0;
    total_categorized = 0;
    file_errors = 0;
    
    % Get all subdirectories (46 folders)
    main_dirs = dir(base_path);
    main_dirs = main_dirs([main_dirs.isdir] & ~ismember({main_dirs.name}, {'.', '..'}));
    
    fprintf('Found %d main directories to process\n', length(main_dirs));
    
    for i = 1:length(main_dirs)
        main_dir_path = fullfile(base_path, main_dirs(i).name);
        fprintf('Processing main directory %d/%d: %s\n', i, length(main_dirs), main_dirs(i).name);
        
        % Get subdirectories (10 subfolders each)
        sub_dirs = dir(main_dir_path);
        sub_dirs = sub_dirs([sub_dirs.isdir] & ~ismember({sub_dirs.name}, {'.', '..'}));
        
        for j = 1:length(sub_dirs)
            sub_dir_path = fullfile(main_dir_path, sub_dirs(j).name);
            fprintf('  Processing subdirectory %d/%d: %s\n', j, length(sub_dirs), sub_dirs(j).name);
            
            % Quick check of directory contents
            all_files = dir(sub_dir_path);
            hea_count = sum(endsWith({all_files.name}, '.hea'));
            mat_count = sum(endsWith({all_files.name}, '.mat'));
            fprintf('    Found %d .hea files and %d .mat files\n', hea_count, mat_count);
            
            % Get all .hea files (header files)
            hea_files = dir(fullfile(sub_dir_path, '*.hea'));
            % Get all .mat files for matching
            mat_files = dir(fullfile(sub_dir_path, '*.mat'));
            
            for k = 1:length(hea_files)
                total_processed = total_processed + 1;
                
                hea_file_path = fullfile(sub_dir_path, hea_files(k).name);
                
                % Find corresponding .mat file (handle duplicates like "filename (1).mat")
                [~, base_name, ~] = fileparts(hea_files(k).name);
                mat_file_path = find_matching_mat_file(sub_dir_path, base_name, mat_files);
                
                % Skip if no matching .mat file found
                if isempty(mat_file_path)
                    fprintf('Warning: No matching .mat file found for %s\n', hea_files(k).name);
                    file_errors = file_errors + 1;
                    continue;
                end
                
                % Extract patient information from header file
                patient_info = extract_patient_info(hea_file_path);
                
                if ~isempty(patient_info)
                    % Determine group based on SNOMED CT codes
                    assigned_group = assign_to_group(patient_info.dx_codes, group_codes);
                    
                    if ~isempty(assigned_group)
                        total_categorized = total_categorized + 1;
                        
                        % Create patient record
                        record = struct();
                        record.patient_id = patient_info.patient_id;
                        record.age = patient_info.age;
                        record.sex = patient_info.sex;
                        record.dx_codes = patient_info.dx_codes;
                        record.hea_file = hea_file_path;
                        record.mat_file = mat_file_path;
                        record.original_path = sub_dir_path;
                        
                        % Add to appropriate group
                        current_data = patient_data(assigned_group);
                        current_data{end+1} = record;
                        patient_data(assigned_group) = current_data;
                    end
                end
                
                % Progress update every 1000 files
                if mod(total_processed, 1000) == 0
                    fprintf('    Processed %d files...\n', total_processed);
                end
            end
        end
    end
    
    % Display distribution
    fprintf('\n=== DISTRIBUTION SUMMARY ===\n');
    fprintf('Total files processed: %d\n', total_processed);
    fprintf('Total files categorized: %d\n', total_categorized);
    fprintf('File errors encountered: %d\n', file_errors);
    fprintf('Distribution by group:\n');
    
    group_sizes = zeros(1, length(groups));
    for i = 1:length(groups)
        group_data = patient_data(groups{i});
        group_sizes(i) = length(group_data);
        fprintf('  %s: %d patients\n', groups{i}, group_sizes(i));
    end
    
    % Find minimum group size for balancing
    min_size = min(group_sizes);
    fprintf('\nMinimum group size (for balancing): %d\n', min_size);
    
    % Create balanced dataset
    fprintf('\n=== CREATING BALANCED DATASET ===\n');
    train_size = floor(min_size * 0.8);
    val_size = min_size - train_size;
    
    fprintf('Training samples per group: %d\n', train_size);
    fprintf('Validation samples per group: %d\n', val_size);
    
    % Create balanced dataset folders and copy files
    for i = 1:length(groups)
        group_name = groups{i};
        group_data = patient_data(group_name);
        
        % Randomly sample to balance the dataset
        if length(group_data) > min_size
            rand_indices = randperm(length(group_data), min_size);
            group_data = group_data(rand_indices);
        end
        
        % Split into training and validation
        train_indices = 1:train_size;
        val_indices = (train_size+1):min_size;
        
        % Create directories
        train_dir = fullfile(output_path, 'training', group_name);
        val_dir = fullfile(output_path, 'validation', group_name);
        
        if ~exist(train_dir, 'dir')
            mkdir(train_dir);
        end
        if ~exist(val_dir, 'dir')
            mkdir(val_dir);
        end
        
        % Copy training files
        fprintf('Copying %s training files...\n', group_name);
        copy_files_with_age_label(group_data(train_indices), train_dir);
        
        % Copy validation files
        fprintf('Copying %s validation files...\n', group_name);
        copy_files_with_age_label(group_data(val_indices), val_dir);
        
        % Save metadata
        save_metadata(group_data(train_indices), fullfile(train_dir, 'metadata.mat'));
        save_metadata(group_data(val_indices), fullfile(val_dir, 'metadata.mat'));
    end
    
    % Generate summary report
    generate_summary_report(output_path, groups, train_size, val_size, patient_data);
    
    fprintf('\n=== REORGANIZATION COMPLETE ===\n');
    fprintf('Output directory: %s\n', output_path);
end

function mat_file_path = find_matching_mat_file(directory, base_name, mat_files)
    % Find matching .mat file for a given base name, handling duplicates
    mat_file_path = '';
    
    % First try exact match
    exact_match = fullfile(directory, [base_name, '.mat']);
    if exist(exact_match, 'file')
        mat_file_path = exact_match;
        return;
    end
    
    % If exact match fails, look through all .mat files for pattern matches
    for i = 1:length(mat_files)
        mat_name = mat_files(i).name;
        [~, mat_base, ~] = fileparts(mat_name);
        
        % Check if this mat file matches the base name (handling duplicates)
        % Remove any duplicate suffixes like " (1)", " (2)", etc.
        pattern = ' \(\d+\)$';
        clean_mat_base = regexprep(mat_base, pattern, '');
        
        if strcmp(clean_mat_base, base_name)
            mat_file_path = fullfile(directory, mat_name);
            return;
        end
    end
    
    % If still no match, try more flexible matching
    for i = 1:length(mat_files)
        mat_name = mat_files(i).name;
        if contains(mat_name, base_name)
            mat_file_path = fullfile(directory, mat_name);
            return;
        end
    end
end

function patient_info = extract_patient_info(hea_file_path)
    % Extract patient information from header file
    patient_info = struct();
    patient_info.patient_id = '';
    patient_info.age = NaN;
    patient_info.sex = '';
    patient_info.dx_codes = [];
    
    try
        fid = fopen(hea_file_path, 'r');
        if fid == -1
            return;
        end
        
        % Read file line by line
        while ~feof(fid)
            line = fgetl(fid);
            if ischar(line)
                % Extract patient ID from first line
                if isempty(patient_info.patient_id) && ~startsWith(line, '#')
                    parts = strsplit(line);
                    if ~isempty(parts)
                        patient_info.patient_id = parts{1};
                    end
                end
                
                % Extract age
                if startsWith(line, '#Age:')
                    age_str = strtrim(strrep(line, '#Age:', ''));
                    patient_info.age = str2double(age_str);
                end
                
                % Extract sex
                if startsWith(line, '#Sex:')
                    patient_info.sex = strtrim(strrep(line, '#Sex:', ''));
                end
                
                % Extract diagnosis codes
                if startsWith(line, '#Dx:')
                    dx_str = strtrim(strrep(line, '#Dx:', ''));
                    if ~strcmp(dx_str, 'Unknown') && ~isempty(dx_str)
                        % Split by comma and convert to numbers
                        dx_parts = strsplit(dx_str, ',');
                        dx_codes = [];
                        for k = 1:length(dx_parts)
                            code = str2double(strtrim(dx_parts{k}));
                            if ~isnan(code)
                                dx_codes(end+1) = code;
                            end
                        end
                        patient_info.dx_codes = dx_codes;
                    end
                end
            end
        end
        
        fclose(fid);
        
    catch ME
        fprintf('Error reading file %s: %s\n', hea_file_path, ME.message);
        if exist('fid', 'var') && fid ~= -1
            fclose(fid);
        end
    end
end

function group = assign_to_group(dx_codes, group_codes)
    % Assign patient to clinical group based on diagnosis codes
    group = '';
    
    if isempty(dx_codes)
        return;
    end
    
    groups = keys(group_codes);
    
    for i = 1:length(groups)
        group_name = groups{i};
        target_codes = group_codes(group_name);
        
        % Check if any diagnosis code matches this group
        if any(ismember(dx_codes, target_codes))
            group = group_name;
            return;
        end
    end
end

function copy_files_with_age_label(patient_records, dest_dir)
    % Copy files with age information in filename
    for i = 1:length(patient_records)
        record = patient_records{i};
        
        % Verify files exist before copying
        if ~exist(record.hea_file, 'file')
            fprintf('Warning: Header file not found: %s\n', record.hea_file);
            continue;
        end
        
        if ~exist(record.mat_file, 'file')
            fprintf('Warning: Data file not found: %s\n', record.mat_file);
            continue;
        end
        
        % Create age-labeled filename
        [~, base_name, ~] = fileparts(record.patient_id);
        age_suffix = sprintf('_age%d', record.age);
        
        try
            % Copy .hea file
            new_hea_name = sprintf('%s%s.hea', base_name, age_suffix);
            dest_hea_path = fullfile(dest_dir, new_hea_name);
            copyfile(record.hea_file, dest_hea_path);
            
            % Copy .mat file
            new_mat_name = sprintf('%s%s.mat', base_name, age_suffix);
            dest_mat_path = fullfile(dest_dir, new_mat_name);
            copyfile(record.mat_file, dest_mat_path);
            
        catch ME
            fprintf('Error copying files for patient %s: %s\n', record.patient_id, ME.message);
            fprintf('  Source HEA: %s\n', record.hea_file);
            fprintf('  Source MAT: %s\n', record.mat_file);
            continue;
        end
    end
end

function save_metadata(patient_records, metadata_file)
    % Save patient metadata for later analysis
    metadata = struct();
    
    for i = 1:length(patient_records)
        record = patient_records{i};
        metadata(i).patient_id = record.patient_id;
        metadata(i).age = record.age;
        metadata(i).sex = record.sex;
        metadata(i).dx_codes = record.dx_codes;
        metadata(i).original_path = record.original_path;
    end
    
    save(metadata_file, 'metadata');
end

function generate_summary_report(output_path, groups, train_size, val_size, patient_data)
    % Generate comprehensive summary report
    report_file = fullfile(output_path, 'reorganization_summary.txt');
    
    fid = fopen(report_file, 'w');
    
    fprintf(fid, '=== ECG DATABASE REORGANIZATION SUMMARY ===\n');
    fprintf(fid, 'Date: %s\n\n', datestr(now));
    
    fprintf(fid, 'CLINICAL GROUPS:\n');
    fprintf(fid, '- SB: Sinus Bradycardia\n');
    fprintf(fid, '- AFIB: Atrial Fibrillation + Atrial Flutter\n');
    fprintf(fid, '- GSVT: Supraventricular Tachycardia, Atrial Tachycardia, AVRT\n');
    fprintf(fid, '- SR: Sinus Rhythm + Sinus Irregularity\n\n');
    
    fprintf(fid, 'BALANCED DATASET CONFIGURATION:\n');
    fprintf(fid, 'Training samples per group: %d (80%%)\n', train_size);
    fprintf(fid, 'Validation samples per group: %d (20%%)\n\n', val_size);
    
    fprintf(fid, 'GROUP DISTRIBUTIONS:\n');
    total_available = 0;
    for i = 1:length(groups)
        group_data = patient_data(groups{i});
        available = length(group_data);
        total_available = total_available + available;
        fprintf(fid, '%s: %d available, %d used in balanced dataset\n', ...
                groups{i}, available, train_size + val_size);
    end
    
    fprintf(fid, '\nTotal available samples: %d\n', total_available);
    fprintf(fid, 'Total used in balanced dataset: %d\n', (train_size + val_size) * length(groups));
    
    % Age distribution analysis
    fprintf(fid, '\nAGE DISTRIBUTION BY GROUP:\n');
    for i = 1:length(groups)
        group_data = patient_data(groups{i});
        if ~isempty(group_data)
            ages = [];
            for j = 1:length(group_data)
                if ~isnan(group_data{j}.age)
                    ages(end+1) = group_data{j}.age;
                end
            end
            
            if ~isempty(ages)
                fprintf(fid, '%s: Mean=%.1f, Std=%.1f, Range=[%d-%d]\n', ...
                        groups{i}, mean(ages), std(ages), min(ages), max(ages));
            end
        end
    end
    
    fprintf(fid, '\nFILE ORGANIZATION:\n');
    fprintf(fid, 'Training data: training/[GROUP]/[PATIENT_ID]_age[AGE].[ext]\n');
    fprintf(fid, 'Validation data: validation/[GROUP]/[PATIENT_ID]_age[AGE].[ext]\n');
    fprintf(fid, 'Metadata files: [training|validation]/[GROUP]/metadata.mat\n');
    
    fclose(fid);
    
    fprintf('Summary report saved to: %s\n', report_file);
end

% Run the reorganization
reorganize_ecg_database();