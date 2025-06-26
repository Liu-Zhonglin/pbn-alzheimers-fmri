% Denoising_and_Binarization.m
% FINAL VERSION: This script recursively finds all CSV files, and performs
% classification by saving the output files into organized subfolders
% that match the input structure (AD, MCI, Normal).

% Define paths
project_dir = '/Users/liuzhonglin/Desktop/URFP/Codes/';
data_dir = fullfile(project_dir, 'Pipeline/pbn_data');  % Input directory with AD/MCI/Normal subfolders
pbn_ready_dir = fullfile(project_dir, 'Pipeline/PBN_Ready_Data');  % Main output directory
output_dir = fullfile(project_dir, 'Pipeline/fMRI Preprocessing for PBN Analysis');

% Create base output directories if they don't exist
if ~exist(pbn_ready_dir, 'dir'), mkdir(pbn_ready_dir); end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

denoised_dir = fullfile(output_dir, 'denoised');
if ~exist(denoised_dir, 'dir'), mkdir(denoised_dir); end

binary_dir = fullfile(output_dir, 'binary');
if ~exist(binary_dir, 'dir'), mkdir(binary_dir); end

% Use the '**' wildcard to search recursively in all subdirectories
csv_files = dir(fullfile(data_dir, '**', '*.csv'));
csv_files = csv_files(~[csv_files.isdir]); % Remove any directory entries

if isempty(csv_files)
    fprintf('No CSV files found in directory: %s or its subdirectories.\n', data_dir);
    return;
end

fprintf('Found %d CSV files to process across all subdirectories.\n', length(csv_files));

% Process each CSV file
for i = 1:length(csv_files)
    current_file = csv_files(i);
    current_file_path = fullfile(current_file.folder, current_file.name);
    
    % ======================================================================
    % --- CLASSIFICATION LOGIC STARTS HERE ---
    % ======================================================================
    
    % STEP 1: Get the group name (e.g., 'AD', 'MCI') from the input folder path.
    [~, group_name] = fileparts(current_file.folder);
    
    % STEP 2: Define the full path for the group-specific output directories.
    denoised_group_dir = fullfile(denoised_dir, group_name);
    binary_group_dir = fullfile(binary_dir, group_name);
    pbn_ready_group_dir = fullfile(pbn_ready_dir, group_name);
    
    % STEP 3: Create these specific output directories if they don't already exist.
    if ~exist(denoised_group_dir, 'dir'), mkdir(denoised_group_dir); end
    if ~exist(binary_group_dir, 'dir'), mkdir(binary_group_dir); end
    if ~exist(pbn_ready_group_dir, 'dir'), mkdir(pbn_ready_group_dir); end
    
    % STEP 4: Generate the final output filenames, ensuring they are placed
    %         within the correct classified subfolder.
    [~, file_base, ~] = fileparts(current_file.name);
    denoised_file = fullfile(denoised_group_dir, [file_base, '_denoised.csv']);
    binary_file = fullfile(binary_group_dir, [file_base, '_binary.csv']);
    pbn_ready_file = fullfile(pbn_ready_group_dir, [file_base, '_PBN_ready.csv']);
    
    % --- END OF CLASSIFICATION LOGIC ---
    
    
    fprintf('\n========== PROCESSING FILE %d/%d ==========\n', i, length(csv_files));
    fprintf('Input file: %s\n', fullfile(group_name, current_file.name)); % Display with group
    fprintf('PBN-ready output file: %s\n', pbn_ready_file);
    
    % Process current file using your existing function
    try
        process_single_file(current_file_path, denoised_file, binary_file, pbn_ready_file, ...
            'iterative_hmm', []);
        fprintf('Successfully processed file %d/%d\n', i, length(csv_files));
    catch ME
        fprintf('Error processing file %s: %s\n', current_file.name, ME.message);
        fprintf('Continuing to next file...\n');
    end
end

fprintf('\n========== ALL PROCESSING COMPLETED ==========\n');
fprintf('Processed %d CSV files.\n', length(csv_files));
fprintf('PBN-ready binary files are saved to their respective group subfolders inside:\n');
fprintf('  %s\n', pbn_ready_dir);