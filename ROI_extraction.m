


% ad_roi_extraction.m - Extract AD-relevant ROIs using AAL Atlas



% Set up project paths
% Change this to your project directory
project_dir = '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline';

% % Change this to your project directory
% project_dir = '/Users/liuzhonglin/Desktop/ADNI_Analysis'; % Or wherever you saved the data





% Create subdirectories
dirs = struct();
dirs.atlas = fullfile(project_dir, 'atlas/AAL3/');
dirs.roi = fullfile(project_dir, 'roi_masks');
dirs.resampled_roi = fullfile(project_dir, 'resampled_rois');
dirs.func = fullfile(project_dir, 'Data/AD');
dirs.output = fullfile(project_dir, 'pbn_data');

% Create directories if they don't exist
fields = fieldnames(dirs);
for i = 1:length(fields)
    if ~exist(dirs.(fields{i}), 'dir')
        mkdir(dirs.(fields{i}));
        fprintf('Created directory: %s\n', dirs.(fields{i}));
    end
end

% Initialize SPM
fprintf('Initializing SPM...\n');
try
    spm('defaults', 'fmri');
    spm_jobman('initcfg');
    fprintf('SPM initialized successfully.\n');
catch ME
    fprintf('Error initializing SPM: %s\n', ME.message);
    fprintf('Make sure SPM is in your MATLAB path.\n');
    return;
end

fprintf('================================\n');
fprintf('AD ROI Extraction using AAL Atlas\n');
fprintf('================================\n\n');


% Locate AAL3 atlas files
fprintf('\n--- STEP 1: LOCATING AAL3 FILES ---\n');

% Identify the AAL3 atlas NIfTI file (preferring 1mm version for precision)
aal3_nii_candidates = {
    fullfile(dirs.atlas, 'AAL3v1_1mm.nii'),
    fullfile(dirs.atlas, 'AAL3v1.nii')
};

aal3_file = '';
for i = 1:length(aal3_nii_candidates)
    if exist(aal3_nii_candidates{i}, 'file')
        aal3_file = aal3_nii_candidates{i};
        fprintf('Found AAL3 atlas: %s\n', aal3_file);
        break;
    end
end

if isempty(aal3_file)
    error('AAL3 atlas file not found. Please check the atlas directory.');
end

% Find the AAL3 labels file
aal3_label_candidates = {
    fullfile(dirs.atlas, 'AAL3v1_1mm.nii.txt'),
    fullfile(dirs.atlas, 'AAL3v1.nii.txt'),
    fullfile(dirs.atlas, 'AAL3v1.txt')
};

aal3_txt = '';
for i = 1:length(aal3_label_candidates)
    if exist(aal3_label_candidates{i}, 'file')
        aal3_txt = aal3_label_candidates{i};
        fprintf('Found AAL3 labels: %s\n', aal3_txt);
        break;
    end
end

% If no text file found, check for the MATLAB label files
if isempty(aal3_txt)
    fprintf('No AAL3 text label file found. Checking for MATLAB label files...\n');
    
    aal3_m = fullfile(dirs.atlas, 'AAL3.m');
    if exist(aal3_m, 'file')
        fprintf('Found AAL3.m file. Will use this for labels.\n');
    else
        fprintf('Checking for other label files...\n');
    end
end

% Load the AAL3 atlas NIfTI file
fprintf('Loading AAL3 atlas image...\n');
V_aal = spm_vol(aal3_file);
aal_img = spm_read_vols(V_aal);
fprintf('Atlas loaded. Dimensions: %d x %d x %d\n', size(aal_img));
fprintf('Value range: [%d to %d]\n', min(aal_img(:)), max(aal_img(:)));


% ======================= MODIFICATION START =======================
% This section has been rewritten to be more robust. It now reads the
% label file line-by-line to handle inconsistent formatting.
fprintf('\n--- STEP 2: LOADING AAL3 LABELS ---\n');

% Initialize variables to store region information
region_indices = [];
region_names = {};

% Method 1: Try to load from text file (line-by-line)
if ~isempty(aal3_txt)
    fprintf('Loading labels from text file: %s\n', aal3_txt);
    fid = fopen(aal3_txt, 'r');
    if fid == -1
        error('Could not open AAL3 labels file: %s', aal3_txt);
    end
    
    line = fgetl(fid);
    while ischar(line)
        % Split the line by spaces or tabs
        parts = strsplit(line);
        
        % Remove any empty parts that result from multiple spaces
        parts = parts(~cellfun('isempty', parts));
        
        if length(parts) >= 2
            % The first part is the index, the second is the name
            index = str2double(parts{1});
            name = parts{2};
            
            if ~isnan(index)
                region_indices(end+1) = index;
                region_names{end+1} = name;
            end
        end
        line = fgetl(fid);
    end
    fclose(fid);
    
    if ~isempty(region_indices)
        fprintf('Successfully read %d regions (line-by-line method)\n', length(region_indices));
    else
        fprintf('Failed to parse label file using line-by-line method.\n');
    end
end
% ======================= MODIFICATION END =========================

% Method 2: If text file loading failed, try AAL3.m
if isempty(region_indices) && exist(fullfile(dirs.atlas, 'AAL3.m'), 'file')
    fprintf('Trying to load labels from AAL3.m...\n');
    aal3_m_path = fullfile(dirs.atlas, 'AAL3.m');
    
    % Add the atlas directory to path temporarily
    old_path = path;
    addpath(dirs.atlas);
    
    % Run the AAL3.m file which should define ROI variables
    try
        run(aal3_m_path);
        if exist('ROI_name', 'var') && exist('ROI_MNI_V5_List', 'var')
            fprintf('Successfully loaded labels from AAL3.m\n');
            num_regions = length(ROI_name);
            region_names = ROI_name;
            region_indices = 1:num_regions;
            fprintf('Found %d regions\n', num_regions);
        else
            fprintf('AAL3.m did not define the expected variables\n');
        end
    catch ME
        fprintf('Error running AAL3.m: %s\n', ME.message);
    end
    
    % Restore original path
    path(old_path);
end

% Method 3: If previous methods failed, try to extract from NIfTI directly
if isempty(region_indices)
    fprintf('Attempting to determine regions directly from the atlas...\n');
    
    % Find unique values in the atlas (excluding 0 which is background)
    unique_indices = unique(aal_img(:));
    unique_indices = unique_indices(unique_indices > 0);
    
    region_indices = unique_indices;
    region_names = cell(length(unique_indices), 1);
    
    % Generate generic names
    for i = 1:length(unique_indices)
        region_names{i} = sprintf('Region_%03d', unique_indices(i));
    end
    
    fprintf('Created generic labels for %d regions\n', length(unique_indices));
    fprintf('WARNING: Using generic region names. Analysis will be limited.\n');
end

% Display the first few regions to verify
fprintf('\nSample of AAL3 regions:\n');
fprintf('%-5s %-30s\n', 'Index', 'Region Name');
fprintf('%-5s %-30s\n', '-----', '------------------------------');
for i = 1:min(10, length(region_indices))
    fprintf('%-5d %-30s\n', region_indices(i), region_names{i});
end
fprintf('...\n');
fprintf('Total regions: %d\n', length(region_indices));


% Define AD-relevant ROIs for Alzheimer's disease analysis
fprintf('\n--- STEP 3: DEFINING AD-RELEVANT REGIONS ---\n');

% Create lookup map for quick region name to index mapping
name_to_index = containers.Map();
for i = 1:length(region_names)
    % Use the region name as the key and the index as the value
    name_to_index(region_names{i}) = region_indices(i);
end

% Define search patterns for the 18 most important AD connectivity ROIs
dm_network = {'Precuneus', 'Angular', 'Frontal_Med_Orb'};  % 6 ROIs (bilateral)
ecn_network = {'Frontal_Sup_2', 'Parietal_Sup'};          % 4 ROIs (bilateral)  
salience_network = {'Insula', 'Supp_Motor_Area'}; % 4 ROIs (bilateral)       
memory_network = {'Hippocampus', 'ParaHippocampal'};      % 4 ROIs (bilateral)

% Create structure to hold the focused ROI selection
ad_networks = struct();
ad_networks.DMN = {};   % Default Mode Network (6 ROIs)
ad_networks.ECN = {};   % Executive Control Network (4 ROIs)
ad_networks.SN = {};    % Salience Network (4 ROIs)
ad_networks.MTL = {};   % Memory/Temporal Lobe Network (4 ROIs)

% Function to find regions matching a set of patterns
find_matching_regions = @(patterns) findMatchingRegions(region_names, patterns);

% Find all regions for each network
for i = 1:length(dm_network)
    pattern = dm_network{i};
    matches = find_matching_regions({pattern});
    ad_networks.DMN = [ad_networks.DMN, matches];
end

for i = 1:length(ecn_network)
    pattern = ecn_network{i};
    matches = find_matching_regions({pattern});
    ad_networks.ECN = [ad_networks.ECN, matches];
end

for i = 1:length(salience_network)
    pattern = salience_network{i};
    matches = find_matching_regions({pattern});
    ad_networks.SN = [ad_networks.SN, matches];
end

for i = 1:length(memory_network)
    pattern = memory_network{i};
    matches = find_matching_regions({pattern});
    ad_networks.MTL = [ad_networks.MTL, matches];
end

% Display the selected networks and regions
fprintf('\nSelected Networks for AD Analysis:\n');
fprintf('-------------------------------------\n');

network_names = fieldnames(ad_networks);
total_rois = 0;

% Create a table to store ROI information
roi_info = table('Size', [0, 5], ...
    'VariableTypes', {'string', 'string', 'double', 'string', 'double'}, ...
    'VariableNames', {'Network', 'Region', 'AAL_Index', 'Hemisphere', 'VoxelCount'});

for n = 1:length(network_names)
    network = network_names{n};
    regions = ad_networks.(network);
    
    fprintf('\n%s Network Regions:\n', network);
    
    for r = 1:length(regions)
        region_name = regions{r};
        
        % Check if region exists in AAL3 atlas
        if isKey(name_to_index, region_name)
            region_idx = name_to_index(region_name);
            
            % Determine hemisphere
            if contains(region_name, '_L')
                hemisphere = 'Left';
            elseif contains(region_name, '_R')
                hemisphere = 'Right';
            else
                hemisphere = 'Bilateral';
            end
            
            % Count voxels in this region
            voxel_count = sum(aal_img(:) == region_idx);
            
            % Add to the table
            roi_info = [roi_info; {network, region_name, region_idx, hemisphere, voxel_count}];
            
            % Display region info
            fprintf('  %2d. %-30s (AAL Index: %3d, %6s, %8d voxels)\n', ...
                total_rois + 1, region_name, region_idx, hemisphere, voxel_count);
            
            total_rois = total_rois + 1;
        else
            fprintf('  ⚠️ Region "%s" not found in AAL3 atlas\n', region_name);
        end
    end
end

fprintf('\nTotal AD-relevant ROIs: %d\n', total_rois);

% Helper function to find regions matching patterns
function matches = findMatchingRegions(region_names, patterns)
    matches = {};
    for i = 1:length(patterns)
        pattern = patterns{i};
        
        % Find all regions containing this pattern
        for j = 1:length(region_names)
            if contains(region_names{j}, pattern)
                matches{end+1} = region_names{j};
            end
        end
    end
    
    % Remove any duplicates
    matches = unique(matches);
end


% Create individual ROI masks for each AD-relevant region
fprintf('\n--- STEP 4: CREATING INDIVIDUAL ROI MASKS ---\n');

% Create directory for ROI masks if it doesn't exist
if ~exist(dirs.roi, 'dir')
    mkdir(dirs.roi);
    fprintf('Created ROI directory: %s\n', dirs.roi);
end

% Load the AAL3 atlas image if not already loaded
if ~exist('aal_img', 'var') || isempty(aal_img)
    fprintf('Reloading AAL3 atlas image...\n');
    V_aal = spm_vol(aal3_file);
    aal_img = spm_read_vols(V_aal);
    fprintf('Atlas loaded. Dimensions: %d x %d x %d\n', size(aal_img));
end

% Get dimensions of AAL atlas
aal_dims = size(aal_img);

% Count how many ROIs to process
num_rois = height(roi_info);
fprintf('Creating individual masks for %d AD-relevant ROIs...\n', num_rois);

% Create a diagnostic image directory
diagnostic_dir = fullfile(dirs.roi, 'diagnostic_images');
if ~exist(diagnostic_dir, 'dir')
    mkdir(diagnostic_dir);
    fprintf('Created diagnostic images directory: %s\n', diagnostic_dir);
end

% Process each ROI in the table
for r = 1:num_rois
    % Get ROI information
    roi_name = roi_info.Region{r};
    network = roi_info.Network{r};
    aal_index = roi_info.AAL_Index(r);
    voxel_count = roi_info.VoxelCount(r);
    
    fprintf('Processing ROI %d of %d: %s (AAL index: %d, %d voxels)\n', ...
        r, num_rois, roi_name, aal_index, voxel_count);
    
    % Create binary mask for this ROI
    roi_mask = (aal_img == aal_index);
    
    % Verify the mask contains expected voxels
    mask_voxel_count = sum(roi_mask(:));
    if mask_voxel_count ~= voxel_count
        fprintf('  ⚠️ WARNING: Mask contains %d voxels, expected %d\n', ...
            mask_voxel_count, voxel_count);
    end
    
    % Skip if empty
    if mask_voxel_count == 0
        fprintf('  ⚠️ ERROR: No voxels found for this ROI! Skipping.\n');
        continue;
    end
    
    % Create output filename (including network prefix for organization)
    % Replace spaces with underscores for safer filenames
    safe_roi_name = strrep(roi_name, ' ', '_');
    output_file = fullfile(dirs.roi, sprintf('%s_%s_%d.nii', network, safe_roi_name, aal_index));
    
    % Create NIfTI header for the mask
    V_mask = V_aal;  % Copy header from atlas
    V_mask.fname = output_file;
    V_mask.dt = [2 0];  % uint8 datatype
    
    % Write mask to file
    spm_write_vol(V_mask, double(roi_mask));
    
    fprintf('  Saved ROI mask to: %s\n', output_file);
    
    % Create a diagnostic image (central slice through the ROI)
    try
        % Find the center of mass of the ROI
        [x, y, z] = ind2sub(aal_dims, find(roi_mask));
        if isempty(x)
            fprintf('  ⚠️ Cannot create diagnostic image: empty mask\n');
            continue;
        end
        
        center_x = round(mean(x));
        center_y = round(mean(y));
        center_z = round(mean(z));
        
        % Create diagnostic figure
        h = figure('Visible', 'off', 'Position', [100, 100, 900, 300]);
        
        % Sagittal view (YZ)
        subplot(1, 3, 1);
        sagittal_slice = squeeze(aal_img(center_x, :, :))';
        imagesc(sagittal_slice);
        colormap('gray');
        hold on;
        sagittal_mask = squeeze(roi_mask(center_x, :, :))';
        if any(sagittal_mask(:))
            h_mask = imagesc(sagittal_mask);
            set(h_mask, 'AlphaData', double(sagittal_mask) * 0.5);
        end
        title(sprintf('Sagittal (x=%d)', center_x));
        axis image;
        
        % Coronal view (XZ)
        subplot(1, 3, 2);
        coronal_slice = squeeze(aal_img(:, center_y, :))';
        imagesc(coronal_slice);
        colormap('gray');
        hold on;
        coronal_mask = squeeze(roi_mask(:, center_y, :))';
        if any(coronal_mask(:))
            h_mask = imagesc(coronal_mask);
            set(h_mask, 'AlphaData', double(coronal_mask) * 0.5);
        end
        title(sprintf('Coronal (y=%d)', center_y));
        axis image;
        
        % Axial view (XY)
        subplot(1, 3, 3);
        axial_slice = squeeze(aal_img(:, :, center_z));
        imagesc(axial_slice);
        colormap('gray');
        hold on;
        axial_mask = squeeze(roi_mask(:, :, center_z));
        if any(axial_mask(:))
            h_mask = imagesc(axial_mask);
            set(h_mask, 'AlphaData', double(axial_mask) * 0.5);
        end
        title(sprintf('Axial (z=%d)', center_z));
        axis image;
        
        % Add global title
        sgtitle(sprintf('ROI: %s (AAL index: %d)', roi_name, aal_index));
        
        % Save the diagnostic image
        % Replace spaces with underscores for safer filenames
        safe_roi_name = strrep(roi_name, ' ', '_');
        diag_file = fullfile(diagnostic_dir, sprintf('%s_%s_%d_diagnostic.png', network, safe_roi_name, aal_index));
        saveas(h, diag_file);
        close(h);
        
        fprintf('  Saved diagnostic image to: %s\n', diag_file);
    catch ME
        fprintf('  ⚠️ Could not create diagnostic image: %s\n', ME.message);
    end
end

% Create a combined atlas of all AD regions
fprintf('\nCreating combined AD atlas...\n');

% Initialize a combined mask for all AD ROIs (using same datatype as atlas)
combined_mask = zeros(aal_dims, class(aal_img));

% Add each ROI to the combined mask with its AAL index
for r = 1:num_rois
    aal_index = roi_info.AAL_Index(r);
    roi_mask = (aal_img == aal_index);
    
    % Convert to same class as combined_mask before multiplication
    roi_mask_converted = cast(roi_mask, class(combined_mask));
    aal_index_converted = cast(aal_index, class(combined_mask));
    
    % Add to combined mask
    combined_mask = combined_mask + (roi_mask_converted * aal_index_converted);
end

% Save the combined mask
combined_file = fullfile(dirs.roi, 'AD_combined_atlas.nii');
V_combined = V_aal;
V_combined.fname = combined_file;
V_combined.dt = V_aal.dt;  % Use same datatype as original atlas
spm_write_vol(V_combined, combined_mask);
fprintf('Saved combined AD atlas to: %s\n', combined_file);

% =========================================================================
% STEP 5 & 6 (COMBINED): PROCESS EACH FUNCTIONAL FILE INDIVIDUALLY
% This new, robust loop will:
%   1. Take one functional file.
%   2. Resample all ROI masks to match THAT specific file.
%   3. Extract the time series using the newly matched masks.
%   4. Repeat for the next functional file.
% =========================================================================
fprintf('\n--- STARTING PER-SUBJECT ROI RESAMPLING AND TIME SERIES EXTRACTION ---\n');

% Find all preprocessed functional files to be processed
func_files = dir(fullfile(dirs.func, 'sub-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'));
if isempty(func_files)
    error('No preprocessed functional files found in %s. Please check the directory.', dirs.func);
end
fprintf('Found %d functional files to process.\n', length(func_files));

% Get the original, high-resolution ROI mask files
original_roi_files = dir(fullfile(dirs.roi, '*.nii'));
original_roi_files = original_roi_files(~contains({original_roi_files.name}, 'combined'));

% Main loop to process each functional file
for f = 1:length(func_files)
    
    % --- PART A: PREPARE THE REFERENCE FUNCTIONAL IMAGE ---
    
    func_gz_file = fullfile(dirs.func, func_files(f).name);
    [~, func_name, ~] = fileparts(func_files(f).name); % Gets the base name like 'sub-...'
    
    fprintf('\n================================================================\n');
    fprintf('Processing functional file %d of %d: %s\n', f, length(func_files), func_name);
    
    % Unzip the functional file for SPM, creating a temporary .nii file
    func_nii_file = strrep(func_gz_file, '.nii.gz', '.nii');
    if ~exist(func_nii_file, 'file')
        fprintf('  Unzipping functional file...\n');
        gunzip(func_gz_file);
    end
    
    % --- PART B: RESAMPLE ALL ROIS TO THIS SUBJECT'S SPACE ---
    
    fprintf('  Resampling %d ROIs to match %s...\n', length(original_roi_files), func_name);
    
    % Create a subject-specific directory for this subject's resampled ROIs
    subject_resampled_dir = fullfile(dirs.resampled_roi, func_name);
    if ~exist(subject_resampled_dir, 'dir')
        mkdir(subject_resampled_dir);
    end
    
    for r = 1:length(original_roi_files)
        original_roi_file = fullfile(dirs.roi, original_roi_files(r).name);
        [~, roi_name_only, ~] = fileparts(original_roi_files(r).name);
        
        resampled_file = fullfile(subject_resampled_dir, [roi_name_only, '_func.nii']);
        
        % Only resample if it hasn't been done already for this subject
        if ~exist(resampled_file, 'file')
            clear matlabbatch;
            matlabbatch{1}.spm.spatial.coreg.write.ref = {[func_nii_file, ',1']};
            matlabbatch{1}.spm.spatial.coreg.write.source = {original_roi_file};
            matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 0; % Nearest neighbor
            matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap = [0 0 0];
            matlabbatch{1}.spm.spatial.coreg.write.roptions.mask = 0;
            matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = 'r';
            
            spm_jobman('run', matlabbatch);
            
            resliced_source_file = fullfile(dirs.roi, ['r', original_roi_files(r).name]);
            movefile(resliced_source_file, resampled_file);
        end
    end
    fprintf('  Resampling for this subject complete.\n');
    
    % --- PART C: EXTRACT TIME SERIES USING THE NEWLY RESAMPLED MASKS ---
    
    fprintf('  Extracting time series...\n');
    V_func = spm_vol(func_nii_file);
    num_volumes = length(V_func);
    
    % Get this subject's resampled ROI masks
    subject_roi_files = dir(fullfile(subject_resampled_dir, '*_func.nii'));
    
    time_series = zeros(num_volumes, length(subject_roi_files));
    roi_names = cell(length(subject_roi_files), 1);
    
    for r = 1:length(subject_roi_files)
        roi_file = fullfile(subject_resampled_dir, subject_roi_files(r).name);
        [~, roi_name_only, ~] = fileparts(subject_roi_files(r).name);
        roi_names{r} = roi_name_only;
        
        V_roi = spm_vol(roi_file);
        roi_img = spm_read_vols(V_roi);
        roi_mask = roi_img > 0.5;
        
        if sum(roi_mask(:)) == 0
            time_series(:, r) = NaN;
            continue;
        end
        
        for vol = 1:num_volumes
            vol_img = spm_read_vols(V_func(vol));
            roi_signal = vol_img(roi_mask); % This line should now work
            time_series(vol, r) = mean(roi_signal);
        end
    end
    
    % --- PART D: SAVE RESULTS AND CLEAN UP ---
    
    % Save time series to CSV file
    output_file = fullfile(dirs.output, [func_name, '_time_series.csv']);
    % Create a table with ROI names as headers
    ts_table = array2table(time_series, 'VariableNames', roi_names);
    % Add a 'Volume' column
    ts_table = addvars(ts_table, (1:num_volumes)', 'Before', 1, 'NewVariableNames', 'Volume');
    writetable(ts_table, output_file);
    fprintf('  Saved time series to: %s\n', output_file);
    
    % Clean up the temporary unzipped functional file for this subject
    delete(func_nii_file);
    fprintf('  Cleaned up temporary .nii file.\n');
    
end

fprintf('\nTime series extraction complete for all subjects.\n');
fprintf('\n=========================================================================\n');
fprintf('ROI EXTRACTION PIPELINE COMPLETED\n');
fprintf('=========================================================================\n\n');
