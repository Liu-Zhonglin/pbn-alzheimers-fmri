% ad_roi_extraction.m - Extract AD-relevant ROIs using AAL Atlas
% FINAL COMPLETE SCRIPT (Version 8): Searches all subdirectories and classifies output.

% =========================================================================
% SETUP AND CONFIGURATION
% =========================================================================
project_dir = '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline';
TARGET_TR = 0.61;
fprintf('Target TR for harmonization set to: %.2fs\n', TARGET_TR);

dirs = struct();
dirs.atlas = fullfile(project_dir, 'atlas/AAL3/');
dirs.roi = fullfile(project_dir, 'roi_masks');
dirs.resampled_roi = fullfile(project_dir, 'resampled_rois');
dirs.func = fullfile(project_dir, 'Data'); 
dirs.output = fullfile(project_dir, 'pbn_data');
dirs.temp = fullfile(project_dir, 'temp_files');

fields = fieldnames(dirs);
for i = 1:length(fields)
    if ~exist(dirs.(fields{i}), 'dir'), mkdir(dirs.(fields{i})); end
end

fprintf('Initializing SPM...\n');
spm('defaults', 'fmri'); spm_jobman('initcfg');
fprintf('SPM initialized successfully.\n');

% =========================================================================
% STEP 1-4: SETUP ROIs (Unchanged)
% =========================================================================
fprintf('\n--- Setting up ROIs ---\n');
aal3_file = fullfile(dirs.atlas, 'AAL3v1.nii');
aal3_txt = fullfile(dirs.atlas, 'AAL3v1.nii.txt');
V_aal = spm_vol(aal3_file); aal_img = spm_read_vols(V_aal);
region_indices = []; region_names = {};
fid = fopen(aal3_txt, 'r'); line = fgetl(fid);
while ischar(line)
    parts = strsplit(line); parts = parts(~cellfun('isempty', parts));
    if length(parts) >= 2
        region_indices(end+1) = str2double(parts{1});
        region_names{end+1} = parts{2};
    end
    line = fgetl(fid);
end
fclose(fid);
name_to_index = containers.Map(region_names, region_indices);
dm_network = {'Precuneus', 'Angular', 'Frontal_Med_Orb'};
ecn_network = {'Frontal_Sup_2', 'Parietal_Sup'};
salience_network = {'Insula', 'Supp_Motor_Area'};
memory_network = {'Hippocampus', 'ParaHippocampal'};
ad_networks = struct();
ad_networks.DMN = findMatchingRegions(region_names, dm_network);
ad_networks.ECN = findMatchingRegions(region_names, ecn_network);
ad_networks.SN = findMatchingRegions(region_names, salience_network);
ad_networks.MTL = findMatchingRegions(region_names, memory_network);
roi_info = table();
network_names = fieldnames(ad_networks);
for n = 1:length(network_names)
    regions = ad_networks.(network_names{n});
    for r = 1:length(regions)
        roi_info = [roi_info; {network_names{n}, regions{r}, name_to_index(regions{r})}];
    end
end
roi_info.Properties.VariableNames = {'Network', 'Region', 'AAL_Index'};
for r = 1:height(roi_info)
    roi_mask = (aal_img == roi_info.AAL_Index(r));
    output_file = fullfile(dirs.roi, sprintf('%s_%d.nii', roi_info.Region{r}, roi_info.AAL_Index(r)));
    V_mask = V_aal; V_mask.fname = output_file; V_mask.dt = [2 0];
    spm_write_vol(V_mask, double(roi_mask));
end
fprintf('ROI setup complete. Found %d ROIs.\n', height(roi_info));

% =========================================================================
% STEP 5 & 6 (COMBINED): PROCESS, HARMONIZE, AND CLASSIFY
% =========================================================================
fprintf('\n--- STARTING PER-SUBJECT PROCESSING, HARMONIZATION & CLASSIFICATION ---\n');

func_files = dir(fullfile(dirs.func, '**', 'sub-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'));
fprintf('Found %d functional files to process across all subdirectories.\n', length(func_files));

original_roi_files = dir(fullfile(dirs.roi, '*.nii'));

for f = 1:length(func_files)
    func_gz_file = fullfile(func_files(f).folder, func_files(f).name);
    
    fprintf('\n================================================================\n');
    fprintf('Processing functional file %d of %d: %s\n', f, length(func_files), func_files(f).name);
    
    % --- NEW: Determine the subject's group from the file path ---
    [~, group_name] = fileparts(func_files(f).folder);
    fprintf('  - Detected group: %s\n', group_name);
    
    % --- NEW: Create the group-specific output directory ---
    output_group_dir = fullfile(dirs.output, group_name);
    if ~exist(output_group_dir, 'dir'), mkdir(output_group_dir); end
    
    base_gz_name = func_files(f).name;
    base_nii_name = strrep(base_gz_name, '.nii.gz', '.nii');
    func_nii_file = fullfile(dirs.temp, base_nii_name);
    
    if ~exist(func_nii_file, 'file'), gunzip(func_gz_file, dirs.temp); end
    
    V_func_orig = spm_vol(func_nii_file);
    original_tr = V_func_orig(1).private.hdr.pixdim(5);
    fprintf('  - Original TR: %.2fs\n', original_tr);
    
    file_to_process = func_nii_file;
    if abs(original_tr - TARGET_TR) > 1e-4
        fprintf('  - TR mismatch detected. Resampling...\n');
        % (Harmonization code is unchanged)
        func_4d_data = spm_read_vols(V_func_orig);
        [nx, ny, nz, nt_orig] = size(func_4d_data);
        original_time_points = (0:nt_orig-1) * original_tr;
        duration = original_time_points(end);
        nt_new = floor(duration / TARGET_TR) + 1;
        new_time_points = (0:nt_new-1) * TARGET_TR;
        reshaped_data = reshape(func_4d_data, [], nt_orig)';
        resampled_reshaped = interp1(original_time_points, reshaped_data, new_time_points, 'linear');
        resampled_4d_data = reshape(resampled_reshaped', [nx, ny, nz, nt_new]);
        harmonized_nii_file = fullfile(dirs.temp, ['harmonized_' base_nii_name]);
        nii_out = nifti;
        nii_out.dat = file_array(harmonized_nii_file, [nx, ny, nz, nt_new], 'FLOAT64-LE');
        nii_out.mat = V_func_orig(1).mat; nii_out.mat0 = V_func_orig(1).mat;
        nii_out.descrip = sprintf('Temporally resampled to TR=%.2f', TARGET_TR);
        nii_out.timing.tspace = TARGET_TR;
        create(nii_out);
        nii_out.dat(:,:,:,:) = resampled_4d_data;
        file_to_process = harmonized_nii_file;
        fprintf('  - Resampling complete.\n');
    else
        fprintf('  - TRs are close enough. No resampling needed.\n');
    end

    fprintf('  Resampling ROIs to match functional space...\n');
    subject_resampled_dir = fullfile(dirs.resampled_roi, base_nii_name);
    if ~exist(subject_resampled_dir, 'dir'), mkdir(subject_resampled_dir); end
    for r = 1:length(original_roi_files)
        original_roi_file = fullfile(dirs.roi, original_roi_files(r).name);
        [~, roi_name_only, ~] = fileparts(original_roi_files(r).name);
        resampled_file = fullfile(subject_resampled_dir, [roi_name_only, '_func.nii']);
        if ~exist(resampled_file, 'file')
            clear matlabbatch;
            matlabbatch{1}.spm.spatial.coreg.write.ref = {[file_to_process, ',1']};
            matlabbatch{1}.spm.spatial.coreg.write.source = {original_roi_file};
            matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 0;
            matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = 'r';
            spm_jobman('run', matlabbatch);
            movefile(fullfile(dirs.roi, ['r' original_roi_files(r).name]), resampled_file);
        end
    end
    
    fprintf('  Extracting time series...\n');
    V_func = spm_vol(file_to_process);
    num_volumes = length(V_func);
    subject_roi_files = dir(fullfile(subject_resampled_dir, '*_func.nii'));
    time_series = zeros(num_volumes, length(subject_roi_files));
    roi_names = cell(length(subject_roi_files), 1);
    for r = 1:length(subject_roi_files)
        roi_file = fullfile(subject_resampled_dir, subject_roi_files(r).name);
        [~, roi_names{r}, ~] = fileparts(subject_roi_files(r).name);
        V_roi = spm_vol(roi_file); roi_mask = spm_read_vols(V_roi) > 0.5;
        if sum(roi_mask(:)) == 0, time_series(:, r) = NaN; continue; end
        for vol = 1:num_volumes
            vol_img = spm_read_vols(V_func(vol));
            time_series(vol, r) = mean(vol_img(roi_mask));
        end
    end
    
    % --- NEW: Define final output path inside the classified folder ---
    output_file = fullfile(output_group_dir, [base_nii_name, '_time_series.csv']);
    ts_table = array2table(time_series, 'VariableNames', roi_names);
    ts_table = addvars(ts_table, (1:num_volumes)', 'Before', 1, 'NewVariableNames', 'Volume');
    writetable(ts_table, output_file);
    fprintf('  Saved classified time series to: %s\n', output_file);
    
    delete(func_nii_file);
    if exist('harmonized_nii_file', 'var') && exist(harmonized_nii_file, 'file')
        delete(harmonized_nii_file);
    end
    fprintf('  Cleaned up temporary files.\n');
end

fprintf('\n\n--- Pipeline Complete ---\n');

function matches = findMatchingRegions(region_names, patterns)
    matches = {};
    for i = 1:length(patterns)
        pattern = patterns{i};
        for j = 1:length(region_names)
            if contains(region_names{j}, pattern), matches{end+1} = region_names{j}; end
        end
    end
    matches = unique(matches);
end