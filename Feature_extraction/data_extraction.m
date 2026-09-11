%% 1. Initialization and Loading Lists
clear; clc; close all;
path = 'C:\data-stavropoulou\'; 
fid = fopen(fullfile(path,'NI','NormalAdults.txt'), 'r');
NormalList = textscan(fid, '%s', 'Delimiter','\n'); NormalList = NormalList{1};
fclose(fid);
fid = fopen(fullfile(path,'mTBI','TBI.csv'), 'r');
TBIList = textscan(fid, '%s', 'Delimiter','\n'); TBIList = TBIList{1};
fclose(fid);
total_subjects = length(NormalList) + length(TBIList);
num_features   = 10;
final_features = zeros(total_subjects, num_features); 
y_labels       = [zeros(length(NormalList),1); ones(length(TBIList),1)];
% Store networks and class sequences for all subjects
all_subjects_networks = cell(total_subjects, 1);
all_subjects_class    = cell(total_subjects, 1);
rng(42);

%% 2. Main Processing Loop
for subj = 1:total_subjects
    if subj <= length(NormalList)
        current_name = NormalList{subj};
        strR_i = fullfile(path, 'NI', ['th_dom_', current_name, '.mat']);
    else
        current_name = TBIList{subj - length(NormalList)};
        strR_i = fullfile(path, 'mTBI', ['th_iplv_dom_TBI_', current_name, '.mat']);
    end
    
    fprintf('Processing %d/%d: %s\n', subj, total_subjects, current_name);
    load(strR_i);
    
    % --- A. Laplacian & Eigenvalues ---
    slides = size(gce,1);
    neigenvalues = zeros(slides, 90);
    all_networks = cell(slides, 1);
    for i = 1:slides
        network = gce{i,1}.weighted6;
        all_networks{i} = network;
        degree_mat = diag(sum(network, 2));
        nlaplacian = eye(90) - (degree_mat^-0.5) * network * (degree_mat^-0.5);
        nlaplacian(isnan(nlaplacian)) = 0;
        neigenvalues(i, :) = sort(real(eig(nlaplacian)))'; 
    end
    
    % --- B. Neural Gas (k=2) ---
    [prot, class] = Vector_Quantization(neigenvalues, 2, 15);
    % --- Sorting states ---
    mean_eig_1 = mean(prot(1,:));
    mean_eig_2 = mean(prot(2,:));
    
    if mean_eig_1 > mean_eig_2
        new_class = class;
        new_class(class == 1) = 2;
        new_class(class == 2) = 1;
        class = new_class;
        prot = [prot(2,:); prot(1,:)];
    end
    
    % Store for group-level analysis
    all_subjects_networks{subj} = all_networks;
    all_subjects_class{subj}    = class;

    % --- C. Feature Extraction ---
    trans_prob  = sum(diff(class) ~= 0) / (length(class) - 1);
    occupancy_1 = sum(class == 1) / length(class);
    m1 = [0; (class(:) == 1); 0]; 
    d1 = diff(m1);
    starts1    = find(d1 == 1);
    ends1      = find(d1 == -1);
    durations1 = ends1 - starts1;
    dwell_1    = mean(durations1);
    m2 = [0; (class(:) == 2); 0];
    d2 = diff(m2);
    starts2    = find(d2 == 1);
    ends2      = find(d2 == -1);
    durations2 = ends2 - starts2;
    dwell_2    = mean(durations2);
    if isempty(durations1), dwell_1 = 0; end
    if isempty(durations2), dwell_2 = 0; end
    state_metrics = zeros(1, 6); 
    for k = 1:2
        idx = find(class == k);
        if ~isempty(idx)
            mean_net = zeros(90, 90);
            for m = 1:length(idx)
                mean_net = mean_net + all_networks{idx(m)};
            end
            mean_net = mean_net / length(idx);
            L   = weight_conversion(mean_net, 'lengths');
            eff = efficiency_wei(mean_net); 
            C_vec           = clustering_coef_wu(mean_net);
            avg_clustering  = mean(C_vec);
            [D, ~]          = distance_wei(L);
            [lambda, ~, ~, ~, ~] = charpath(D);
            state_metrics((k-1)*3 + 1 : k*3) = [eff, avg_clustering, lambda];
        end
    end
    
    final_features(subj, :) = [ ...
        trans_prob, ...
        occupancy_1, ...
        dwell_1, ...
        dwell_2, ...
        state_metrics(1), ...   % GE  S1
        state_metrics(2), ...   % CC  S1
        state_metrics(3), ...   % CPL S1
        state_metrics(4), ...   % GE  S2
        state_metrics(5), ...   % CC  S2
        state_metrics(6), ...   % CPL S2
    ];
end

%% 3. Group-Level Mean Connectivity Matrices
fprintf('\nComputing group-level mean connectivity matrices...\n');
mean_net_s1_healthy = zeros(90, 90);
mean_net_s2_healthy = zeros(90, 90);
mean_net_s1_mtbi    = zeros(90, 90);
mean_net_s2_mtbi    = zeros(90, 90);
count_s1_h = 0; count_s2_h = 0;
count_s1_m = 0; count_s2_m = 0;
for subj = 1:total_subjects
    networks = all_subjects_networks{subj};
    class    = all_subjects_class{subj};
    
    for t = 1:length(class)
        net = networks{t};
        
        if y_labels(subj) == 0  % healthy
            if class(t) == 1
                mean_net_s1_healthy = mean_net_s1_healthy + net;
                count_s1_h = count_s1_h + 1;
            else
                mean_net_s2_healthy = mean_net_s2_healthy + net;
                count_s2_h = count_s2_h + 1;
            end
        else  % mTBI
            if class(t) == 1
                mean_net_s1_mtbi = mean_net_s1_mtbi + net;
                count_s1_m = count_s1_m + 1;
            else
                mean_net_s2_mtbi = mean_net_s2_mtbi + net;
                count_s2_m = count_s2_m + 1;
            end
        end
    end
end
mean_net_s1_healthy = mean_net_s1_healthy / count_s1_h;
mean_net_s2_healthy = mean_net_s2_healthy / count_s2_h;
mean_net_s1_mtbi    = mean_net_s1_mtbi    / count_s1_m;
mean_net_s2_mtbi    = mean_net_s2_mtbi    / count_s2_m;
fprintf('Counts — S1 healthy: %d | S2 healthy: %d | S1 mTBI: %d | S2 mTBI: %d\n', ...
        count_s1_h, count_s2_h, count_s1_m, count_s2_m);

%% 4. Saving
save('dynamic_features_for_kan.mat', 'final_features', 'y_labels');
save('mean_connectivity_matrices.mat', ...
     'mean_net_s1_healthy', 'mean_net_s2_healthy', ...
     'mean_net_s1_mtbi',    'mean_net_s2_mtbi');
% save('per_subject_iPLV.mat', 'mean_iPLV_s1', 'mean_iPLV_s2', 'y_labels');
disp('Extraction completed successfully!');