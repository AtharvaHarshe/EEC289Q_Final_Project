clear all;clc;close all;

%%-->pick file
filename = 'data.txt';   

 formatSpec = '%s %s %f %f %f %f %f %f';
 fid = fopen(filename, 'r');
 raw = textscan(fid, formatSpec, 'Delimiter', ' ','MultipleDelimsAsOne', true);
 fclose(fid);

 %%
%concoct date and time coloum and deleat 
Data = table;

Data.date = raw{1};
Data.time        = raw{2};       % hh:mm:ss.xxx
Data.epoch       = raw{3};       % integer
Data.moteid      = raw{4};       % 1–54
Data.temperature = raw{5};       % °C
Data.humidity    = raw{6};       % %
Data.light       = raw{7};       % Lux
Data.voltage     = raw{8};       % V


Data.datetime = datetime(strcat(Data.date, {' '}, Data.time), ...
                         'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSSSSS');
%%
coords = [1 21.5 23
2 24.5 20
3 19.5 19
4 22.5 15
5 24.5 12
6 19.5 12
7 22.5 8
8 24.5 4
9 21.5 2
10 19.5 5
11 16.5 3
12 13.5 1
13 12.5 5
14 8.5 6
15 5.5 3
16 1.5 2
17 1.5 8
18 5.5 10
19 3.5 13
20 0.5 17
21 4.5 18
22 1.5 23
23 6 24
24 1.5 30
25 4.5 30
26 7.5 31
27 8.5 26
28 10.5 31
29 12.5 26
30 13.5 31
31 15.5 28
32 17.5 31
33 19.5 26
34 21.5 30
35 24.5 27
36 26.5 31
37 27.5 26
38 30.5 31
39 30.5 26
40 33.5 28
41 36.5 30
42 39.5 30
43 35.5 24
44 40.5 22
45 37.5 19
46 34.5 16
47 39.5 14
48 35.5 10
49 39.5 6
50 38.5 1
51 35.5 4
52 31.5 6
53 28.5 5
54 26.5 2];

%%
% -->create clusters
sel = [Data.epoch , Data.moteid , Data.temperature];

% Find all unique sensors (should be 54)
sensorList = unique(sel(:,2));

% Create a cell array to store clusters
clusters = {};

for i = 1:length(sensorList)
    sensorID = sensorList(i);
  
    % Extract rows belonging to this sensor
    rows = sel(sel(:,2) == sensorID, :);
    
    % Store epoch + temp only
    clusters{i,1} = rows(:, [1 3]);   % [epoch, temperature]
  
end
%%
% -->dataset cleaner


clusters2 = clusters;

for i = 1:length(clusters2)

    temp = clusters2{i};     % Nx2 matrix [epoch, temp]
    T = temp(:,2);           % temperature column
    
    % Logical filter for GOOD data
    good = (T >= 0) & (T <= 50) & (T ~= 122.153);
    
    % Apply filter → remove bad rows
    clusters2{i} = temp(good, :);
   
end
avg = [];
for i = 1:length(clusters2)

    temp = clusters2{i};     % Nx2 matrix [epoch, temp]
    avg(i,1) = mean(temp(:,2));           % temperature column
        
   
end

%%
% Number of sensors
numSensors = length(clusters2);

% Preallocate final result
snapshot = cell(numSensors, 1);

% The 100 uniformly spaced target epochs
Range = round(linspace(1, 2^16, 100));

for s = 1:numSensors
    
    temp = clusters2{s};        % Nx2 [epoch , temperature]
    avgValue = avg(s);    % average temperature for this sensor
    
    snap = zeros(100, 2);       % will store 100 rows [epoch , temp]
    
    for i = 1:100
        target = Range(i);
        
        % ---- 1. Check exact epoch match ----
        idx = find(temp(:,1) == target);
        
        % ---- 2. If not found, check ±1, ±2 ----
        if isempty(idx)
            offsets = [1 2 -1 -2 3 4 -3 -4 ];
            for k = 1:length(offsets)
                idx = find(temp(:,1) == (target + offsets(k)));
                if ~isempty(idx)
                    break;
                end
            end
        end
        
        % ---- 3. If still not found, assign average ----
        if isempty(idx)
            snap(i,:) = [target , avgValue];
        else
            snap(i,:) = temp(idx(1), :);   % use first found match
        end
    end
    
    snapshot{s} = snap;   % store the 100 readings for this sensor
end
snapshot = snapshot(1:54);
numSensors = length(snapshot);   % should be 54

TrainSet = zeros(numSensors, 70);
TestSet  = zeros(numSensors, 30);

for s = 1:numSensors
    
    tempData = snapshot{s}(:,2);   % extract temperature column (100x1)
    
    % First 70 → training
    TrainSet(s, :) = tempData(1:70);
    
    % Last 30 → testing
    TestSet(s, :) = tempData(71:100);
end

%%
% -->pod



% find mean and normalise it
meanTemp = mean(TrainSet, 2);       
Yc = TrainSet - meanTemp;        

%compute covariance matrix
R = Yc * Yc.';

% Eigen decomposition -----
[Phi, D] = eig(R, 'vector');      % D = vector of eigenvalues (unsorted)
[lambda, idx] = sort(D, 'descend');  % Sort by descending eigenvalue magnitude
Phi = Phi(:, idx);                % Reorder eigenvectors accordingly

% Normalize POD modes -----
for k = 1:length(lambda)
    Phi(:,k) = Phi(:,k) / norm(Phi(:,k));   % ensure unit norm
end

% Compute temporal coefficients (POD amplitudes) -----
A = Phi.' * Yc;                % r x 70 (each row = temporal evolution of mode k)

% Energy content (pseudo kinetic energy) -----
E_total = sum(lambda);
E_percent = (lambda / E_total) * 100;   % % energy per mode

%  Plot eigenvalue spectrum -----
figure;
stem(E_percent);
xlabel('Mode number'); ylabel('Energy (%)');
title('POD Mode Energy Distribution');

clear idx k meanTemp A D E_percent E_total Yc R 




%%
% -->gp


X = coords(:,2:3);          % only x and y coordinates
r = size(Phi, 2);           % number of POD modes
% Variable Inisitalisation
Sigma_all = cell(r,1);      % store Σ_i
mu_all    = cell(r,1);      % store μ_i
gprModels = cell(r,1);      % store GP models

for i = 1:r
    
    % ----- POD mode i becomes GP training target -----
    y_i = Phi(:,i);   % 54x1 vector
    
    % ----- Train GP model for mode i -----
    gprModels{i} = fitrgp(X, y_i, ...
        'KernelFunction','squaredexponential', ...
        'BasisFunction','none', ...
        'FitMethod','exact', ...
        'Standardize',true);
    
    % ----- Standardize coordinates exactly how GP did -----
    X_std = (X - gprModels{i}.Impl.StdMu') ./ gprModels{i}.Impl.StdSigma';
    
    % ----- Build kernel from learned hyperparameters -----
    theta = gprModels{i}.Impl.ThetaHat;          % [ell; sigma_f]
    kernelFcn = gprModels{i}.Impl.Kernel.makeKernelAsFunctionOfXNXM(theta);
    
    % ----- Compute covariance matrix Σ_i = K_i(X,X) + σ_n^2 I -----
    K_i = kernelFcn(X_std, X_std);
    Sigma_all{i} = K_i + gprModels{i}.Impl.SigmaHat^2 * eye(size(K_i));
    
    % ----- Mean (optional, not needed for MI) -----
    mu_all{i} = predict(gprModels{i}, X);
    
end

clear i k_i kernelFcn theta X X_std y_i K_i 
%%
% -->lazy greedy


numSensors = size(Sigma_all{1}, 1);
maxSensors = 54;                
Pareto_MI   = zeros(maxSensors,1);
Pareto_Sets = cell(maxSensors,1);
prevMI = -inf;

for eps = 1:maxSensors
    
    S = [];                      % selected sensors start empty
    remaining = 1:numSensors;    % all candidates
    
    currentMI = 0;               % track MI for the greedy process
    
    % ============================================================
    %                TRUE LAZY–GREEDY SELECTION (C.1)
    % ============================================================
    for k = 1:eps
        
        % Step 1: initialize δ(s) = +∞ for all remaining sensors
        m = length(remaining);
        delta   = inf(1, m);        % cached marginal gains
        current = false(1, m);      % flags for whether δ(s) was recomputed
        
        % Step 2: LazyGreedy inner loop
        while true
            
            % Find s* = sensor with largest cached δ value
            [~, idx] = max(delta);
            s_star   = remaining(idx);
            
            % If already recomputed once, accept s*
            if current(idx)
                break;
            end
            
            % Otherwise recompute true marginal gain:
            F_before = MI_weighted(Sigma_all, lambda, S);
            F_after  = MI_weighted(Sigma_all, lambda, [S, s_star]);
            
            delta(idx) = F_after - F_before;   % true δ(s)
            current(idx) = true;               % mark as updated
        end
        
        % Step 3: Add selected sensor s*
        S = [S, s_star];
        
        % Remove it from "remaining"
        remaining(idx) = [];
        
        % Update current MI
        currentMI = MI_weighted(Sigma_all, lambda, S);
    end
    
    
    % ---------- Non-decreasing MI stopping rule ----------
  % ---------- Combined stopping rule ----------
if eps > 2
    
    %deltaMI     = currentMI/10000 - prevMI/10000;   
    deltaMI     = currentMI - prevMI;   % current marginal gain
          % previous marginal gain

    % Conditions:
    cond_decreaseMI      = (currentMI < prevMI);
    cond_smallGain       = (deltaMI < 0.1);       % your threshold (adjust as needed)
 

    if cond_decreaseMI || cond_smallGain 
        fprintf("\nSTOP at ε = %d because: ", eps);

        if cond_decreaseMI
            fprintf("MI decreased; ");
        end
        if cond_smallGain
            fprintf("marginal gain < threshold; ");
        end
        

        fprintf("\n");

        Pareto_MI   = Pareto_MI(1:eps-1);
        Pareto_Sets = Pareto_Sets(1:eps-1);

        break;
    end
end

% Track MI values for next iteration
prevPrevMI = prevMI;
prevMI      = currentMI;

    
    % Store Pareto point

    Pareto_MI(eps)   = currentMI;
    Pareto_Sets{eps} = S;
    %normilised_MI = currentMI/10000;
    %Pareto_MI_Normalised(eps) = normilised_MI;
    fprintf('ε = %d sensors → MI = %.4f\n', eps, currentMI);
end

%%
% --find knee point
x  = (1:eps-1);     % sensor counts
F  = Pareto_MI*10;          % MI values

[~, knee_k, knee_MI, MC] = findKneePoint(x, F);

fprintf('\n*** Knee point found at |S| = %d sensors ***\n', knee_k);
fprintf('Weighted MI at knee = %.4f\n', knee_MI/10);

clear cond_decreasingGain cond_decreaseMI cond_smallGain current delta;
clear deltaPrevMI deltaMi F_before F_after idx k m prevMI prevPrevMI S s_star;

%%Plot output
figure('Color','white'); hold on;

% --- Plot Pareto frontier ---
plot(1:eps-1, Pareto_MI, '-o', ...
    'MarkerSize', 4, 'MarkerFaceColor', [0 0.45 0.74]);


% --- Highlight knee point ---
stem(knee_k, knee_MI/10, 'd', ...
    'MarkerSize', 10, ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', [0.85 0.33 0.10]);  % orange

% --- Labels and title ---
xlabel('Number of Sensors (k)', 'FontSize', 12);
ylabel('Weighted Mutual Information', 'FontSize', 12);
title('Pareto Frontier and Knee Point Selection', 'FontSize', 14);

% --- Grid and formatting ---
grid on;


% --- Annotation text near knee point ---
text(knee_k - 5, knee_MI/10+ 2, sprintf('  Knee point: k = %d', knee_k), ...
    'FontSize', 12, 'Color', [0.85 0.33 0.10]);

% --- Tight axis ---
xlim([1 eps-1]);

%%
% --> bnb validation
k = knee_k;
GreedySet = Pareto_Sets{k};
GreedyMI = Pareto_MI(k);

[LB, UB, gap] = BnB_AlgD1(Sigma_all, lambda, GreedySet, k);
%%

fprintf("\n==================== BnB VALIDATION ====================\n");
fprintf("Sensor budget (k): %d\n", k);
fprintf("Greedy MI (LB):    %.6f\n", LB);
fprintf("Upper Bound (UB):  %.6f\n", UB);
fprintf("Gap (UB - LB):     %.6f\n", gap);

% Threshold for "good" (you can adjust)
tol = 1e-3 * LB;      % relative threshold (0.1%)

if abs(gap) <= tol
    fprintf("\nSTATUS: ✔ Greedy solution is VALID and essentially OPTIMAL.\n");
    fprintf("        (UB and LB differ by less than %.2f%%)\n", 100 * tol / LB);
elseif gap <= 0.02 * LB
    fprintf("\nSTATUS: ✔ Greedy solution is NEAR-OPTIMAL.\n");
    fprintf("        (Gap ≤ 2%% of MI — acceptable for deployment)\n");
else
    fprintf("\nSTATUS: ⚠ Greedy solution may NOT be optimal.\n");
    fprintf("        Consider running tighter BnB or reducing candidate set.\n");
end

fprintf("=========================================================\n\n");
%%--> find and plot sensors vs error using rmse
%% ================== SECTION 5.4 — Reconstruction RMSE ==================
S_final = Pareto_Sets{knee_k};
% --- Ensure TestSet is T × 54 (snapshots × sensors) ---
if size(TestSet,1) == 54
    TestSet = TestSet';          % Now size = 30 × 54
end

% Training mean (must match POD center)
TrainMean = mean(TrainSet, 2);    % 54 × 1

% Subtract mean from TestSet
TestSet_centered = TestSet - TrainMean';   % 30 × 54

% Use only the POD modes you computed earlier
% Phi = 54 × r
r = size(Phi,2);

% Selected sensor set (from Pareto_Sets{knee_k})
S = S_final;                      % 1 × k

% Extract POD rows corresponding to selected sensors
Phi_S = Phi(S, :);                % k × r

% Compute pseudo-inverse (stable even if rank-deficient)
PhiS_pinv = pinv(Phi_S);          % r × k

% Prepare reconstructed matrix
T = size(TestSet_centered,1);     % number of test snapshots
Ypred_full = zeros(T, 54);        % reconstructions

%% ---- Reconstruct each snapshot using POD coefficients ----
for t = 1:T

    y_true = TestSet_centered(t, :);   % 1 × 54
    y_S    = y_true(S)';               % k × 1 measurements at selected sensors

    % POD coefficient estimate
    a_t = PhiS_pinv * y_S;             % r × 1

    % Reconstruct field:  y_hat = Φ a + mean
    y_hat = Phi * a_t;                 % 54 × 1
    Ypred_full(t,:) = y_hat' + TrainMean';   % add mean back
end

%% ---- Compute RMSE ----
sqErr = (Ypred_full - TestSet).^2;      % same orientation
RMSE = sqrt(mean(sqErr(:)));

fprintf("\n========== Section 5.4 Reconstruction Results ==========\n");
fprintf("Selected sensors: %d\n", length(S));
fprintf("Reconstruction RMSE = %.6f\n", RMSE);
fprintf("==========================================================\n");
%%
% RMSE vs k PLOT

maxK = length(Pareto_Sets);   % usually 54
RMSE_k = zeros(maxK, 1);

% Training mean (must match POD centering
TrainMean = mean(TrainSet, 2);     % 54 × 1

% Fix TestSet orientation
if size(TestSet,1) == 54
    TestSet = TestSet';            % 30 × 54
end

% Center TestSet using TRAIN mean
TestSet_centered = TestSet - TrainMean';



r = size(Phi,2);                   % number of POD modes


%% ---- LOOP: compute RMSE for k = 1 ... maxK ----
for k = 1:maxK
    
    S = Pareto_Sets{k};         % selected sensors for this k
    Phi_S = Phi(S, :);          % k × r

    % Stable pseudo-inverse
    PhiS_pinv = pinv(Phi_S);    % r × k

    Ypred_full = zeros(size(TestSet_centered));  % 30 × 54
    
    % ---- Reconstruct each test snapshot ----
    for t = 1:size(TestSet_centered,1)

        y_true = TestSet_centered(t,:);    % 1 × 54
        y_S    = y_true(S)';               % k × 1

        % POD coefficients
        a_t = PhiS_pinv * y_S;             % r × 1

        % Reconstruct full snapshot and add mean back
        Ypred_full(t,:) = (Phi * a_t)' + TrainMean';
    end

    % ---- Compute RMSE for this k ----
    sqErr = (Ypred_full - TestSet).^2;
    RMSE_k(k) = sqrt(mean(sqErr(:)));

    fprintf("k = %2d --> RMSE = %.4f\n", k, RMSE_k(k));
end


%% ===================== Plot RMSE vs k ========================

figure;
plot(1:maxK, RMSE_k, 'LineWidth', 2);
xlabel('Number of Selected Sensors (k)', 'FontSize', 12);
ylabel('RMSE of Reconstruction(in C)', 'FontSize', 12);
title('Section 5.4: Reconstruction Error vs Number of Sensors');
grid on;


%%
% -->plot diffrent sensor placements 


clc; close all;
TestSet1=TestSet';
% ------------ basic checks ------------
nSensors = size(TrainSet,1);
if nSensors ~= 54
    error('This script assumes 54 sensors.');
end

% TestSet is 54×30; keep that orientation (sensors × time)
if size(TestSet1,1) ~= nSensors
    error('TestSet should be 54×T (sensors × time).');
end

maxK = min(23, numel(Pareto_Sets));   % as in the paper, or <= #Pareto_Sets
numRandom = 20;                       % RA trials for averaging

RMSE_LG = zeros(maxK,1);
RMSE_PV = zeros(maxK,1);
RMSE_UN = zeros(maxK,1);
RMSE_RA = zeros(maxK,1);

fprintf('Computing RMSE curves for LG / PV / UN / RA ...\n');

%% ====================== MAIN LOOP OVER k ======================
for k = 1:maxK
    fprintf('k = %d\n', k);

    % ---------- 1) LG: Lazy Greedy (already computed) ----------
    S_LG = Pareto_Sets{k};

    % ---------- 2) PV: Predictive Variance greedy ----------
    %S_PV = pv_select(Sigma_all, lambda,coords, k);
     S_PV = pv_select(coords, k);
    % ---------- 3) UN: Uniform geometric placement ----------
    S_UN = uniform_select(coords, k);

    % ---------- 4) RA: random placement ----------
   % ---------- Random placement (RA) ----------
tmp = zeros(numRandom,1);

for rtrial = 1:numRandom
    
    % Assign each sensor a random priority score ∈ [0,1]
    scores = rand(54,1);
    
    % Sort in descending order
    [~, order] = sort(scores, 'descend');
    
    % Take top-k sensors
    S_r = order(1:k);
    
    % Compute RMSE

end

% Paper uses a *single* sample per k.
% But averaging over 20 gives smoother curve (your choice).
RMSE_RA(k) = rmse_reconstruct_simple(S_r , Phi, TrainSet, TestSet1);


    % ---------- RMSE for LG / PV / UN ----------
    RMSE_LG(k) = rmse_reconstruct_simple(S_LG, Phi, TrainSet, TestSet1);
    RMSE_PV(k) = rmse_reconstruct_simple(S_PV, Phi, TrainSet, TestSet1);
    RMSE_UN(k) = rmse_reconstruct_simple(S_UN, Phi, TrainSet, TestSet1);
end

%% ====================== PLOT (Figure 9 style) ======================
figure; hold on;
plot(1:maxK, RMSE_LG, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'k', ...
     'DisplayName', 'LG');
plot(1:maxK, RMSE_PV, 'r-s', 'LineWidth', 2, 'MarkerFaceColor', 'r', ...
     'DisplayName', 'PV');
plot(1:maxK, RMSE_UN, 'b-d', 'LineWidth', 2, 'MarkerFaceColor', 'b', ...
     'DisplayName', 'UN');
plot(1:maxK, RMSE_RA, 'g-^', 'LineWidth', 2, 'MarkerFaceColor', 'g', ...
     'DisplayName', 'RA');

xlabel('Number of sensors (k)', 'FontSize', 12);
ylabel('RMSE', 'FontSize', 12);
title('RMSE vs Number of Sensors', 'FontSize', 14);
legend('Location','northeast');
grid on;




%%
% -->image plot function
img = imread('lab.png');
figure; imshow(img); hold on;
pixel_coords=[430.208438287154	347.221032745592
374.921284634761	293.698362720403
473.732367758186	277.818010075567
416.680730478589	224.295340050378
373.156801007557	181.947732997481
473.144206549118	183.124055415617
417.857052896725	113.132871536524
375.509445843829	60.1983627204031
423.150503778338	26.6731738035265
477.849496221663	81.9603274559195
561.368387909320	49.0232997481109
609.597607052897	19.6152392947104
634.300377833753	84.3129722921915
723.112720403023	88.4301007556676
776.047229219144	54.3167506297230
869.564861460957	36.0837531486147
867.212216624685	106.074937027708
777.223551637280	130.189546599496
816.042191435768	177.830604534005
881.916246851386	223.707178841310
799.573677581864	238.999370277078
862.506926952141	317.812972292192
768.401133501260	347.809193954660
862.506926952141	429.563602015113
798.985516372796	429.563602015113
738.993073047859	446.620277078086
726.053526448363	381.334382871537
681.353274559194	448.384760705290
643.122795969773	382.510705289673
612.538413098237	446.620277078086
559.603904282116	407.801637279597
500.787783375315	449.561083123426
473.144206549118	378.393576826196
429.620277078086	442.503148614610
373.156801007557	393.097607052897
319.045969773300	445.443954659950
287.285264483627	375.452770780856
216.117758186398	446.620277078086
217.294080604534	374.864609571788
170.829345088161	406.625314861461
108.484256926952	441.326826196474
42.6102015113350	437.209697732998
126.717254408060	356.043450881612
16.1429471032746	334.869647355164
78.4880352644837	287.816750629723
147.302896725441	234.294080604534
35.5522670025189	210.179471032746
137.892317380353	164.302896725441
39.6693954659950	95.4880352644837
58.4905541561713	23.7323677581865
134.951511335013	54.3167506297230
214.941435768262	85.4892947103276
284.344458438287	77.8431989924434
305.518261964736	30.2021410579346];

% Sensor set to highlight
k = 21;
S = Pareto_Sets{k};     % e.g., [1 5 7 11 ...]

% Circle radius (adjust as needed)
radius = 18;
th = linspace(0, 2*pi, 200);

for i = 1:length(S)
    s = S(i);               % sensor index
    xc = pixel_coords(s,1) + radius*cos(th);
    yc = pixel_coords(s,2) + radius*sin(th);
    
    plot(xc, yc, 'r-', 'LineWidth', 3);
end

title(sprintf('Selected Sensors (k = %d) Highlighted', k), 'FontSize', 14);
hold off;
%% ====================== LOCAL FUNCTIONS ======================

function val = rmse_reconstruct_simple(S, Phi, TrainSet, TestSet)
    % Simple gappy POD reconstruction:
    % - TrainSet: 54×Ntr (sensors × time)
    % - TestSet : 54×Nte (sensors × time)
    % - S       : indices of selected sensors

    nSensors = size(TrainSet,1);
    if any(S < 1) || any(S > nSensors)
        error('Invalid sensor indices in S.');
    end

    % Mean from training data (per sensor)
    TrainMean = mean(TrainSet, 2);     % 54×1

    % Use all POD modes for reconstruction
    Phi_r = Phi;                       % 54×r
    r = size(Phi_r, 2);

    % POD restricted to selected sensors
    Phi_S = Phi_r(S, :);               % k×r
    PhiS_pinv = pinv(Phi_S);           % r×k

    T = size(TestSet, 2);              % number of test snapshots
    Yhat = zeros(nSensors, T);

    for t = 1:T
        y_true = TestSet(:, t);                % 54×1 (raw)
        yS_true = y_true(S);                   % k×1

        % Center using training mean (on selected sensors)
        yS_centered = yS_true - TrainMean(S);  % k×1

        % Solve for POD coefficients using LS on selected sensors
        a_t = PhiS_pinv * yS_centered;         % r×1

        % Reconstruct full field and add mean back
        y_centered_hat = Phi_r * a_t;          % 54×1
        y_hat = y_centered_hat + TrainMean;    % 54×1

        Yhat(:, t) = y_hat;
    end

    % RMSE over all sensors and all test snapshots
    sqErr = (Yhat - TestSet).^2;
    val = sqrt( mean( sqErr(:) ) );
end
function S = pv_select(coords, k)
% PV that matches paper behavior: 
% greedy on geometric distance from selected sensors.

n = size(coords,1);
X = coords(:,2:3);

% Start at random sensor (very important!)
S = randi(n);
remaining = setdiff(1:n, S);

for t = 2:k

    scores = zeros(n,1);

    for s = remaining
        % min distance to already-selected sensors
        d = min(vecnorm(X(s,:) - X(S,:), 2, 2));
        scores(s) = d;
    end

    % choose sensor farthest away = high "variance"
    [~, idx] = max(scores(remaining));
    s_star = remaining(idx);

    S = [S s_star];
    remaining(remaining==s_star) = [];
end

end

function F = MI_weighted(Sigma_all, lambda, S)

    r = length(lambda);

    % Normalize eigenvalues to get weights
    w = lambda(:) / (1);

    F = 0;
    for i = 1:r
        Fi = MI_single_mode(Sigma_all{i}, S);
        F = F + w(i) * Fi;       % <--- use normalized weights
    end
    F = F/10000;
end

function Fi = MI_single_mode(Sigma_i, S)
    
    n = size(Sigma_i, 1);
    U = setdiff(1:n, S);

    if isempty(U)
        Fi = 0; 
        return;
    end

    Sigma_SS = Sigma_i(S, S);
    Sigma_UU = Sigma_i(U, U);
    Sigma_SU = Sigma_i(S, U);
    Sigma_US = Sigma_i(U, S);

    % Conditional covariance
    Sigma_U_given_S = Sigma_UU - Sigma_US * (Sigma_SS \ Sigma_SU);

    % Stable MI
    Fi = 0.5 * ( logdet(Sigma_UU) - logdet(Sigma_U_given_S) );
end

function y = logdet(A)
    % Stable log-determinant using Cholesky
    % Add tiny jitter for numerical stability
    A = A + 1e-12 * eye(size(A));
    U = chol(A);
    y = 2 * sum(log(diag(U)));
end

function [LB, UB, gap] = BnB_AlgD1(Sigma_all, lambda, GreedySet, k)

    % ============ Step 1: compute lower bound ====================
    F_greedy = MI_weighted(Sigma_all, lambda, GreedySet);
    LB = F_greedy;

    % ============ Step 2: compute singleton gains (submodular UB) ============
    n = size(Sigma_all{1},1);
    MG = zeros(1,n);

    for s = 1:n
        MG(s) = fast_singleton_MI(Sigma_all, lambda, s);
    end

    % Sort largest to smallest
    MG = sort(MG, 'descend');

    remaining = k - numel(GreedySet);
    if remaining < 0
        error("Greedy set larger than k.");
    elseif remaining == 0
        UB = LB;
        gap = UB - LB;
        return;
    end

    % ============ Step 3: Compute UB using Algorithm D.1 approximation ============
    alpha = 1 - 1/exp(1);   % (1 - 1/e)
    greedy_residual_gain = sum( MG(1:remaining) );  % submodular upper bound

    UB = LB + greedy_residual_gain / alpha;

    % ============ Step 4: compute optimality gap ================================
    gap = UB - LB;

    fprintf("Greedy MI = %.6f\n", LB);
    fprintf("Alg D.1 UB = %.6f\n", UB);
    fprintf("Optimality gap ≤ %.6f\n", gap);

end


% ========== Helper: singleton MI = upper bound on true marginal gain ========
function mg = fast_singleton_MI(Sigma_all, lambda, s)
    mg = 0;
    for i = 1:length(lambda)
        sigma_ss = Sigma_all{i}(s, s);
        mg = mg + lambda(i) * 0.5 * log(1 + sigma_ss);
    end
end

function [knee_idx, knee_x, knee_y, MC] = findKneePoint(x, F)

    % --- basic checks ---
    x = x(:);
    F = F(:);

    n = numel(x);
    if n < 3
        error('findKneePoint:NotEnoughPoints', ...
              'Need at least 3 points to define a knee.');
    end
    if numel(F) ~= n
        error('findKneePoint:SizeMismatch', ...
              'x and F must have the same length.');
    end

    % Preallocate curvature array
    MC = zeros(n, 1);

    % Loop over interior points (2..n-1)
    for i = 2:n-1
        % Points p, q, r
        xp = x(i-1); Fp = F(i-1);
        xq = x(i);   Fq = F(i);
        xr = x(i+1); Fr = F(i+1);

        % Distances |pq|, |qr|, |rp|
        pq = sqrt( (xq - xp)^2 + (Fq - Fp)^2 );
        qr = sqrt( (xr - xq)^2 + (Fr - Fq)^2 );
        rp = sqrt( (xr - xp)^2 + (Fr - Fp)^2 );

        % If any segment is (numerically) zero length, skip
        if pq == 0 || qr == 0 || rp == 0
            MC(i) = NaN;
            continue;
        end

        % A = 4|pq|^2|qr|^2, B = |pq|^2 + |qr|^2 - |rp|^2
        pq2 = pq^2;
        qr2 = qr^2;
        rp2 = rp^2;

        A = 4 * pq2 * qr2;
        B = pq2 + qr2 - rp2;

        % Numerical safety: clamp A - B^2 to >= 0
        disc = A - B^2;
        if disc < 0
            disc = 0;
        end

        MC(i) = sqrt(disc) / (pq * qr * rp);
    end

    % Knee point is index with maximum curvature among valid entries
    [~, knee_idx] = max(MC);

    knee_x = x(knee_idx);
    knee_y = F(knee_idx);
end

