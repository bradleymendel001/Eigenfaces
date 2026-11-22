%% =========================================================================
%  Face recognition with Eigenfaces (PCA) - assignment submission
%  -------------------------------------------------------------------------
%  Based on the Chapter 4 code (object_recognition.m). Uses only built-in
%  MATLAB functions - no toolboxes, no outside code or libraries.
%
%  Step 0  Split the data: 85% training, 15% testing
%  Step 1  PCA (eigenfaces) learned from the training photos only
%  Step 2  Accuracy for every number of eigenfaces k = 1..all  -> Graph 1
%  Step 3  Upper similarity threshold for every k              -> Graph 2
%
%  Put this file in the same folder as the s1, s2, ... folders and press Run.
%  The two graphs are saved in the "results" folder.
%% =========================================================================
clear; close all; clc;

% ----------------------------- Settings ----------------------------------
SEED          = 42;      % fixed random seed -> the same split every run
TEST_FRACTION = 0.15;    % 15% testing, 85% training
NPIX          = 112*92;  % every photo is 112 x 92 = 10304 pixels
rand('twister', SEED);   % makes randperm repeatable

dataDir    = fileparts(mfilename('fullpath'));          % folder of this file
if isempty(dataDir), dataDir = pwd; end
resultsDir = fullfile(dataDir, 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

%% ======================= Load all the face photos ========================
% Find the folders s1, s2, ... (one folder per person, 10 photos each)
listing  = dir(fullfile(dataDir, 's*'));
subjects = [];
for n = 1:numel(listing)
    if listing(n).isdir && ~isempty(regexp(listing(n).name, '^s\d+$', 'once'))
        subjects(end+1) = str2double(listing(n).name(2:end)); %#ok<AGROW>
    end
end
subjects = sort(subjects);
nSub = numel(subjects);
nPer = 10;

% Like the Chapter 4 code: every photo becomes one column of 10304 numbers
All_image_vec = uint8(zeros(NPIX, nSub*nPer));
labels        = zeros(1, nSub*nPer);              % which person each column is
for a = 1:nSub
    for j = 1:nPer
        img = imread(fullfile(dataDir, sprintf('s%d', subjects(a)), sprintf('%d.pgm', j)));
        col = j + (a-1)*nPer;
        All_image_vec(:, col) = reshape(img, NPIX, 1);
        labels(col) = subjects(a);
    end
end

%% ================ Step 0: split 85% training / 15% testing ===============
% 15% of 380 photos = 57 test photos. 57 / 38 people = 1.5, so every person
% gets 1 test photo and 19 randomly chosen people get a 2nd one. Everyone
% keeps photos in both sets.
N          = size(All_image_vec, 2);
nTestTotal = round(TEST_FRACTION * N);
base       = floor(nTestTotal / nSub);
extra      = nTestTotal - base*nSub;
nTestOf    = base * ones(1, nSub);
pick       = randperm(nSub);
nTestOf(pick(1:extra)) = nTestOf(pick(1:extra)) + 1;

trainIdx = []; testIdx = [];
for a = 1:nSub
    cols = find(labels == subjects(a));
    cols = cols(randperm(numel(cols)));             % shuffle this person's photos
    testIdx  = [testIdx,  cols(1:nTestOf(a))];       %#ok<AGROW>
    trainIdx = [trainIdx, cols(nTestOf(a)+1:end)];   %#ok<AGROW>
end
trainIdx = sort(trainIdx);  testIdx = sort(testIdx);

Train_image_vec = All_image_vec(:, trainIdx);
Test_image_vec  = All_image_vec(:, testIdx);
y_train = labels(trainIdx);   y_test = labels(testIdx);
nTrain  = numel(trainIdx);    nTest  = numel(testIdx);
fprintf('Training photos: %d (%.0f%%)   Test photos: %d (%.0f%%)\n', ...
        nTrain, 100*nTrain/N, nTest, 100*nTest/N);

%% ========== Step 1: PCA / eigenfaces (same lines as Chapter 4) ===========
% Average of all the training faces
avg = uint8(mean(Train_image_vec, 2));

% Subtract the average face from every training face ("normal vectors")
Normal_Train_Vec = [];
for i = 1:nTrain
    Normal_Train_Vec(:, i) = Train_image_vec(:, i) - avg;
end

% Covariance matrix and its eigenvectors/eigenvalues (sorted largest first)
M = Normal_Train_Vec' * Normal_Train_Vec;
[U, E, V] = svd(M);

% Eigenfaces: project the normal vectors onto the eigenvectors (keep ALL of them)
U = Normal_Train_Vec * U;
nEig = size(U, 2);

% Features: each row = one photo described by its eigenface weights
Training_Feature = [];
for i = 1:nTrain
    Training_Feature(i, :) = uint8(single(Normal_Train_Vec(:, i))' * single(U));
end
Test_Feature = [];
for t = 1:nTest
    Normal_Test_img    = uint8(Test_image_vec(:, t) - avg);  % remove TRAINING average
    Test_Feature(t, :) = uint8(single(Normal_Test_img)' * single(U));
end

%% ===== Step 2: accuracy for k = 1, 2, ..., all eigenfaces (Graph 1) ======
% For each k, a test photo is compared with every training photo using the
% first k eigenfaces: distance = norm(train - test, 2). The training photo
% with the LOWEST distance is the match. Correct if it is the same person.
% (The squared distance for k = the one for k-1 plus one more term, so we
%  add one eigenface at a time instead of starting again for every k.)
TrF = single(Training_Feature);   TeF = single(Test_Feature);
D2        = zeros(nTest, nTrain);     % running squared distances
location  = zeros(nTest, nEig);       % best-matching training photo, per test photo and k
bestScore = zeros(nTest, nEig);       % distance of that best match
maxScore  = zeros(1, nEig);           % largest distance of any pair (used in Step 3)
for k = 1:nEig
    D2 = D2 + (repmat(TeF(:, k), 1, nTrain) - repmat(TrF(:, k)', nTest, 1)).^2;
    Match_score = sqrt(D2);           % = norm(..., 2) for every test/train pair
    [score, loc] = min(Match_score, [], 2);
    location(:, k)  = loc;
    bestScore(:, k) = score;
    maxScore(k)     = max(Match_score(:));
end
correct  = y_train(location) == repmat(y_test', 1, nEig);   % right person? (test x k)
accuracy = 100 * mean(correct, 1);                          % accuracy (%) for each k
ks       = 1:nEig;
[bestAcc, best_k] = max(accuracy);

% ---------------------------- Graph 1 ----------------------------------
fig1 = figure('Name', 'Graph 1', 'Position', [100 100 900 420]);
plot(ks, accuracy, '-', 'Color', [0.16 0.47 0.84], 'LineWidth', 2); hold on;
plot(best_k, bestAcc, 'o', 'MarkerSize', 8, 'MarkerFaceColor', [0.16 0.47 0.84], 'MarkerEdgeColor', 'w');
plot([best_k best_k + 12], [bestAcc bestAcc - 15], '-', 'Color', [0.5 0.5 0.5]);   % pointer line
text(best_k + 13, bestAcc - 16, sprintf('Best: %.1f%% with %d eigenfaces', bestAcc, best_k));
grid on; box off; xlim([0 nEig + 1]); ylim([0 102]);
xlabel('Number of eigenfaces used (largest eigenvalue first)');
ylabel('Recognition accuracy (%)');
title('Number of eigenfaces vs. accuracy (lowest-distance match)');

%% ======= Step 3: upper similarity threshold for every k (Graph 2) ========
% similarity = 1 - (distance of best match) / (largest distance for that k)
%   1 = identical, 0 = as different as it gets (same scale for every k).
% The match is ACCEPTED only if similarity >= threshold.
% Accuracy = % of test photos matched to the right person AND accepted.
% Upper similarity threshold = the HIGHEST threshold that still gives the
% full Step 2 accuracy (the strictest setting that loses no correct match).
safeMax = maxScore;  safeMax(safeMax == 0) = 1;     % avoid 0/0 when all distances are 0
similarity = 1 - bestScore ./ repmat(safeMax, nTest, 1);
thresholds = 0:0.001:1;
nThr     = numel(thresholds);
accThr   = zeros(nEig, nThr);          % accuracy for every k and every threshold
upperThr = zeros(1, nEig);             % upper similarity threshold for every k
for k = 1:nEig
    accepted     = repmat(similarity(:, k), 1, nThr) >= repmat(thresholds, nTest, 1);
    accThr(k, :) = 100 * mean(accepted & repmat(correct(:, k), 1, nThr), 1);
    upperThr(k)  = thresholds(find(abs(accThr(k, :) - accuracy(k)) < 1e-9, 1, 'last'));
end

% ---------------------------- Graph 2 ----------------------------------
% Every k is drawn as a thin grey line; the best k and the Chapter 4 default
% (15 eigenfaces) are highlighted, with a dot at their upper threshold.
fig2 = figure('Name', 'Graph 2', 'Position', [100 100 900 460]); hold on;
hAll = plot(thresholds, accThr', '-', 'Color', [0.80 0.80 0.78], 'LineWidth', 0.5);
h15  = plot(thresholds, accThr(15, :), '-', 'Color', [0.92 0.41 0.20], 'LineWidth', 2);
hBest = plot(thresholds, accThr(best_k, :), '-', 'Color', [0.16 0.47 0.84], 'LineWidth', 2.5);
plot(upperThr(15), accuracy(15), 'o', 'MarkerSize', 8, 'MarkerFaceColor', [0.92 0.41 0.20], 'MarkerEdgeColor', 'w');
plot(upperThr(best_k), bestAcc, 'o', 'MarkerSize', 8, 'MarkerFaceColor', [0.16 0.47 0.84], 'MarkerEdgeColor', 'w');
text(upperThr(best_k) + 0.015, bestAcc + 1.5, sprintf('upper threshold %.3f', upperThr(best_k)));
text(upperThr(15) + 0.015, accuracy(15) + 1.5, sprintf('upper threshold %.3f', upperThr(15)));
legend([hBest h15 hAll(1)], {sprintf('k = %d eigenfaces (best)', best_k), ...
       'k = 15 eigenfaces (Chapter 4 default)', sprintf('every other k (1 to %d)', nEig)}, ...
       'Location', 'northeast');
legend boxoff; grid on; box off; xlim([0 1]); ylim([0 102]);
xlabel('Similarity threshold (match accepted only if similarity \geq threshold)');
ylabel('Accuracy (%)');
title('Similarity threshold vs. accuracy, for every number of eigenfaces');

%% ============================ Save & print ===============================
% Newer MATLAB follows the Mac's dark mode; force white graphs for the report.
figs = {fig1, 'Graph1_eigenfaces_vs_accuracy.png'; fig2, 'Graph2_similarity_vs_accuracy.png'};
for g = 1:2
    if exist('theme'), theme(figs{g, 1}, 'light'); end %#ok<EXIST>
    exportgraphics(figs{g, 1}, fullfile(resultsDir, figs{g, 2}), 'Resolution', 150, 'BackgroundColor', 'white');
end

fprintf('\nStep 2: best accuracy %.1f%% (%d of %d) with k = %d eigenfaces\n', ...
        bestAcc, sum(correct(:, best_k)), nTest, best_k);
fprintf('        k = 15: %.1f%%    all %d eigenfaces: %.1f%%\n', accuracy(15), nEig, accuracy(nEig));
fprintf('Step 3: upper similarity threshold %.3f at k = %d (k = 15: %.3f, all: %.3f)\n', ...
        upperThr(best_k), best_k, upperThr(15), upperThr(nEig));
fprintf('\n    k | accuracy | upper similarity threshold\n');
for k = [1 5 10 15 20 25 30 best_k 40 50 100 150 200 250 300 nEig]
    fprintf('%5d | %6.1f %% | %.3f\n', k, accuracy(k), upperThr(k));
end
fprintf('\nGraphs saved in: %s\n', resultsDir);
