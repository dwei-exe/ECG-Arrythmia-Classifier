%% Enhanced Training and Validation using AlexNet with Comprehensive Analysis
% Dataset: Lead II Scalogram Images (227x227 pixels)
% Classes: SR (Sinus Rhythm), SB (Sinus Bradycardia), AFIB (Atrial Fibrillation)

clear; clc; close all;

%% Setup and Data Loading
DatasetPath = 'C:\Users\henry\Downloads\ECG-Dx\Lead2_Scalogram_Dataset';
ResultsPath = 'C:\Users\henry\Downloads\ECG-Dx\Training_Results';

% Create results directory if it doesn't exist
if ~exist(ResultsPath, 'dir')
    mkdir(ResultsPath);
end

% Reading Images from Image Database Folder
fprintf('Loading dataset from: %s\n', DatasetPath);
images = imageDatastore(DatasetPath, "IncludeSubfolders", true, "LabelSource", "foldernames");

% Display dataset information
fprintf('Total images: %d\n', numel(images.Files));
labelCounts = countEachLabel(images);
disp('Class distribution:');
disp(labelCounts);

% Split the data into training and validation sets
numTrainFiles = 6476;
[TrainImages, TestImages] = splitEachLabel(images, numTrainFiles, 'randomized');

fprintf('Training images: %d\n', numel(TrainImages.Files));
fprintf('Validation images: %d\n', numel(TestImages.Files));

%% Network Architecture Setup
net = alexnet; % Import pretrained AlexNet
layersTransfer = net.Layers(1:end-3); % Preserve all layers except the last three
numClasses = 3; % Number of output classes: SR, SB, AFIB

% Define layers of AlexNet
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, "WeightLearnRateFactor", 20, "BiasLearnRateFactor", 20)
    softmaxLayer
    classificationLayer
];

%% Training Options
options = trainingOptions("sgdm", ...
    "MiniBatchSize", 32, ...
    "MaxEpochs", 8, ...
    "InitialLearnRate", 1e-4, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", TestImages, ...
    "ValidationFrequency", 20, ...
    "Verbose", false, ...
    "Plots", "training-progress", ...
    "OutputNetwork", "last-iteration");

%% Training AlexNet
fprintf('\nStarting training...\n');
tic;
[netTransfer, trainInfo] = trainNetwork(TrainImages, layers, options);
trainingTime = toc;
fprintf('Training completed in %.2f seconds (%.2f minutes)\n', trainingTime, trainingTime/60);

%% Save the Trained Model
modelSavePath = fullfile(ResultsPath, 'trained_alexnet_model.mat');
save(modelSavePath, 'netTransfer', 'trainInfo', 'trainingTime');
fprintf('Model saved to: %s\n', modelSavePath);

%% Model Evaluation - Get Predictions and Scores
fprintf('\nEvaluating model performance...\n');

% Get prediction scores for ROC analysis
[Ypred, scores] = classify(netTransfer, TestImages);
Yvalidation = TestImages.Labels;

% Calculate overall accuracy
accuracy = sum(Ypred == Yvalidation) / numel(Yvalidation);
fprintf('Overall Accuracy: %.4f (%.2f%%)\n', accuracy, accuracy * 100);

%Plotting Confusion Matrix
plotconfusion(Yvalidation,Ypred);

%% Detailed Performance Metrics
classes = categories(Yvalidation);
numClasses = length(classes);

% Initialize metrics arrays
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);
specificity = zeros(numClasses, 1);

% Calculate confusion matrix
confMat = confusionmat(Yvalidation, Ypred);

% Calculate metrics for each class
for i = 1:numClasses
    TP = confMat(i, i);
    FP = sum(confMat(:, i)) - TP;
    FN = sum(confMat(i, :)) - TP;
    TN = sum(confMat(:)) - TP - FP - FN;
    
    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    specificity(i) = TN / (TN + FP);
end

% Handle NaN values (in case of division by zero)
precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1Score(isnan(f1Score)) = 0;
specificity(isnan(specificity)) = 0;

%% Display Detailed Results
fprintf('\n=== DETAILED PERFORMANCE METRICS ===\n');
fprintf('%-8s %-10s %-10s %-10s %-10s\n', 'Class', 'Precision', 'Recall', 'F1-Score', 'Specificity');
fprintf('%-8s %-10s %-10s %-10s %-10s\n', '-----', '---------', '------', '--------', '-----------');

for i = 1:numClasses
    fprintf('%-8s %-10.4f %-10.4f %-10.4f %-10.4f\n', ...
        string(classes{i}), precision(i), recall(i), f1Score(i), specificity(i));
end

% Calculate macro and weighted averages
macroPrecision = mean(precision);
macroRecall = mean(recall);
macroF1 = mean(f1Score);
macroSpecificity = mean(specificity);

% Class weights for weighted average
classWeights = sum(confMat, 2) / sum(confMat(:));
weightedPrecision = sum(precision .* classWeights);
weightedRecall = sum(recall .* classWeights);
weightedF1 = sum(f1Score .* classWeights);
weightedSpecificity = sum(specificity .* classWeights);

fprintf('\n%-8s %-10.4f %-10.4f %-10.4f %-10.4f\n', 'Macro', macroPrecision, macroRecall, macroF1, macroSpecificity);
fprintf('%-8s %-10.4f %-10.4f %-10.4f %-10.4f\n', 'Weighted', weightedPrecision, weightedRecall, weightedF1, weightedSpecificity);

%% Enhanced Confusion Matrix
figure('Position', [100, 100, 800, 600]);
confusionchart(Yvalidation, Ypred, 'Title', 'Confusion Matrix - ECG Classification', ...
    'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
saveas(gcf, fullfile(ResultsPath, 'confusion_matrix.png'));
saveas(gcf, fullfile(ResultsPath, 'confusion_matrix.fig'));

%% ROC Curves
figure('Position', [200, 100, 1200, 400]);

% Convert labels to numeric for ROC calculation
[~, ~, Yvalidation_numeric] = unique(Yvalidation);
auc_scores = zeros(numClasses, 1);

for i = 1:numClasses
    subplot(1, numClasses, i);
    
    % Create binary labels (one-vs-rest)
    binaryLabels = (Yvalidation_numeric == i);
    classScores = scores(:, i);
    
    % Calculate ROC
    [X, Y, T, AUC] = perfcurve(binaryLabels, classScores, 1);
    auc_scores(i) = AUC;
    
    % Plot ROC curve
    plot(X, Y, 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(sprintf('ROC Curve - %s (AUC = %.4f)', string(classes{i}), AUC));
    grid on;
    
    % Add diagonal line
    hold on;
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 1);
    legend('ROC Curve', 'Random Classifier', 'Location', 'southeast');
    hold off;
end

sgtitle('ROC Curves for All Classes');
saveas(gcf, fullfile(ResultsPath, 'roc_curves.png'));
saveas(gcf, fullfile(ResultsPath, 'roc_curves.fig'));

%% Training Progress Visualization
if isfield(trainInfo, 'TrainingLoss')
    figure('Position', [300, 100, 1200, 500]);
    
    % Training and Validation Loss
    subplot(1, 2, 1);
    plot(trainInfo.TrainingLoss, 'b-', 'LineWidth', 2, 'DisplayName', 'Training Loss');
    hold on;
    if isfield(trainInfo, 'ValidationLoss')
        plot(trainInfo.ValidationLoss, 'r-', 'LineWidth', 2, 'DisplayName', 'Validation Loss');
    end
    xlabel('Iteration');
    ylabel('Loss');
    title('Training and Validation Loss');
    legend('Location', 'best');
    grid on;
    hold off;
    
    % Training and Validation Accuracy
    subplot(1, 2, 2);
    plot(trainInfo.TrainingAccuracy, 'b-', 'LineWidth', 2, 'DisplayName', 'Training Accuracy');
    hold on;
    if isfield(trainInfo, 'ValidationAccuracy')
        plot(trainInfo.ValidationAccuracy, 'r-', 'LineWidth', 2, 'DisplayName', 'Validation Accuracy');
    end
    xlabel('Iteration');
    ylabel('Accuracy');
    title('Training and Validation Accuracy');
    legend('Location', 'best');
    grid on;
    hold off;
    
    sgtitle('Training Progress');
    saveas(gcf, fullfile(ResultsPath, 'training_progress.png'));
    saveas(gcf, fullfile(ResultsPath, 'training_progress.fig'));
end

%% Performance Summary Visualization
figure('Position', [400, 100, 1000, 600]);

% Metrics by class
subplot(2, 2, 1);
x = 1:numClasses;
bar(x, [precision, recall, f1Score], 'grouped');
set(gca, 'XTickLabel', classes);
xlabel('Classes');
ylabel('Score');
title('Performance Metrics by Class');
legend('Precision', 'Recall', 'F1-Score', 'Location', 'best');
grid on;

% AUC scores
subplot(2, 2, 2);
bar(auc_scores);
set(gca, 'XTickLabel', classes);
xlabel('Classes');
ylabel('AUC Score');
title('AUC Scores by Class');
grid on;
ylim([0, 1]);

% Confusion matrix heatmap
subplot(2, 2, 3);
imagesc(confMat);
colorbar;
colormap('Blues');
set(gca, 'XTick', 1:numClasses, 'XTickLabel', classes);
set(gca, 'YTick', 1:numClasses, 'YTickLabel', classes);
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix Heatmap');

% Add text annotations to confusion matrix
for i = 1:numClasses
    for j = 1:numClasses
        text(j, i, num2str(confMat(i,j)), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'FontSize', 12, 'FontWeight', 'bold');
    end
end

% Overall metrics summary
subplot(2, 2, 4);
axis off;
summaryText = {
    sprintf('Overall Accuracy: %.2f%%', accuracy * 100);
    sprintf('Macro Precision: %.4f', macroPrecision);
    sprintf('Macro Recall: %.4f', macroRecall);
    sprintf('Macro F1-Score: %.4f', macroF1);
    sprintf('Weighted Precision: %.4f', weightedPrecision);
    sprintf('Weighted Recall: %.4f', weightedRecall);
    sprintf('Weighted F1-Score: %.4f', weightedF1);
    sprintf('Training Time: %.2f min', trainingTime/60);
};

text(0.1, 0.9, summaryText, 'FontSize', 12, 'VerticalAlignment', 'top', ...
    'Units', 'normalized', 'FontWeight', 'bold');

sgtitle('ECG Classification Performance Summary');
saveas(gcf, fullfile(ResultsPath, 'performance_summary.png'));
saveas(gcf, fullfile(ResultsPath, 'performance_summary.fig'));

%% Save Results to Text File
resultsFile = fullfile(ResultsPath, 'training_results.txt');
fid = fopen(resultsFile, 'w');

fprintf(fid, '=== ECG CLASSIFICATION TRAINING RESULTS ===\n');
fprintf(fid, 'Date: %s\n', datestr(now));
fprintf(fid, 'Dataset: %s\n', DatasetPath);
fprintf(fid, 'Model: AlexNet Transfer Learning\n');
fprintf(fid, 'Training Time: %.2f minutes\n\n', trainingTime/60);

fprintf(fid, 'DATASET INFORMATION:\n');
fprintf(fid, 'Total Images: %d\n', numel(images.Files));
fprintf(fid, 'Training Images: %d\n', numel(TrainImages.Files));
fprintf(fid, 'Validation Images: %d\n', numel(TestImages.Files));
fprintf(fid, 'Classes: %s\n\n', strjoin(string(classes), ', '));

fprintf(fid, 'OVERALL PERFORMANCE:\n');
fprintf(fid, 'Accuracy: %.4f (%.2f%%)\n\n', accuracy, accuracy * 100);

fprintf(fid, 'CLASS-WISE PERFORMANCE:\n');
fprintf(fid, '%-8s %-10s %-10s %-10s %-10s %-10s\n', 'Class', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC');
fprintf(fid, '%-8s %-10s %-10s %-10s %-10s %-10s\n', '-----', '---------', '------', '--------', '-----------', '---');

for i = 1:numClasses
    fprintf(fid, '%-8s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n', ...
        string(classes{i}), precision(i), recall(i), f1Score(i), specificity(i), auc_scores(i));
end

fprintf(fid, '\nAVERAGE METRICS:\n');
fprintf(fid, 'Macro - Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n', macroPrecision, macroRecall, macroF1);
fprintf(fid, 'Weighted - Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n', weightedPrecision, weightedRecall, weightedF1);

fprintf(fid, '\nCONFUSION MATRIX:\n');
fprintf(fid, '%-8s', 'True\\Pred');
for i = 1:numClasses
    fprintf(fid, '%-8s', string(classes{i}));
end
fprintf(fid, '\n');

for i = 1:numClasses
    fprintf(fid, '%-8s', string(classes{i}));
    for j = 1:numClasses
        fprintf(fid, '%-8d', confMat(i,j));
    end
    fprintf(fid, '\n');
end

fclose(fid);

%% Display Final Summary
fprintf('\n=== TRAINING COMPLETED SUCCESSFULLY ===\n');
fprintf('Results saved to: %s\n', ResultsPath);
fprintf('Model file: %s\n', modelSavePath);
fprintf('Results summary: %s\n', resultsFile);
fprintf('\nGenerated files:\n');
fprintf('- trained_alexnet_model.mat (trained model)\n');
fprintf('- confusion_matrix.png/.fig\n');
fprintf('- roc_curves.png/.fig\n');
fprintf('- training_progress.png/.fig\n');
fprintf('- performance_summary.png/.fig\n');
fprintf('- training_results.txt\n');

fprintf('\n=== FINAL PERFORMANCE SUMMARY ===\n');
fprintf('Overall Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Macro F1-Score: %.4f\n', macroF1);
fprintf('Weighted F1-Score: %.4f\n', weightedF1);
fprintf('Training completed in %.2f minutes\n', trainingTime/60);