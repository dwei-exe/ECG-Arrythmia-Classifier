%Training and validation using Alexnet
DatasetPath = 'C:\Users\henry\Downloads\ECG-Dx\Lead2_Scalogram_Dataset';

%Reading Images from Image Database Folder
images = imageDatastore(DatasetPath, "IncludeSubfolders",true,"LabelSource","foldernames");

% Split the data into training and validation sets
numTrainFiles = 779;
[TrainImages, TestImages] = splitEachLabel(images, numTrainFiles, 'randomized');

net = alexnet; %importing pretrained Alexnet
layersTransfer = net.Layers(1:end-3); %Preserves all layers except the last three layers (full connected layer, softmax layer, classification layer)

numClasses = 4; %Number of output classes: GVST, SR, SB, AFIB

%Defining layers of Alexnet
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, "WeightLearnRateFactor", 20, "BiasLearnRateFactor",20)
    softmaxLayer
    classificationLayer];

%Training Options
options = trainingOptions("sgdm","MiniBatchSize",20,"MaxEpochs", 8,"InitialLearnRate",1e-4,"Shuffle","every-epoch", "ValidationData",TestImages,"ValidationFrequency",10, "Verbose",false,"Plots","training-progress");

%Training AlexNet
netTransfer = trainNetwork(TrainImages, layers, options);

%Classifying Images
Ypred = classify(netTransfer,TestImages);
Yvalidation = TestImages.Labels;
accuracy = sum(Ypred==Yvalidation)/numel(Yvalidation);

%Plotting Confusion Matrix
plotconfusion(Yvalidation,Ypred);