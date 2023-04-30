% STEP 1: Load the images
image_folder = 'images';
imds = imageDatastore(image_folder);

% Add column headers in the text file before STEP 2.
% STEP 2: Load the groundtruth file
groundtruth_file = 'labels.txt';
gt = readtable(groundtruth_file);

% STEP 3: Convert the images to grayscale and resize them to a common size
imageSize = [227 227];
numImages = length(imds.Files);
images = zeros([imageSize numImages], 'uint8');
for i = 1:numImages
    I = readimage(imds, i);
    I = rgb2gray(I);
    I = imresize(I, imageSize);
    images(:,:,i) = I;
end

% STEP 4: Extract HOG and texture features
hogFeature = @(img) extractHOGFeatures(img, 'CellSize', [8 8]); % Create an anonymous function that takes an input image and extracts its HOG features using a cell size of 8x8.
textureFeature = @(img) extractLBPFeatures(img, 'CellSize', [32 32]); % Create another anonymous function that takes an input image and extracts its LBP features using a cell size of 32x32.
hogFeatureSize = length(hogFeature(images(:,:,1))); % Calculate the length of the HOG feature vector for the first image in the dataset by calling the hogFeature function on that image and taking the length of the resulting feature vector.
textureFeatureSize = length(textureFeature(images(:,:,1))); % Calculate the length of the LBP feature vector for the first image in the dataset by calling the textureFeature function on that image and taking the length of the resulting feature vector.
features = zeros(numImages, hogFeatureSize + textureFeatureSize, 'single'); % This matrix will be used to store the feature vectors for all of the images in the dataset.

% Loop over the imageDatastore to perform feature extraction on every image

for i = 1:numImages
    hogFeat = hogFeature(images(:,:,i));
    textureFeat = textureFeature(images(:,:,i));
    features(i,:) = [hogFeat textureFeat];
end


% STEP 5: Convert the labels to numeric format
labels = gt.Label;
labels = cellfun(@(x)strcmp(x,'malignant'), labels);

% STEP 6: Train the SVM model using 10-fold cross-validation
rng(1); % let's all use the same seed for the random number generator 
svm = fitcsvm(features, labels); 
cvsvm = crossval(svm, 'KFold', 10); 
pred = kfoldPredict(cvsvm); 
[cm, order] = confusionmat(labels, pred); 

% Saving the model
save('SVMModel.mat', 'svm');

% STEP 7: Compute performance metrics

% Compute sensitivity and specificity
TP = cm(2,2);
TN = cm(1,1);
FP = cm(1,2);
FN = cm(2,1);
sensitivity = TP / (TP + FN);
specificity = TN / (TN + FP);
fprintf('Sensitivity = %.2f%%\n', sensitivity*100);
fprintf('Specificity = %.2f%%\n', specificity*100);

% Compute accuracy
accuracy = sum(diag(cm)) / sum(cm(:));
fprintf('Accuracy = %.2f%%\n', accuracy*100);

% Compute precision
precision = cm(2,2) / sum(cm(:,2));
fprintf('Precision = %.2f%%\n', precision*100);

% Compute F1 Score
f1Score = 2 * (precision * sensitivity) / (precision + sensitivity);
fprintf('F1 Score = %.2f%%\n', f1Score*100);