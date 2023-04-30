% Load the saved SVM model
load('SVMModel.mat');

% Prompt the user to select an image
[filename, path] = uigetfile({'*.jpg;*.png;*.bmp', 'Image files (*.jpg, *.png, *.bmp)'}, 'Select an image file');
if isequal(filename, 0) % if the user cancels the file selection
    disp('No image selected. Please run again.');
    return;
end

% Read in the selected image to be classified
img = imread(fullfile(path, filename));

% Converting the image to grayscale and resize to a common size
img = rgb2gray(img);
img = imresize(img, [227 227]);

% Extract HOG and texture features from the image
hogFeature = extractHOGFeatures(img, 'CellSize', [8 8]); 
textureFeature = extractLBPFeatures(img, 'CellSize', [32 32]);
features = [hogFeature textureFeature];

% Make a prediction using the SVM model
prediction = predict(svm, features);

% Display the prediction
if prediction
    disp('The image is malignant.');
else
    disp('The image is benign.');
end
