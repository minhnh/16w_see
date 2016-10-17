load('overleaf/matlab/poseData.mat');
addpath('overleaf/matlab/');

% pcaCalculation(data_straight, ' Straight');
% pcaCalculation(data_slight_left, ' Slight Left');
% pcaCalculation(data_slight_right, ' Slight Right');
% pcaCalculation(data_left, ' Left');
% pcaCalculation(data_right, ' Right');

J2 = undistortImage(imread('/home/minh/frame0000_.jpg'),cameraParams,'OutputView','full');
figure; imshow(J2);
title('Full Output View');