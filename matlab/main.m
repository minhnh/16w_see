load('overleaf/matlab/poseData.mat');
addpath('overleaf/matlab/');

% projected_coords_straight =  pcaCalculation(data_straight, ' Straight');
fitGaussian(data_straight, ' Straight');
fitGaussian(data_straight_angles, ' Straight');

% projected_coords_slightLeft =  pcaCalculation(data_slight_left, ' Slight Left');
fitGaussian(data_slight_left, ' Slight Left');
fitGaussian(data_slight_left_angles, ' Slight Left');

% projected_coords_slightRight =  pcaCalculation(data_slight_right, ' Slight Right');
fitGaussian(data_slight_right, ' Slight Right');
fitGaussian(data_slight_right_angles, ' Slight Right');

% projected_coords_left =  pcaCalculation(data_left, ' Left');
fitGaussian(data_left, ' Left');
fitGaussian(data_left_angles, ' Left');

% projected_coords_right = pcaCalculation(data_right, ' Right');
fitGaussian(data_right, ' Right');
fitGaussian(data_right_angles, ' Right');

% J2 = undistortImage(imread('/home/minh/frame0000_.jpg'),cameraParams,'OutputView','full');
% figure; imshow(J2);
% title('Full Output View');