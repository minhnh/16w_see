function fitGaussian(data, direction)
%% Processes experiment
%Does chi square test for 2 principal components and angle
%Authors P.Lukin,  E. Ovchinnikova
    [~,dim] = size(data);
    
    if dim==1
       normPhi = rad2deg(data) - mean(rad2deg(data));
       figure(3)
       h3 = histogram(normPhi);
       chiSquareTestGaussian(normPhi, 0.1,...
           [' angular error during  ',direction,' movement,' ], h3.NumBins)
       xlabel('Angular error, deg');
       ylabel('Frequency');
       title(strcat('Deviation from mean angle. Histogram for Going ', direction));       
    end
    
    if dim ==2
    [coeffs, projected_coords, latent] = pca(data);

    figure(1);
    subplot(2, 1, 1);
    h1 = histogram(projected_coords(:,1));
    xlabel('Projected coordinates, mm');
    ylabel('Frequency');
    title(strcat('1st Principal Comp. Histogram for Going ', direction));
    subplot(2, 1, 2);
    h2 = histogram(projected_coords(:,2));
    xlabel('Projected coordinates, mm');
    ylabel('Frequency');
    title(strcat('2nd Principal Comp. Histogram for Going ', direction));
    
    chiSquareTestGaussian(projected_coords(:,1),0.1,...
        [' projection of 1st principal component during ',direction,' movement,' ],...
        h1.NumBins)
    chiSquareTestGaussian(projected_coords(:,2),0.1,...
        [' projection of 2nd principal component during ',direction,' movement,'],...
        h2.NumBins)

    figure(2);
    subplot(1, 2, 1);
    plot(projected_coords(:,2), projected_coords(:,1),'+');
    grid on
    axis equal
    xlabel('1st Principal Component, mm');
    ylabel('2nd Principal Component, mm');
    title(strcat('Projected Coordinates for Going ', direction));
    subplot(1, 2, 2);
    mean_x = mean(data(:,1));
    mean_y = mean(data(:,2));
    plot(data(:,1), data(:,2), 'g+',...
        [mean_x, mean_x + coeffs(1)*sqrt(latent(1))], [mean_y, mean_y - coeffs(3)*sqrt(latent(1))], 'r',...
        [mean_x, mean_x + coeffs(2)*sqrt(latent(2))], [mean_y, mean_y - coeffs(4)*sqrt(latent(2))], 'b');
    axis equal
    grid on
    xlabel('x (mm)');
    ylabel('y (mm)');
    title('PCA vectors');

        
    end
    
   
end