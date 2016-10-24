function fitGaussian(data, direction)
%% Processes experiment
%Does chi square test for 2 principal components and angle
%Authors P.Lukin,  E. Ovchinnikova
    [~,dim] = size(data);
    
    if dim==1
       normPhi = rad2deg(data) - mean(rad2deg(data));
       chiSquareTestGaussian(normPhi,0.1,[' angular error during  ',direction,' movement,' ])
       figure(3)
       hist(normPhi);
       xlabel('Angular error, deg');
       ylabel('Frequency');
       title(strcat('Deviation from mean angle. Histogram for Going ', direction));       
    end
    
    if dim ==2
    [coeffs, projected_coords, latent] = pca(data);
    chiSquareTestGaussian(projected_coords(:,1),0.1,[' projection of 1st principal component during ',direction,' movement,' ])
    chiSquareTestGaussian(projected_coords(:,2),0.1,[' projection of 2st principal component during ',direction,' movement,'])

    figure(1);
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
        [mean_x, mean_x + coeffs(1)*sqrt(latent(1))], [mean_y, mean_y + coeffs(3)*sqrt(latent(1))], 'r',...
        [mean_x, mean_x + coeffs(2)*sqrt(latent(2))], [mean_y, mean_y + coeffs(4)*sqrt(latent(2))], 'b');
    axis equal
    grid on
    xlabel('x');
    ylabel('y');
    title('PCA vectors');

    figure(2);
    subplot(2, 1, 1);
    hist(projected_coords(:,1));
    xlabel('Projected coordinates');
    ylabel('Frequency');
    title(strcat('1st Principal Comp. Histogram for Going ', direction));
    subplot(2, 1, 2);
    hist(projected_coords(:,2));
    xlabel('Projected coordinates');
    ylabel('Frequency');
    title(strcat('2nd Principal Comp. Histogram for Going ', direction));
        
        
    end
    
   
end