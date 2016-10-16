function pcaCalculation(data, direction)
    [coeffs, projected_coords, latent] = pca(data);

    figure();
    subplot(1, 2, 1);
    plot(projected_coords(:,1), projected_coords(:,2),'+');
    xlabel('1st Principal Component');
    ylabel('2nd Principal Component');
    title(strcat('Projected Coordinates for Going ', direction));
    subplot(1, 2, 2);
    biplot(coeffs);
    xlabel('x');
    ylabel('y');
    title('Principal Components in Original Coordinates')

    figure();
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