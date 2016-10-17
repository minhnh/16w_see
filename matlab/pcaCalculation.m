function pcaCalculation(data, direction)

    [coeffs, projected_coords, latent] = pca(data);

    figure(1);
    subplot(1, 2, 1);
    plot(projected_coords(:,1), projected_coords(:,2),'+');
    xlabel('1st Principal Component');
    ylabel('2nd Principal Component');
    title(strcat('Projected Coordinates for Going ', direction));
    subplot(1, 2, 2);
    coeffs(4)
    plot(data(:,1),data(:,2),'g+',[mean(data(:,1)),mean(data(:,1))+coeffs(3)],[mean(mean(data(:,2))),mean(mean(data(:,2)))+coeffs(1)],'r',...
    [mean(data(:,1)),mean(data(:,1))+coeffs(4)],[mean(data(:,2)),mean(data(:,2))+coeffs(2)],'b');
    grid on
    xlabel('x');
    ylabel('y');
    title('PCA vectors')

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