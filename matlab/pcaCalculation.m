function [coeffs, projected_coords] = pcaCalculation(data)
    coeffs = pca(data);

    data_straight_m = data - repmat(mean(data), [length(data), 1]);

    projected_coords = data_straight_m * coeffs;
end


