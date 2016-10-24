function [ out ] = chiSquareTestGaussian(data, confidence, var_name )
%% Do chi square test for given 1-D data
%Authors P.Lukin,  E. Ovchinnikova

% Estimate parameters of the distribution
m = mean(data);
sigma = sqrt(var(data));
data = sort(data);

% Determine number og histogram bins
b = 2;
n = length(data);
r = round(b*log(n));
interval_size = (max(data)-min(data))/r;


% Count data points that fall into each bin
k1 = min(data);
i = 1;
while i<=r
    k2 = 0;
    for j=1:n
        if (data(j)>=k1) & (data(j)<=k1+interval_size)
            k2 = k2+1;
        end
    end
    frequencies(i) = k2;
    k1 = min(data)+interval_size*i;
    i = i+1;
end

% Calculate theoretica frequencies
for i=1:r    
    p = normcdf([min(data)+(i-1)*interval_size min(data)+i*interval_size],m,sigma);
    P(i) = p(2) - p(1);
end

% Find chi square value
chi = 0;
for i=1:r
    chi = chi+((frequencies(i)-n*P(i)).^2)/(n*P(i));
end

% Test obtained chi square value
chietabl = chi2inv(1-confidence, r-1);
if chietabl>chi
    out = ['Data:',var_name,' is Gaussian distributed with: mean = ',num2str(m),...
        ' and standard deviation sigma = ',num2str(sigma),' with confidence ', num2str(1-confidence),'%.' ];
else 
    out = ['Data: ',var_name,' can not be proven to have Gaussian Distribution with confidence',' ', num2str(1-confidence),'%.' ];
end

end

