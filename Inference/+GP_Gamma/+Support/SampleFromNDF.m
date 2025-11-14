function xs = SampleFromNDF(x,ndf,Ns)
    % Calculate normalized CDF
    [~,cdf] = GP_Gamma.Support.NormalizedProbabilityMetrics(ndf,x);

    % Rescale between [0,1] (should already be the case)
    cdf = (cdf-cdf(1))./(cdf(end)-cdf(1));

    % Remove duplicates
    [cdf,i_unique] = unique(cdf,'stable');
    x = x(i_unique);

    % Generate uniform random samples
    u = rand(Ns,1);

    % Invert CDF
    xs = interp1(cdf,x,u,'linear','extrap');
end