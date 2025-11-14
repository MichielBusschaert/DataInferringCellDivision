function hyp_init = MomentInitialize(spec_decomp,alpha)
arguments
    spec_decomp;
    alpha = 1e-3;
end
    %% MOMENTINITIALIZE Initialize hyperparameters based on the methods described by Basak et al. "Numerical issues in maximum likelihood parameter estimation for Gaussian process interpolation"
    % Parse input & output data
    x = spec_decomp.x_train;
    y = spec_decomp.y_train;

    % Get lengthscale and variance
    s2 = var(y);
    l2 = var(x);

    % Noise variance as percentage of variance
    sn2 = alpha^2*s2;

    % Log transform
    hyp_init = struct('logs',0.5*log(s2),...
        'logl',0.5*log(l2),...
        'logsn',0.5*log(sn2));
end