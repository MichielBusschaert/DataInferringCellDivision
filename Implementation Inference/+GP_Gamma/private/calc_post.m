function [mu_post,sigma_post] = calc_post(hyp,spec_decomp,core)
%% CALC_POST Calculates the posterior distribution (mean & covariance)
    % Parse spectral decomposition parameters
    spec_fun = spec_decomp.spec_fun;
    lambda = spec_decomp.lambda;
    phi_train = spec_decomp.phi_train;
    phi_test = spec_decomp.phi_test;

    % Parse core parameters
    L = core.L;
    alpha = core.alpha;
    sW = core.sW;

    % Evaluate covariance and cross-covariance matrix
    K = CovApxConstructKernel(hyp,spec_fun,lambda,phi_test,phi_test);
    Kc = CovApxConstructKernel(hyp,spec_fun,lambda,phi_test,phi_train);
    V = L\(repmat(sW,1,size(phi_test,1)).*Kc');

    % Evaluate posterior mean and covariance;
    mu_post = Kc*alpha;
    sigma_post = K - V'*V;
end