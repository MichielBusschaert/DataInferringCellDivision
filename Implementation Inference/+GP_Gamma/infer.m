function [likelihood,posterior,core] = infer(hyp,spec_decomp,NameValueArgs)
%% INFER Infers the log marginal likelihood and cross validation likelihood of a Gaussian process regression
arguments (Input)
    hyp struct;
    spec_decomp struct;
    NameValueArgs.CalculateDerivative logical = false;
    NameValueArgs.Hyperprior function_handle = @(theta) [];
end

    % Check whether to calculate derivatives
    if NameValueArgs.CalculateDerivative
        % Core parameters
        [core,dcore] = calc_core(hyp,spec_decomp,NameValueArgs.Hyperprior);
        
        % Log marginal likelihood and CV-likelihoods
        [nlml,dnlml] = calc_nlml(core,dcore);
        [CV,dCV] = calc_CV(core,dcore);
        [CVLOO,dCVLOO] = calc_CVLOO(core,dcore);
        if isempty(NameValueArgs.Hyperprior())
            nlhl = NaN;
            dnlhl = struct('logs',NaN,'logl',NaN,'logsn',NaN);
        else
            [nlhl,dnlhl] = NameValueArgs.Hyperprior(hyp);
        end
        likelihood = struct('NLML',nlml,'NLHL',nlhl,'CV',CV,'CVLOO',CVLOO,'dNLML',dnlml,'dNLHL',dnlhl,'dCV',dCV,'dCVLOO',dCVLOO);
    else
        % Core parameters
        core = calc_core(hyp,spec_decomp);
        
        % Log marginal likelihood and CV-likelihoods
        nlml = calc_nlml(core);
        CV = calc_CV(core);
        CVLOO = calc_CVLOO(core);
        if isempty(NameValueArgs.Hyperprior([]))
            nlhl = NaN;
        else
            [nlhl] = NameValueArgs.Hyperprior(hyp);
        end
        likelihood = struct('NLML',nlml,'NLHL',nlhl,'CV',CV,'CVLOO',CVLOO);
    end

    % Mean and covariance of posterior
    [mean_post,cov_post] = calc_post(hyp,spec_decomp,core);
    posterior = struct('x',spec_decomp.x_test,'mean',mean_post,'cov',cov_post);
end