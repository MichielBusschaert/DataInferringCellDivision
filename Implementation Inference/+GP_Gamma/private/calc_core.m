function varargout = calc_core(hyp,spec_decomp,hyperprior)
arguments
    hyp struct;
    spec_decomp struct;
    hyperprior function_handle = @(theta) [];
end
%% CALC_CORE Calculates the core parameters for Gaussian process regression
    %Determine core parameters: L, alpha & sW
    %Evaluate covariance matrix based on usage mode

    % Parse spectral decomposition
    spec_fun = spec_decomp.spec_fun;
    lambda = spec_decomp.lambda;
    phi = spec_decomp.phi_train;
    Y = spec_decomp.y_train;
    batch_idces = spec_decomp.batch_train;

    %Decide whether to pass hyperparameter derivative info
    if nargout > 1
        %Parse covariance and its derivative
        [K,dK] = CovApxConstructKernel(hyp,spec_fun,lambda,phi,phi);
    else
        %Parse covariance only
        K = CovApxConstructKernel(hyp,spec_fun,lambda,phi,phi);
    end

    %Dimensionality
    N = size(Y,1);

    %Square root of covariance
    sW = repmat(exp(-hyp.logsn),N,1);

    %Evaluate Cholesky decomposition
    L = chol(repmat(sW,1,N).*K.*repmat(sW',N,1)+eye(N),'lower');

    %Evaluate projected output
    alpha = sW.*(L'\(L\(sW.*Y)));

    %Parse output
    core = struct('L',L,'alpha',alpha,'sW',sW,'batch_idces',batch_idces);
    if nargout > 1
        dcore = dK;
        if ~isempty(hyperprior([]))
            %Pass hyperprior
            [nhpl,dnhpl] = hyperprior(hyp);
            core.hyperprior_lik = nhpl;
            dcore.hyperprior_dlik = dnhpl;
        end

        %Parse output
        varargout = {core,dcore};
    else
        if ~isempty(hyperprior([]))
            %Pass hyperprior
            nhpl = hyperprior(hyp);
            core.hyperprior_lik = nhpl;
        end

        %Parse output
        varargout = {core};
    end
end