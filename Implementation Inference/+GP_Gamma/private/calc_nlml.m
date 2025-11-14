function [varargout] = calc_nlml(core,dcore)
%% CALC_NLML Calculates the negative log marginal likelihood
arguments (Input)
    core struct;
    dcore struct = struct();
end
    %Parse posterior parameters
    L = core.L;
    alpha = core.alpha;
    sW = core.sW;

    %Evaluate negative log marginal likelihood
    nlml = 0.5*(sum((L'*(alpha./sW)).^2) + 2*sum(log(diag(L))) - 2*sum(log(sW)) + numel(alpha)*log(2*pi));

    %Parse hyperprior information
    if isfield(core,'hyperprior_lik')
        nlml = nlml + core.hyperprior_lik;
    end

    %Evaluate derivative of the log marginal likelihood
    if nargin > 1
        %Determine derivative submatrix
        P = repmat(sW,1,numel(alpha)).*(L'\eye(numel(alpha)));
        Q = P*P' - alpha.*alpha';

        %Determine derivative of NLML
        dnlml = struct();
        field_names = fieldnames(dcore);
        for idx = 1:numel(field_names)
            if ~strcmp(field_names{idx},'hyperprior_dlik')
                dnlml.(field_names{idx}) = 0.5*sum(Q.*dcore.(field_names{idx})','all');
            end
        end
        dnlml.logsn = sum((diag(Q)./sW)./sW);

        %Parse output
        varargout = {nlml,dnlml};
    else
        %Parse output
        varargout = {nlml};
    end
end