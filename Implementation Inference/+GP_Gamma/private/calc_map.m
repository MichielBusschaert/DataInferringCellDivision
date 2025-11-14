function [varargout] = calc_map(core,dcore)
%% CALC_MAP Calculates the negative maximum a posteriori objective, which is the sum of the negative log marginal likelihood and the negative log hyperprior likelihood
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
        %P = repmat(sW,1,numel(alpha)).*(L'\eye(numel(alpha)));
        P = sW.*(L'\eye(numel(alpha)));
        Q = P*P' - alpha.*alpha';

        %Determine derivative of NLML
        dnlml = struct();
        field_names = fieldnames(dcore);
        for idx = 1:numel(field_names)
            if ~strcmp(field_names{idx},'hyperprior_dlik')
                dnlml.(field_names{idx}) = 0.5*sum(Q.*dcore.(field_names{idx})','all');
                if isfield(dcore,'hyperprior_dlik')
                    dnlml.(field_names{idx}) = dnlml.(field_names{idx}) + dcore.hyperprior_dlik.(field_names{idx});
                end
            end
        end
        dnlml.logsn = sum((diag(Q)./sW)./sW);
        if isfield(dcore,'hyperprior_dlik')
            dnlml.logsn = dnlml.logsn + dcore.hyperprior_dlik.logsn;
        end

        %Parse output
        varargout = {nlml,dnlml};
    else
        %Parse output
        varargout = {nlml};
    end
end