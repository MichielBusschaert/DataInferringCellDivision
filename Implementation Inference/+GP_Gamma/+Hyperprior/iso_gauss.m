function varargout = iso_gauss(theta, mu, sigma)
% mu & sigma must be in order of mu = [logs, logl, logsn]
    % Test if function is defined
    if isempty(theta)
        varargout = {1};
        return;
    end

    % Identical to multivariate Gaussian
    if nargout > 1
        [nlhl, dnlhl] = GP_Gamma.Hyperprior.gauss(theta, mu, diag(sigma));
        varargout = {nlhl, dnlhl};
    else
        nlhl = GP_Gamma.Hyperprior.gauss(theta, mu, diag(sigma));
        varargout = {nlhl};
    end
end