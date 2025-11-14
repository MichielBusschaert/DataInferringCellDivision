function varargout = gauss(theta, mu, sigma)
% mu & sigma must be in order of mu = [logs, logl, logsn]
    % Test if function is defined
    if isempty(theta)
        varargout = {1};
        return;
    end

    % Parse hyperparameters
    theta_vec = [theta.logs;theta.logl;theta.logsn];
 
    % Calculate log likelihood
    L = chol(sigma,'lower');
    nlhl = 0.5*((theta_vec-mu)'*(L'\(L\(theta_vec-mu))) + 2*sum(log(diag(L))) + numel(theta_vec)*log(2*pi));

    % If needed, calculate derivative
    if nargout > 1
        % Calculate derivative
        dnlhl_vec = L'\(L\(theta_vec-mu));
        % Parse as stuct
        dnlhl = struct('logs',dnlhl_vec(1),'logl',dnlhl_vec(2),'logsn',dnlhl_vec(3));
        varargout = {nlhl, dnlhl};
    else
        % Parse output
        varargout = {nlhl};
    end
end