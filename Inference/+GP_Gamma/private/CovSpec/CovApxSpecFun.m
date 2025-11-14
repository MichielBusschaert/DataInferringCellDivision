function varargout = CovApxSpecFun(hyp,w)
    %Parse spectral output
    S = SpectralDensity(hyp,w);

    %Parse outputs
    if nargout > 1
        dS = struct('logs',SpectralDensity_dlogs(hyp,w),...
            'logl',SpectralDensity_dlogl(hyp,w));
        varargout = {S,dS};
    else
        varargout = {S};
    end
end

%% Spectral density functions
function S = SpectralDensity(hyp,w)
    %Spectral density by applying Bochner's theorem to the squared
    %exponential kernel
    %
    % k(r) = s^2*exp(-0.5*r^2/l^2)
    %
    % S(w) = s^2*sqrt(2*pi*l^2)*exp(-0.5*l^2*w^2)

    %Parse hyperparameters
    s2 = exp(2*hyp.logs);
    l2 = exp(2*hyp.logl);

    %Calculate spectral density
    S = s2*sqrt(2*pi*l2)*exp(-0.5*l2*w.^2);
end

function dS = SpectralDensity_dlogs(hyp,w)
    %Derivative of the spectral density approximation with respect to logs

    %Parse hyperparameters
    s2 = exp(2*hyp.logs);
    l2 = exp(2*hyp.logl);

    %Calculate spectral density
    dS = 2*s2*sqrt(2*pi*l2).*exp(-0.5*l2*w.^2);
end

function dS = SpectralDensity_dlogl(hyp,w)
    %Derivative of the spectral density approximation with respect to logs

    %Parse hyperparameters
    s2 = exp(2*hyp.logs);
    l2 = exp(2*hyp.logl);

    %Calculate spectral density
    dS = (s2*sqrt(2*pi*l2)).*((1 - l2*w.^2).*exp(-0.5*l2*w.^2));
end