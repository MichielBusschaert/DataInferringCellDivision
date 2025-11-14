function varargout = CovApxConstructKernel(hyp,spec_fun,lambda,phi1,phi2)
    %Determine whether derivatives need to be known
    if nargout > 1
        %Include derivatives
        [Sn,dSn] = spec_fun(hyp,sqrt(lambda));
        K = phi1*(phi2.*Sn')';
        dK = struct('logs',phi1*(phi2.*dSn.logs')',...
            'logl',phi1*(phi2.*dSn.logl')');

        %Parse output
        varargout = {K,dK};
    else
        %Only covariance matrix
        Sn = spec_fun(hyp,sqrt(lambda));
        K = phi1*(phi2.*Sn')';

        %Parse output
        varargout = {K};
    end
end