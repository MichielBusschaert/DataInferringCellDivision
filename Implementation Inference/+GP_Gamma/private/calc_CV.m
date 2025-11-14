function varargout = calc_CV(core,dcore)
%% CALC_CV Calculates the cross-validation likelihood per batch
arguments (Input)
    core struct;
    dcore struct = struct();
end
    %Parse posterior parameters
    L = core.L;
    alpha = core.alpha;
    sW = core.sW;
    batch_idces = core.batch_idces;

    %Batch mean & covariance parameters
    P = repmat(sW,1,numel(alpha)).*(L'\eye(numel(alpha))); %chol(Kinv)
    Q = P*P'; % = Kinv
    Y = P'\(P\alpha);

    %Inverse covariance and derivative
    if nargout > 1
        CV_dnlpl = struct('logs',0,...
            'logl',0,...
            'logsn',0);
        Z = struct('logs',Q*dcore.logs,...
            'logl',Q*dcore.logl,...
            'logsn',Q./repmat(0.5.*sW.^2,1,numel(alpha)));
        ZQ = struct('logs',Z.logs*Q,...
            'logl',Z.logl*Q,...
            'logsn',Z.logsn*Q);
        Zalpha = struct('logs',Z.logs*alpha,...
            'logl',Z.logl*alpha,...
            'logsn',Z.logsn*alpha);
    end

    %Iterate over batches
    CV_nlpl = 0;
    for idx = 1:max(batch_idces)
        %Used parameters
        is_p = (batch_idces == idx);
        Nbatch = nnz(is_p);
        Q_p = Q(is_p,is_p); % Kinv_ii
        alpha_p = alpha(is_p);

        %Cholesky decomposition of submatrix
        P_p = chol(Q_p+1e-9.*norm(Q_p)*eye(Nbatch),'lower');
        %P_p = chol(Q_p,'lower');
        
        %Predictive likelihood
        CV_nlpl_p = 0.5*(alpha_p'*(P_p'\(P_p\alpha_p)) - 2*sum(log(diag(P_p))) + Nbatch*log(2*pi));
        CV_nlpl = CV_nlpl + CV_nlpl_p;
        
        %Calculate derivative
        if nargout >1
            hyp_names = fieldnames(Z);
            R = P_p*P_p' + alpha_p*alpha_p';
            for jdx = 1:numel(hyp_names)
                %Used parameters
                ZQ_p = ZQ.(hyp_names{jdx})(is_p,is_p);
                Zalpha_p = Zalpha.(hyp_names{jdx})(is_p);

                %Derivative of predictive likelihood
                S = P_p'\(P_p\(ZQ_p*(P_p'\(P_p\eye(Nbatch)))));
                CV_dnlpl_p = 0.5*(sum(R.*S,'all')-2*alpha_p'*(P_p'\(P_p\Zalpha_p)));
                CV_dnlpl.(hyp_names{jdx}) = CV_dnlpl.(hyp_names{jdx}) + CV_dnlpl_p;
            end
        end
    end

    %Parse output
    if nargout < 2
        varargout = {CV_nlpl};
    else
        varargout = {CV_nlpl,CV_dnlpl};
    end
end