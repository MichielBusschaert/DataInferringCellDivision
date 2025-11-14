function [simulate_result] = simulate(posterior,sim_fun,NameValueArgs)
arguments (Input)
    posterior struct;
    sim_fun function_handle;
    NameValueArgs.Validation = {};
    NameValueArgs.Nsamples int64 = 0;
    NameValueArgs.VariableJacobian = @(x) ones(size(x));
end
    try
        % Evaluate Gaussian process mean
        gamma_x = posterior.x;
        gamma_mu = posterior.mean;
        gamma_sigma = posterior.cov;
        gamma_chol = chol(gamma_sigma+1e-6*norm(gamma_sigma)*eye(size(gamma_sigma)),'lower');
    
        %Construct mean and evaluate
        %gamma_fun = griddedInterpolant([gamma_x;gamma_x(end)+1e-9],[max(gamma_mu,0);0],'linear','nearest');
        gamma_fun = griddedInterpolant(gamma_x,max(gamma_mu,0),'linear','nearest');
        [x,ndf_mean] = sim_fun(gamma_fun);
        gamma_mean = gamma_fun(x);

        % Simulate over grid
    
        % Evaluate EMD if available
        if ~isempty(NameValueArgs.Validation)
            EMD_mean_list = zeros(1,numel(NameValueArgs.Validation));
            for jdx = 1:numel(NameValueArgs.Validation)
                EMD_mean_list(1,jdx) = calc_emd(x,ndf_mean,NameValueArgs.Validation{jdx}.x,NameValueArgs.Validation{jdx}.ndf,NameValueArgs.VariableJacobian);
            end
        end
    
        % Store results
        simulate_result = struct('x',x,'gamma_mean',gamma_mean,'ndf_mean',ndf_mean,'EMD_mean',EMD_mean_list);
        
        % Calculate samples
        if NameValueArgs.Nsamples > 0
            % Initialize EMD if needed
            if ~isempty(NameValueArgs.Validation)
                EMD_sample_list = zeros(numel(NameValueArgs.Validation),NameValueArgs.Nsamples);
            end
    
            gamma_sample = zeros(numel(x),NameValueArgs.Nsamples);
            ndf_sample = zeros(numel(x),NameValueArgs.Nsamples);
            for idx = 1:NameValueArgs.Nsamples
                % Draw sample
                rand_vec = randn(numel(gamma_x),1);
                gamma_rand = gamma_mu + gamma_chol*rand_vec;
                gamma_fun = griddedInterpolant(gamma_x,max(gamma_rand,0),'linear','nearest');
                [~,ndf_s] = sim_fun(gamma_fun);
                gamma_s = gamma_fun(x);
                ndf_sample(:,idx) = ndf_s;
                gamma_sample(:,idx) = gamma_s;
    
                % Calculate EMD
                if ~isempty(NameValueArgs.Validation)
                    for jdx = 1:numel(NameValueArgs.Validation)
                        EMD_sample_list(jdx,idx) = calc_emd(x,ndf_s,NameValueArgs.Validation{jdx}.x,NameValueArgs.Validation{jdx}.ndf,NameValueArgs.VariableJacobian);
                    end
                end
            end
    
            % Store results
            simulate_result.gamma_sample = gamma_sample;
            simulate_result.ndf_sample = ndf_sample;
            simulate_result.EMD_sample = EMD_sample_list;
        end
    catch
        %% INCLUDE CATCH STATE%ENT!!!!!!!!!
        x = posterior.x;

        simulate_result = struct('x',x,...
            'gamma_mean',NaN(numel(x),1),...
            'ndf_mean',NaN(numel(x),1),...
            'EMD_mean',NaN(1,numel(NameValueArgs.Validation)));
        if NameValueArgs.Nsamples > 0
            simulate_result.gamma_sample = NaN(numel(x),NameValueArgs.Nsamples);
            simulate_result.ndf_sample = NaN(numel(x),NameValueArgs.Nsamples);
            simulate_result.EMD_sample = NaN(numel(NameValueArgs.Validation),NameValueArgs.Nsamples);
        end
    end
end