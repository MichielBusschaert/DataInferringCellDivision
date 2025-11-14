function kernel_result = MultipleBeta(x_test,ndf_test,train_struct,valid_struct,beta_fun_list,sim_fun,div_beta_fun,NameValueArgs)
arguments (Input)
    x_test;
    ndf_test;
    train_struct;
    valid_struct;
    beta_fun_list cell;
    sim_fun function_handle;
    div_beta_fun function_handle;
    NameValueArgs.L = 2*max(x_test);
    NameValueArgs.m = 500;
    NameValueArgs.ParallelPool char = '';
    NameValueArgs.Lambda double = 0;
    NameValueArgs.Hyp0 struct = struct();
    NameValueArgs.Parameters struct = struct();
    NameValueArgs.Description cell = {};
    NameValueArgs.SingleHyperprior = @(theta) [];
    NameValueArgs.EMDVariableJacobian = @(x) ones(size(x));
end

    % Initialize
    GP_blank = struct('Train',[],'Infer',[],'Sim',[]);
    Hyperprior_blank = struct('Mean',[],'Cov',[],'Hyp',[]);
    Hyperprior_result = repmat(Hyperprior_blank,1,numel(beta_fun_list));
    GP_result = repmat(GP_blank,numel(NameValueArgs.Lambda),numel(beta_fun_list));
    GP_result_indiv = repmat(GP_blank,numel(train_struct.ndf),numel(beta_fun_list));
    
    L = NameValueArgs.L;
    m = NameValueArgs.m;
    hyp0 = NameValueArgs.Hyp0;

    % Setup parallel pool
    if any(strcmp(parallel.listProfiles,NameValueArgs.ParallelPool))
        delete(gcp('nocreate'));
        disp("Setting up parallel pool...");
        ppool = parpool(NameValueArgs.ParallelPool);
    end

    % Calculate division matrices
    div_beta_mat_cells = cell(numel(beta_fun_list),1);
    if canUseParallelPool && any(strcmp(parallel.listProfiles,NameValueArgs.ParallelPool))
        parfor beta_idx = 1:numel(beta_fun_list)
            disp("Evaluating division matrix "+num2str(beta_idx)+" out of "+num2str(numel(beta_fun_list))+".");
            div_beta_mat_cells{beta_idx,1} = div_beta_fun(beta_fun_list{beta_idx});
        end
    else
        for beta_idx = 1:numel(beta_fun_list)
            disp("Evaluating division matrix "+num2str(beta_idx)+" out of "+num2str(numel(beta_fun_list))+".");
            div_beta_mat_cells{beta_idx,1} = div_beta_fun(beta_fun_list{beta_idx});
        end
    end

    %% Iterate over division kernels - Train individual
    if canUseParallelPool && any(strcmp(parallel.listProfiles,NameValueArgs.ParallelPool))
        % Parallel computing
        parfor beta_idx = 1:numel(beta_fun_list)
            sfunc = @(g) sim_fun(g,div_beta_mat_cells{beta_idx});
            [GP_result_indiv(:,beta_idx), Hyperprior_result(1,beta_idx)] = eval_loop1(x_test,ndf_test,train_struct,valid_struct,beta_fun_list{beta_idx},sfunc,L,m,hyp0,NameValueArgs.SingleHyperprior,NameValueArgs.EMDVariableJacobian);
        end
    else
        % Sequential computing
        for beta_idx = 1:numel(beta_fun_list)
            sfunc = @(g) sim_fun(g,div_beta_mat_cells{beta_idx});
            [GP_result_indiv(:,beta_idx), Hyperprior_result(1,beta_idx)] = eval_loop1(x_test,ndf_test,train_struct,valid_struct,beta_fun_list{beta_idx},sfunc,L,m,hyp0,NameValueArgs.SingleHyperprior,NameValueArgs.EMDVariableJacobian);
        end
    end

    %% Iterate over division kernels - Train all
    for lambda_idx = 1:numel(NameValueArgs.Lambda)
        % Initialize
        GP_row = repmat(GP_blank,1,numel(beta_fun_list));
        lambda = NameValueArgs.Lambda(lambda_idx);
        % Select computing mode
        if canUseParallelPool && any(strcmp(parallel.listProfiles,NameValueArgs.ParallelPool))
            % Parallel computing
            parfor beta_idx = 1:numel(beta_fun_list)
                sfunc = @(g) sim_fun(g,div_beta_mat_cells{beta_idx});
                GP_row(1,beta_idx) = eval_loop2(x_test,ndf_test,train_struct,valid_struct,beta_fun_list{beta_idx},sfunc,L,m,hyp0,lambda,Hyperprior_result(1,beta_idx),NameValueArgs.EMDVariableJacobian);
            end
        else
            % Sequential computing
            for beta_idx = 1:numel(beta_fun_list)
                sfunc = @(g) sim_fun(g,div_beta_mat_cells{beta_idx});
                GP_row(1,beta_idx) = eval_loop2(x_test,ndf_test,train_struct,valid_struct,beta_fun_list{beta_idx},sfunc,L,m,hyp0,lambda,Hyperprior_result(1,beta_idx),NameValueArgs.EMDVariableJacobian);
            end
        end
        % Store result
        GP_result(lambda_idx,:) = GP_row;
    end

    % Stop parallel pool
    if canUseParallelPool && any(strcmp(parallel.listProfiles,NameValueArgs.ParallelPool))
        delete(ppool);
    end

    % Store results
    kernel_result = struct('Beta',{beta_fun_list},...
        'Lambda',NameValueArgs.Lambda,...
        'GP',GP_result,...
        'GPIndiv',GP_result_indiv,...
        'Hyperprior',Hyperprior_result, ...
        'DivBetaMat',{div_beta_mat_cells});
    if ~isempty(NameValueArgs.Parameters)
        kernel_result.Parameters = NameValueArgs.Parameters;
    end
    if ~isempty(NameValueArgs.Description)
        kernel_result.Description = {NameValueArgs.Description};
    end
end

%% For-loop statement 1
function [GP_result,Hyperprior_result] = eval_loop1(x_test,ndf_test,train_struct,valid_struct,beta_fun,sim_fun,L,m,hyp0,hyperprior,emd_jac)
    % Initialize
    x_train = train_struct.x;
    y_train = train_struct.y;
    ndf_train = train_struct.ndf;
    op_mode_train = train_struct.op_mode;
    ndf_valid = valid_struct.ndf;
    for i = 1:numel(ndf_train)
        ndf_train{i}.beta_fun = beta_fun;
    end
    for i = 1:numel(ndf_valid)
        ndf_valid{i}.beta_fun = beta_fun;
    end

    % Initialize
    GP_blank = struct('Train',[],'Infer',[],'Sim',[]);
    GP_result = repmat(GP_blank,numel(ndf_valid),1);

    % Step 1: Train on individual NDFs
    for i = 1:numel(x_train)
        % Calculate spectral decomposition
        spec_decomp = GP_Gamma.spectral([ndf_train(i);{[]}],ndf_test,[x_train(i);{0}],[y_train(i);{0}],{x_test},[op_mode_train(i);'I'],'I',L,m);

        % Initialize hyperparameters
        if isempty(fieldnames(hyp0))
            hyp0 = GP_Gamma.Hyperparameter.MomentInitialize(spec_decomp);
        end

        % Train hyperparameters
        [hyp_opt,obj_opt,dobj_opt,opt_info] = GP_Gamma.train(hyp0,spec_decomp,'MAP',Ntrain=5e1,DeltaTrain=5,Hyperprior=hyperprior);
        train_result = struct('Hyp',hyp_opt,'ObjVal',obj_opt,'ObjGrad',dobj_opt,'OptInfo',opt_info);

        % Forward simulation
        [likelihood,posterior] = GP_Gamma.infer(hyp_opt,spec_decomp);
        inference_result = struct('Likelihood',likelihood,'Posterior',posterior);
        sim_result = GP_Gamma.simulate(posterior,sim_fun,Nsamples=30,Validation=valid_struct.ndf,VariableJacobian=emd_jac);

        % Store results
        GP_result(i,1) = struct('Train',train_result,...
            'Infer',inference_result,...
            'Sim',sim_result);
    end

    % Calculate hyperprior parameters
    hyp_list = [arrayfun(@(i) GP_result(i,1).Train.Hyp.logs, 1:numel(GP_result));
        arrayfun(@(i) GP_result(i,1).Train.Hyp.logl, 1:numel(GP_result));
        arrayfun(@(i) GP_result(i,1).Train.Hyp.logsn, 1:numel(GP_result))];
    mu = sum(hyp_list,2)/size(hyp_list,2);
    sigma = ((hyp_list-mu)*(hyp_list-mu)')/(size(hyp_list,2)-1);
    Hyperprior_result = struct('Mean',mu,...
        'Cov',sigma, ...
        'Hyp',hyp_list);
end

%% For-loop statement 2
function GP_result = eval_loop2(x_test,ndf_test,train_struct,valid_struct,beta_fun,sim_fun,L,m,hyp0,lambda,hyperprior_struct,emd_jac)
    % Initialize
    x_train = train_struct.x;
    y_train = train_struct.y;
    ndf_train = train_struct.ndf;
    op_mode_train = train_struct.op_mode;
    ndf_valid = valid_struct.ndf;
    for i = 1:numel(ndf_train)
        ndf_train{i}.beta_fun = beta_fun;
    end
    for i = 1:numel(ndf_valid)
        ndf_valid{i}.beta_fun = beta_fun;
    end

    % Define hyperprior
    mu = hyperprior_struct.Mean;
    sigma = hyperprior_struct.Cov;
    if lambda == 0
        hyperprior = @(theta) [];
    else
        hyperprior = @(theta) GP_Gamma.Hyperprior.gauss(theta, mu, sigma./lambda);
    end

    % Calculate spectral decomposition
    spec_decomp = GP_Gamma.spectral([ndf_train;{[]}],ndf_test,[x_train;{0}],[y_train;{0}],{x_test},[op_mode_train;'I'],'I',L,m);
    if isempty(fieldnames(hyp0))
        hyp0 = GP_Gamma.Hyperparameter.WrapVec2Struct(mu);
    end

    % Train hyperparameters
    [hyp_opt,obj_opt,dobj_opt,opt_info] = GP_Gamma.train(hyp0,spec_decomp,'MAP',Ntrain=5e1,DeltaTrain=5,Hyperprior=hyperprior);
    train_result = struct('Hyp',hyp_opt,'ObjVal',obj_opt,'ObjGrad',dobj_opt,'OptInfo',opt_info);
    
    % Forward simulation
    [likelihood,posterior] = GP_Gamma.infer(hyp_opt,spec_decomp,Hyperprior=hyperprior);
    inference_result = struct('Likelihood',likelihood,'Posterior',posterior);
    sim_result = GP_Gamma.simulate(posterior,sim_fun,Nsamples=30,Validation=valid_struct.ndf,VariableJacobian=emd_jac);

    % Store result
    GP_result = struct('Train',train_result,...
        'Infer',inference_result,...
        'Sim',sim_result);
end