function regularization_result = RegularizationRange(spec_decomp,hyperprior,lambda_list,hyp_init,NameValueArgs)
arguments (Input)
    spec_decomp struct;
    hyperprior function_handle;
    lambda_list double;
    hyp_init struct;
    NameValueArgs.ParallelPool = '';
    NameValueArgs.SimFun function_handle = @(post) [];
end
    % Initialize
    GP_res = repmat(struct('Train',[],'Infer',[],'Sim',[]),numel(hyp_init),numel(lambda_list));
    sim_fun = NameValueArgs.SimFun;
    if any(strcmp(NameValueArgs.ParallelPool,parallel.listProfiles))
        % Parallel computation
        delete(gcp('nocreate'))
        ppool = parpool(NameValueArgs.ParallelPool);
        parfor i = 1:numel(hyp_init)
            % Iterate over regularizations
            [GP] = eval_loop(spec_decomp, hyperprior, lambda_list, hyp_init(i), sim_fun);
            GP_res(i,:) = GP;
        end
        delete(ppool);
    else
        % Sequential computation
        for i = 1:numel(hyp_init)
            % Iterate over regularizations
            [GP] = eval_loop(spec_decomp, hyperprior, lambda_list, hyp_init(i), sim_fun);
            GP_res(i,:) = GP;
        end
    end
   
    % Parse output
    regularization_result = struct('Lambda',reshape(lambda_list,1,numel(lambda_list)),...
        'HypInit',reshape(hyp_init,numel(hyp_init),1),...
        'GP',GP_res);
end

function [GP] = eval_loop(spec_decomp, hyperprior_lambda, lambda_list, hyp, sim_fun)
    % Initialize
    GP = repmat(struct('Train',[],'Infer',[],'Sim',[]),1,0);
    for j = 1:numel(lambda_list)
        % Define hyperprior
        hyperprior = @(theta) hyperprior_lambda(theta, lambda_list(j));
        
        % Train hyperparameters
        [hyp_opt,obj_opt,dobj_opt,opt_info] = GP_Gamma.train(hyp,spec_decomp,'MAP',Ntrain=1,DeltaTrain=0,Hyperprior=hyperprior);
        train_result = struct('Hyp',hyp_opt,'ObjVal',obj_opt,'ObjGrad',dobj_opt,'OptInfo',opt_info);
        hyp = hyp_opt;
        
        % Forward simulation
        [likelihood,posterior] = GP_Gamma.infer(hyp,spec_decomp,Hyperprior=hyperprior);
        inference_result = struct('Likelihood',likelihood,'Posterior',posterior);
        if isempty(sim_fun)
            sim_result = [];
        else
            sim_result = sim_fun(posterior);
        end
        
        % Store results
        GP(1,j) = struct('Train',train_result,'Infer',inference_result,'Sim',sim_result);
    end
end

% lambda_list = logspace(-5,5,50)';
% hyp_init = all_result.GP.Train.OptInfo.Hyp;
% regularization_result = struct('Lambda',lambda_list,'HypInit',hyp_init,'GP',repmat(GP_result,0,0));
% for i = 1:numel(hyp_init)
%     % Iterate over regularizations
%     for j = 1:numel(lambda_list)
%         % Define hyperprior
%         hyperprior_reg = @(theta) GP_Gamma.Hyperprior.gauss(theta, mu, sigma./lambda_list(j));
% 
%         % Train hyperparameters
%         [hyp_opt_reg,obj_opt_reg,dobj_opt_reg,opt_info_reg] = GP_Gamma.train(hyp_init(i),spec_decomp,'MAP',Ntrain=1,DeltaTrain=0,Hyperprior=hyperprior_reg);
%         train_result_reg = struct('Hyp',hyp_opt_reg,'ObjVal',obj_opt_reg,'ObjGrad',dobj_opt_reg,'OptInfo',opt_info_reg);
%         hyp_init(i) = hyp_opt_reg;
% 
%         % Forward simulation
%         [likelihood_reg,posterior_reg] = GP_Gamma.infer(hyp_opt_reg,spec_decomp,Hyperprior=hyperprior_reg);
%         inference_result_reg = struct('Likelihood',likelihood_reg,'Posterior',posterior_reg);
%         sim_result_reg = GP_Gamma.simulate(posterior_reg,sim_fun,Nsamples=30,Validation=valid_struct.ndf);
% 
%         % Store results
%         GP_reg = struct('Train',train_result_reg,'Infer',inference_result_reg,'Sim',sim_result_reg);
%         regularization_result.GP(i,j) = GP_reg;
%     end
% end