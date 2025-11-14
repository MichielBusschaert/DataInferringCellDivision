function [hyp_opt,obj_opt,dobj_opt,opt_info] = train(hyp_init,spec_decomp,obj,NameValueArgs)
arguments
    hyp_init struct;
    spec_decomp struct;
    obj char {mustBeMember(obj,{'MAP','NLML','CV','CVLOO'})} = 'MAP';
    NameValueArgs.Ntrain int64 = 1;
    NameValueArgs.DeltaTrain double = 0;
    NameValueArgs.Hyperprior function_handle = @(theta) [];
    NameValueArgs.MinimizeLength double = -5000;
    NameValueArgs.ShowLog logical = true;
end
    % Parse output
    Ntrain = NameValueArgs.Ntrain;
    DeltaTrain = NameValueArgs.DeltaTrain;

    %Define function wrappers
    fwrap_s2v = @(str) GP_Gamma.Hyperparameter.WrapStruct2Vec(str);
    fwrap_v2s = @(vec) GP_Gamma.Hyperparameter.WrapVec2Struct(vec);

    % Define objective function
    Jfun = @(theta) GPDivInfer_Train_Obj(theta,obj,spec_decomp,NameValueArgs.Hyperprior,fwrap_v2s,fwrap_s2v);
    
    %LHS training grid
    hyp_init_vec = fwrap_s2v(hyp_init);
    hyp_lb_vec = hyp_init_vec - DeltaTrain;
    hyp_ub_vec = hyp_init_vec + DeltaTrain;
    hyp_search_vec = zeros(numel(hyp_init_vec),Ntrain);
    for idx = 1:numel(hyp_init_vec)
        hyp_search_vec(idx,:) = (hyp_ub_vec(idx)-hyp_lb_vec(idx)).*((randperm(Ntrain)-rand(1,Ntrain))/double(Ntrain)) + hyp_lb_vec(idx);
    end

    % Perform multistart optimization
    J_eval_vec = zeros(1,Ntrain);
    dJ_eval_vec = zeros(numel(hyp_init_vec),Ntrain);
    hyp_eval_vec = zeros(numel(hyp_init_vec),Ntrain);

    %Multistart
    for idx = 1:Ntrain
        hyp_eval = minimize(hyp_search_vec(:,idx),Jfun,NameValueArgs.MinimizeLength);
        [J_eval,dJ_eval] = Jfun(hyp_eval);

        % Log
        if NameValueArgs.ShowLog
            disp("----------");
            disp("Objective value at optimum: "+num2str(J_eval));
            disp("Gradient at optimum:");
            disp(num2str(reshape(dJ_eval,numel(dJ_eval),1)));
            disp("----------");
        end

        %Append vectors
        J_eval_vec(idx) = J_eval;
        dJ_eval_vec(:,idx) = dJ_eval;
        hyp_eval_vec(:,idx) = hyp_eval;
    end

    % Sort outputs
    [J_sort,sort_idx] = sort(J_eval_vec,'ascend');
    dJ_sort = [];
    hyp_sort = [];
    for idx = 1:numel(sort_idx)
        dJ_sort = [dJ_sort;fwrap_v2s(dJ_eval_vec(:,sort_idx(idx)))];
        hyp_sort = [hyp_sort;fwrap_v2s(hyp_eval_vec(:,sort_idx(idx)))];
    end

    % Select optimum
    hyp_opt = hyp_sort(1);
    obj_opt = J_sort(1);
    dobj_opt = dJ_sort(1);
    
    opt_info = struct('Objective',J_sort,'Gradient',dJ_sort,'Hyp',hyp_sort);

    % Log
    if NameValueArgs.ShowLog
        disp("----------");
        disp("----------");
        disp("----------");
        disp("Objective value at optimum: "+num2str(obj_opt));
        disp("Gradient at optimum: ");
        disp(num2str(reshape(fwrap_s2v(dobj_opt),3,1)));
        disp("----------");
    end
end

function [J,dJ] = GPDivInfer_Train_Obj(hyp,obj,spec_decomp,hyperprior,fwrap_v2s,fwrap_s2v)
    %Calculate core parameters
    try
        % Calculate core parameters
        [core,dcore] = calc_core(fwrap_v2s(hyp),spec_decomp,hyperprior);

        % Select objective for optimization
        switch obj
            case 'MAP'
                [J,dJ] = calc_map(core,dcore);
            case 'NLML'
                [J,dJ] = calc_nlml(core,dcore);
            case 'CV'
                [J,dJ] = calc_CV(core,dcore);
            case 'CVLOO'
                [J,dJ] = calc_CVLOO(core,dcore);
            otherwise
            error("Training objective "+obj+" not recognized. Choose NLML, CV, or CVLOO.");
        end
        dJ = fwrap_s2v(dJ);
    catch ME
        warning(ME.message);
        J = Inf;
        dJ = zeros(size(hyp));
    end
end