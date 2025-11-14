function spec_decomp = spectral(ndf_train,ndf_test,x_train,y_train,x_test,op_mode_train,op_mode_test,L,m)
%% SPEC_DECOMP Performs the spectral decomposition
    %Eigenvalues and eigenfunctions
    lambda_fun = @(n) CovApxEigenval(L,n);
    phi_fun = @(x,n) CovApxEigenfun(x,L,n);
    lambda = arrayfun(@(n) lambda_fun(n),1:m)';
    spec_fun = @(hyp,w) CovApxSpecFun(hyp,w);

    %Evaluate train cases
    x_train_vec = [];
    phi_train = [];
    batch_train = [];
    y_train_vec = [];
    for idx = 1:numel(ndf_train)
        %Input space
        phi_train_inst = CovApxEigenOp(x_train{idx},op_mode_train(idx),ndf_train{idx},phi_fun,lambda_fun,m);
        phi_train = [phi_train;phi_train_inst];
        x_train_vec = [x_train_vec;x_train{idx}];
        batch_train = [batch_train;idx*ones(numel(x_train{idx}),1)];
        y_train_vec = [y_train_vec;y_train{idx}];
    end

    %Evaluate test cases
    x_test_vec = [];
    phi_test = [];
    batch_test = [];
    for idx = 1:numel(ndf_test)
        phi_test_inst = CovApxEigenOp(x_test{idx},op_mode_test(idx),ndf_test{idx},phi_fun,lambda_fun,m);
        phi_test = [phi_test;phi_test_inst];
        x_test_vec = [x_test_vec;x_test{idx}];
        batch_test = [batch_test;idx*ones(numel(x_test{idx}),1)];
    end

    % Store results
    spec_decomp = struct('spec_fun',spec_fun,...
        'lambda_fun',lambda_fun,...
        'phi_fun',lambda_fun,...
        'lambda',lambda,...
        'x_train',x_train_vec,...
        'x_test',x_test_vec,...
        'phi_train',phi_train,...
        'phi_test',phi_test,...
        'y_train',y_train_vec,...
        'batch_train',batch_train,...
        'batch_test',batch_test);
end