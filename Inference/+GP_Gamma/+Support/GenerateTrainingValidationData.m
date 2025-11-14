function [train_struct,valid_struct] = GenerateTrainingValidationData(D_freq,D_grid,mu_exp,kde_bw,train_data,D2x,dD2x)
%% GENERATETRAININGVALIDATIONDATA Constructs the training & validation data for inference.
% Important! Diameters must be micrometers
arguments
    D_freq cell;
    D_grid double;
    mu_exp double;
    kde_bw double;
    train_data;
    D2x = @(x) x;
    dD2x = @(x) ones(size(x));
end

    % Identify data to assign to training or validation sets
    if isscalar(train_data)
        % Percentage given -- randomly assign
        rand_selec = randperm(numel(D_freq))./numel(D_freq);
        is_train_data = rand_selec <= train_data;
    else
        % Training/validation specified
        is_train_data = train_data;
    end

    %Iterate over training data
    x_cell = cell(numel(D_freq),1);
    y_cell = cell(numel(D_freq),1);
    ndf_cell = cell(numel(D_freq),1);
    op_mode = repmat('L',numel(D_freq),1);
    for idx = 1:numel(D_freq)
        %Kernel density estimate
        ndf_D = ksdensity(D_freq{idx},D_grid,Bandwidth=kde_bw);
        ndf_D = GP_Gamma.Support.NormalizedProbabilityMetrics(ndf_D,D_grid);
        [x_grid,ndf_x] = GP_Gamma.Support.ConvertNDF(ndf_D,D_grid,D2x,dD2x,Normalize=true);
        %ndf_x = ndf_D;
    
        %Find derivatives
        exp_dmuxndfdx = derivative_approx(x_grid,mu_exp.*x_grid.*ndf_x);
        Lx_gamma = mu_exp.*ndf_x + exp_dmuxndfdx;

        x_cell{idx} = x_grid;
        y_cell{idx} = Lx_gamma;
        ndf_cell{idx} = struct('x',x_grid,'ndf',ndf_x);
    end
    % Assign to struct
    train_struct = struct('x',{x_cell(is_train_data)},...
        'y',{y_cell(is_train_data)},...
        'ndf',{ndf_cell(is_train_data)},...
        'op_mode',op_mode(is_train_data));
    valid_struct = struct('x',{x_cell(~is_train_data)},...
        'y',{y_cell(~is_train_data)},...
        'ndf',{ndf_cell(~is_train_data)},...
        'op_mode',op_mode(~is_train_data));
end