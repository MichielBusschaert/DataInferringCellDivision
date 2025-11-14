% Step 0: Initialize workspace
clear all;
clc;
close all;
addpath(genpath("."));

%% Step 1: Load & process experimental data
%Load data
FCM_data = load("fcm_data.mat").fcm_structs;
diameter_freq = arrayfun(@(i) FCM_data(i).D_conv.*1e6, (1:numel(FCM_data))',UniformOutput=false);
exp_mu_avg = 0.0215;

%Geometric parameters
rho_chlam = 1.15; %pg{1} mum{-3} -- Chioccioli, 2014, Flow Cytometry Pulse Width Data Enables Rapid and Sensitvie Estimation of Biomass Dry Weight in the Microalgae Chlamydomonas reinhardtii and Chlorella vulgaris
x_chlam = @(D) rho_chlam*pi/6*D.^3; %x: pg{1}
dxdD_chlam = @(D) rho_chlam*pi/2*D.^2; %x: pg{1} mum{-1}
D_chlam = @(x) (6/pi*x/rho_chlam).^(1/3); %D: mum{1}
dDdx_chlam = @(x) (6/pi/rho_chlam/27*x.^(-2)).^(1/3);
x2D_chlam = @(x)  deal(D_chlam(x), dDdx_chlam(x));
D2x_chlam = @(D) deal(x_chlam(D), dxdD_chlam(D));

% Diameter grid 
D_grid = (0:.5:15)';
bw = .8;
train_per = 0.75;
[train_struct,valid_struct] = GP_Gamma.Support.GenerateTrainingValidationData(diameter_freq,D_grid,exp_mu_avg,bw,train_per,x_chlam,dxdD_chlam);

%Test space
D_test = linspace(min(D_grid),max(D_grid),300)';
%x_test = x_chlam(D_test);
x_test = linspace(min(x_chlam(D_grid)),max(x_chlam(D_grid)),300)';
ndf_test = {struct()};

%% Generalized division rate
%Weights & parameters
Z_ndiv = @(ndiv_max) [[-Inf;(2:ndiv_max)'],[(2:ndiv_max)';Inf]];
k_ndiv = @(ndiv_max) (2.^(1:ndiv_max))';
p_list_univar = @(ki,p1) arrayfun(@(l) max(round((1/l^2*(l-1)*ki(1)^2*(p1*ki(1)+1)/(ki(1)-1)-1)/l),1),ki);
weight_fun = @(y,div_idx,x_crit,w_std,Z) 0.5*(erf((max(log2(y./x_crit),-1e99)-Z(div_idx,1))/sqrt(2*w_std^2)) - erf((log2(y./x_crit)-Z(div_idx,2))/sqrt(2*w_std^2)));

%Division kernel and parameters
beta_fun_uni_div = @(x,y,k,p) max(k/beta(p,p*(k-1)).*((x./y).^(p-1)).*((1-x./y).^(p*(k-1)-1))./y.*(x>0).*(x<y),0);
beta_fun_multi = @(x,y,p_list,k_list,x_crit,w_std,Z) sum(cell2mat(arrayfun(@(di) weight_fun(y,di,x_crit,w_std,Z).*beta_fun_uni_div(x,y,k_list(di),p_list(di)),1:numel(k_list),UniformOutput=false)')); %Liu's weights

% Wrap beta fun multi
beta_fun_wrap = @(x,y,p,x_crit,s,ndiv_max) beta_fun_multi(x,y,p_list_univar(k_ndiv(ndiv_max),p),k_ndiv(ndiv_max),x_crit,s,Z_ndiv(ndiv_max));

%LHS sampling
p_list = 31:1:150;
N_mod = numel(p_list);
x_crit_list = logspace(log10(1),log10(1000),N_mod);
s_list = logspace(log10(1e-1),log10(1),N_mod);
ndiv_max_list = randi(4-1,1,N_mod)+1;
%nvdiv_max_list = repmat(4,1,N_mod);

% Reshuffle
p_list = p_list(randperm(N_mod));
x_crit_list = x_crit_list(randperm(N_mod));
s_list = s_list(randperm(N_mod));

beta_fun_list = cell(N_mod,1);
parameters = repmat(struct('p',[],'x_crit',[],'s',[],'ndiv_max',[]),N_mod,1);
for idx = 1:N_mod
    beta_fun_list{idx,1} = @(x,y) beta_fun_wrap(x,y,p_list(idx),x_crit_list(idx),s_list(idx),ndiv_max_list(idx));
    parameters(idx,1) = struct('p', p_list(idx), 'x_crit', x_crit_list(idx), 's', s_list(idx),'ndiv_max',ndiv_max_list(idx));
end

%% Simulation properties
% Forward simulation properties
x_sim = linspace(min(x_test),max(x_test),numel(x_test)+1)';
B0 = -1;
div_beta_fun = @(beta) GP_Gamma.Support.DivisionKernelMatrix(x_sim,beta);
sim_fun_beta = @(g,beta) GP_Gamma.Support.SolvePBMSteadyState(x_sim,exp_mu_avg,g,beta,B0);

%% Call MultiBeta function
hyp0 = struct();
lambda_list = [0.1,0.5,1,5,10,50,100,500,1000];
kernel_result = GP_Gamma.Support.MultipleBeta(x_test,ndf_test,train_struct,valid_struct,beta_fun_list,sim_fun_beta,div_beta_fun,...
    Hyp0=hyp0,L=1.5*max(x_test),m=500,Lambda=lambda_list,Parameters=parameters,EMDVariableJacobian=dDdx_chlam,...
    ParallelPool='');

%% Calculate average division numbers
nu = zeros(numel(x_test),numel(kernel_result.Beta));
for beta_idx = 1:numel(kernel_result.Beta)
    beta_fun = kernel_result.Beta{beta_idx};
    for x_idx = 1:numel(x_test)
        nu(x_idx,beta_idx) = integral(@(xi) beta_fun(xi,x_test(x_idx)),0,x_test(x_idx));
    end
end