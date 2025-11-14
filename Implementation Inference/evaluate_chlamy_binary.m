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

% Division rate
beta_fun_uni_div = @(x,y,k,p) max(k/beta(p,p*(k-1)).*((x./y).^(p-1)).*((1-x./y).^(p*(k-1)-1))./y.*(x>0).*(x<y),0);
beta_fun = @(x,y) beta_fun_uni_div(x,y,2,60);

%% Simulation properties
% Forward simulation properties
x_sim = linspace(min(x_test),max(x_test),numel(x_test)+1)';
B0 = -1;
div_beta_fun = @(beta) GP_Gamma.Support.DivisionKernelMatrix(x_sim,beta);
sim_fun_beta = @(g,beta) GP_Gamma.Support.SolvePBMSteadyState(x_sim,exp_mu_avg,g,beta,B0);

%% Call MultiBeta function (using 1 beta)
hyp0 = struct();
lambda_list = [0.1,0.5,1,5,10,50,100,500,1000];
kernel_result = GP_Gamma.Support.MultipleBeta(x_test,ndf_test,train_struct,valid_struct,{beta_fun},sim_fun_beta,div_beta_fun,...
    Hyp0=hyp0,L=1.5*max(x_test),m=500,Lambda=lambda_list,EMDVariableJacobian=dDdx_chlam,...
    ParallelPool='');