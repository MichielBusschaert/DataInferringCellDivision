function [xc,ndf] = SolvePBMSteadyState(x,mu_ct,gamma,beta_mat,B0)
    %% Solve Finite Volume Method
    % Define grid space
    xc = 0.5*(x(1:end-1)+x(2:end));
    dx = x(2:end)-x(1:end-1);
    N = numel(xc);

    % Evaluate division rates
    gamma_vec = gamma(xc);

    % Initialize linear PBM equations
    A_PBM  = zeros(N,N);
    b_PBM = zeros(N,1);
    for i = 1:N
        % Balance term
        A_PBM(i,i) = A_PBM(i,i) + mu_ct*dx(i);

        % Growth term
        if i ~= 1
            A_PBM(i,i-1) = A_PBM(i,i-1) - mu_ct*xc(i-1);
        end
        A_PBM(i,i) = A_PBM(i,i) + mu_ct*xc(i);

        % Division - Death
        A_PBM(i,i) = A_PBM(i,i) + gamma_vec(i)*dx(i);

        % Division - Birth
        for j = 1:N
            A_PBM(i,j) = A_PBM(i,j) - beta_mat(i,j)*gamma_vec(j)*dx(j); 
        end
    end

    % Set normalization constant
    if B0 <= 0
        % Normalize with respect to 1-measure
        A_norm = reshape(dx,1,N);
        b_norm = -B0;
    else
        % Normalize with respect to x-measure
        A_norm = reshape(xc.*dx,1,N);
        b_norm = B0;
    end

    % Solve system of equations
    A = [A_PBM;A_norm];
    b = [b_PBM;b_norm];
    ndf = A\b;
end