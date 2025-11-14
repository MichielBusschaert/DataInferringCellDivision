function beta_mat = DivisionKernelMatrix(x,beta_fun)
    %% Construct via numerical integration
    % x contains grid boundaries, points in the center
    xc = 0.5*(x(1:end-1)+x(2:end));
    dx = x(2:end)-x(1:end-1);

    % Construct beta division matrix
    beta_mat = zeros(numel(xc),numel(xc));
    % Iterate over each grid point
    for i = 1:numel(xc)
        for j = i:numel(xc)
            beta_mat(i,j) = integral(@(z) beta_fun(z,xc(j)),xc(i)-0.5*dx(i),min(xc(i)+0.5*dx(i),xc(j)));
        end
    end
end