function phi = CovApxEigenOp(x,mode,ndf_info,phi_fun,lambda_fun,m)
    %Initialize
    phi = zeros(numel(x),m);
    lambda = zeros(m,1);

    %Parse NDF
    if strcmp(mode,'L')
        %ndf_fun = @(z) interp1(ndf_info.x,ndf_info.ndf,z,'linear',0);
        ndf_fun = griddedInterpolant(ndf_info.x,ndf_info.ndf,'linear','nearest');
        beta_fun = ndf_info.beta_fun;
        xmax = max(ndf_info.x);
    else
        ndf_fun = @(z) zeros(size(z));
        beta_fun = @(y,z) zeros(numel(y),numel(z));
        xmax = 0;
    end

    %Iterate over spectral frequencies
    for idx = 1:numel(x)
        z = x(idx);
        for n = 1:m
            %Eigenvalue
            %lambda(n) = lambda_fun(n);
            lambda(n) = lambda_fun(n);
            % phi_length = 2*pi/sqrt(lambda(n));
            % x_evalspace = linspace(z,xmax,max(ceil((xmax-z)/phi_length*40)+30,5e4))'; % Number of sample points based on wavelength of phi_fun + offset

            %Transformed eigenfunction
            switch mode
                case 'I'
                    %Identity operator
                    phi(idx,n) = phi_fun(z,n);
                case 'L'
                    %Division operator
                    %int_term = integral(@(xi) beta_fun(z,xi).*ndf_fun(xi).*phi_fun(xi,n),z,xmax,RelTol=1e-4,ArrayValued=true);
                    %phi(idx,n) = -ndf_fun(z)*phi_fun(z,n) + int_term;
                    integrand = @(xi) (beta_fun(z,xi).*ndf_fun(xi).*phi_fun(xi,n));
                    %int_term = integral(@(xi) beta_fun(z,xi).*ndf_fun(xi).*phi_fun(xi,n),z,xmax,RelTol=1e-4,ArrayValued=true);
                    int_term = integral(integrand,z,xmax,RelTol=1e-4);
                    phi(idx,n) = -ndf_fun(z)*phi_fun(z,n) + int_term;
            end
        end
    end
end