function phi = CovApxEigenfun(x,L,n)
    %Solution of the eigenproblem
    % phi_n(x) = 1/sqrt(L)*sin(n*pi*(x-L)/(2*L))
    phi = 1/sqrt(L)*sin(n*pi*(x-L)/(2*L)).*(x<=L).*(x>=0);
end