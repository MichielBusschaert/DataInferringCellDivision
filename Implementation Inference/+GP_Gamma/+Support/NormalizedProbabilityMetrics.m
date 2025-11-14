function [n_norm,c_norm,norm_const] = NormalizedProbabilityMetrics(n,x)
arguments
    n double;
    x double;
end
    % Determine (unnormalized) cumulative density function
    c = zeros(size(n));
    c(1) = 0.5*n(1)*x(1);
    for i = 2:numel(x)
        c(i) = c(i-1) + 0.5*(n(i)+n(i-1))*(x(i)-x(i-1));
    end

    % Normalize
    norm_const = c(end);
    n_norm = n./norm_const;
    c_norm = c./norm_const;
end