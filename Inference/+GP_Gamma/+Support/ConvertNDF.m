function [y,n_y] = ConvertNDF(n_x,x,f_x2y,df_x2y,NameValueArgs)
arguments 
    n_x double;
    x double;
    f_x2y;
    df_x2y;
    NameValueArgs.y double = [];
    NameValueArgs.Normalize logical = false;
end
    % Convert at prespecified grid points
    y = arrayfun(f_x2y, x);
    dydx = arrayfun(df_x2y, x);
    n_y = n_x./dydx;
    n_y(isinf(n_y)|isnan(n_y)) = 0;

    % If y is specified, resample NDF
    if ~isempty(NameValueArgs.y)
        n_y = max(interp1(y,n_y,NameValueArgs.y,"linear","extrap"),0);
        y = NameValueArgs.y;
    end

    % Normalize if needed
    if NameValueArgs.Normalize
        n_y = GP_Gamma.Support.NormalizedProbabilityMetrics(n_y,y);
    end
end