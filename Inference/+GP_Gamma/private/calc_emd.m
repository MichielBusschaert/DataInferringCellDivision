function [EMD] = calc_emd(x1,ndf1,x2,ndf2,df_x2y)
arguments (Input)
    x1 double;
    ndf1 double;
    x2 double;
    ndf2 double;
    df_x2y = @(x) ones(size(x));
end

    % Define resampled grid
    x = linspace(min([x1;x2]),max([x1;x2]),max(numel([x1;x2])))';

    % Resample
    ndf1 = interp1(x1,ndf1,x,'spline','extrap');
    ndf2 = interp1(x2,ndf2,x,'spline','extrap');

    % Coordinate transformation
    %Tndf1 = arrayfun(@(i) );

    % Calculate cumulative density function
    cdf1 = cumtrapz(x,ndf1);
    cdf2 = cumtrapz(x,ndf2);

    % Normalize
    cdf1 = cdf1./cdf1(end);
    cdf2 = cdf2./cdf2(end);

    % Calculate in transformed coordinates
    dydx = arrayfun(df_x2y, x);
    dydx(isinf(dydx)|isnan(dydx)) = 0;

    % Evaluate EMD value
    EMD = trapz(x,abs(cdf1-cdf2).*dydx);
end

%%% EMD in coordinate transformation basis:
% y = f(x), with df(x)dx > 0 and f(0) = 0
% x = f-1(y), df-1dy(y) = 1/dfdx(x) > 0, f-1(0) = 0
% Then ny(y) = nx(x)/dydx(x)
% and cy(y) = cx(x)
% EMD_y = int{|cy_1(y) - cy2(y)|*dy} = int{|cx_1(x) - cx2(x)|dfdx(x)*dx}