function dfdx = derivative_approx(x,f)
    %Initialize
    dfdx = zeros(size(x));
    for idx = 1:numel(x)
        if idx == 1
            %Forward difference
            dfdx(idx) = (f(idx+1)-f(idx))/(x(idx+1)-x(idx));
        elseif idx == numel(x)
            %Backward difference
            dfdx(idx) = (f(idx)-f(idx-1))/(x(idx)-x(idx-1));
        else
            %1st order central difference
            dfdx(idx) = (f(idx+1)-f(idx-1))/(x(idx+1)-x(idx-1));
        end        
    end
end