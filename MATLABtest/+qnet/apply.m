function a = apply(n,param)
%LOGSIG.APPLY Apply transfer function to inputs

% Copyright 2012-2015 The MathWorks, Inc.
  disp('apply');
  disp(n);
  dm = (log(size(n,1))/log(2) ) ;
  disp(dm);
  a = polylog(dm, -exp(n))  ;
  disp(a);
end


