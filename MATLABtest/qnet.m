function a = qnet(n,varargin)
%QNET Quadrature Network Transfer Fcn
%
% Transfer functions convert a neural network layer's net input into
% its net output.
%	

% NNET 7.0 Compatibility
% WARNING - This functionality may be removed in future versions
disp(n);
if nargin > 0
    n = convertStringsToChars(n);
end
if ischar(n)
  disp('ischar');
  a = nnet7.transfer_fcn(mfilename,n,varargin{:});
  disp(a);  
  return
end

% Apply
disp(qnet.apply(n));
a = qnet.apply(n);

