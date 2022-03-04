%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Q-NET: 1D Tests
% 
% Calculate Integrals
%    I  over [-1,1]  and 
%    Isub over [.5,1] 
% of 4 test functions 
%
% Step 1: Train proxy
% Step 2: Calculate I
% Step 3: Calculate Isub
% Step 4: Output
% Step 5: Plot
% 
% Proxy object p usage: 
%    p = Proxy(x,y,k,...)
%    p.Integral(-1,1) 
% 
%
% Same for d>1
% Only change is the training input 
%   x is d x N
%   y is 1 x N
%%%%%%%%%%%%%%%%%%%%%%%%%%
fv=[] ;
fv{1} = @(x) (x<.5).*x.^2 + (x>.5).*x ;                                             % test function 1
fv{2} = @(x) (x<.5).*sin(2*pi*x) + (x>.175).*(x<.625)*.5  ;                         % test function 2
fv{3} = @(x) (x<.5).*sin(20*pi*x) + (x>.175).*(x<.625)*.5 + (x>.7).*cos(5*pi*x) ;   % test function 3
fv{4} = @(x) sin(8*pi*x) >0 ;                                                       % test function 4

% settings
k = 500 ;               % number of neurons 
nbatches = 1;         % set >1 if training data will not fit in memory
nreps = 1;             % Repeat training procedure with iterative re-initialization
nepochs = 50;           % Need more epochs if the function is difficult to learn
useGPU = 'no' ;        % 'yes' is Only beneficial if N and k are large

load('volume_data.mat');
x = a(:, 1:3)';
y = a(:, 4)';

disp(size(x))
disp(size(y))

data = (x(:,x(3,:) == 0));
expec = reshape(y(x(3,:) == 0), 117,117);
disp(size(data));


imagesc(expec);


xs = repmat(0.5, 1, 117);
zs = linspace(-1,1,117);
ys = linspace(-1,1,117);

trainst = tic ;
    p = Proxy(x,y,k,nbatches,nreps,nepochs,useGPU) ; % proxy object p
traintime = toc(trainst) ;

res = reshape(p.EvalProxy(data), 117, 117);


disp(size(res));
imagesc(res);
