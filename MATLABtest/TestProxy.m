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
nepochs = 20;           % Need more epochs if the function is difficult to learn
useGPU = 'no' ;        % 'yes' is Only beneficial if N and k are large

load('Frame31.mat');
x = a(:, 1:3)';
y = a(:, 4)';

disp(size(x))
disp(size(y))

disp(a(26, 1:3));

data = (x(:,x(1,:) == (25/(54-1))*2 - 1));
disp(size(data));
expec = reshape(y(x(1,:) == (25/(54-1))*2 - 1), 102, 56);

imagesc(expec);

xs = repmat(0.5, 1, 117);
zs = linspace(-1,1,117);
ys = linspace(-1,1,117);

trainst = tic ;
    p = Proxy(x,y,k,nbatches,nreps,nepochs,useGPU) ; % proxy object p
traintime = toc(trainst) ;

res = reshape(p.EvalProxy(data), 102, 56);

pw1 = p.w1;
pb1 = p.b1;
pw2 = p.w2;
pb2 = p.b2;
yoffset = p.ys.xoffset;
ymin = p.ys.ymin;
yrange = p.ys.gain;


%save('Frame31_weights_v1', 'pw1', 'pb1', 'pw2', 'pb2', 'ymin', 'yrange', 'yoffset')   



disp(size(res));
disp(min(min(res)))
surf(res);
