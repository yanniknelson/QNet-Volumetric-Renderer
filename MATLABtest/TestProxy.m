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

gpuDevice(1);

% settings
k = 600 ;               % number of neurons 
nbatches = 40;         % set >1 if training data will not fit in memory
nreps = 1;             % Repeat training procedure with iterative re-initialization
nepochs = 20;           % Need more epochs if the function is difficult to learn
useGPU = 'yes' ;        % 'yes' is Only beneficial if N and k are large

load('Anim.mat');

disp(a(1,:));
disp(a(2,:));


x = a(:, 1:4)';
y = a(:, 5)';

disp(size(x))
disp(size(y))

data = (x(1:3,x(4,:) == -1));
disp(size(data));
data = reshape(data, [3, 54,60,102]);
data = data(:, 26, :, :);
data = reshape(data, [3, 60*102]);
disp(size(data));
expec = reshape(y(x(4,:) == -1), 54,102,60);
expec = expec(2,:,:);
disp(size(expec));
expec = reshape(expec, 102,60);
disp(size(data));

imagesc(expec);

xs = repmat(0.5, 1, 117);
zs = linspace(-1,1,117);
ys = linspace(-1,1,117);

trainst = tic ;
    p = Proxy(x,y,k,nbatches,nreps,nepochs,useGPU) ; % proxy object p
traintime = toc(trainst) ;

res = reshape(p.EvalProxy(data), 63, 117);

pw1 = p.w1;
pb1 = p.b1;
pw2 = p.w2;
pb2 = p.b2;
yoffset = p.ys.xoffset;
ymin = p.ys.ymin;
yrange = p.ys.gain;


%save('anim_weights_v1', 'pw1', 'pb1', 'pw2', 'pb2', 'ymin', 'yrange', 'yoffset')



disp(size(res));
disp(min(min(res)))
surf(res);
