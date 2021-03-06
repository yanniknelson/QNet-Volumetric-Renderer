
% Author: Kartic Subr

% Representative code for Q-NET
% Note: This is simplified code that is a re-implementation. 
%       It is not the code used in the paper and so could contain bugs 
%       that are not representative of the rigorously validated version for the paper.

classdef Proxy
    properties
        fnet                 % Shallow Neural network proxy for fx        
        I double {mustBeReal}   % Integral
        
        k  double {mustBeReal}  % number of neurons in hidden layer
        d  double {mustBeReal}  % number of inputs (dimensions)
        w1 (:,:) double {mustBeReal}
        w2 (:,:) double {mustBeReal}
        b1 (:,:) double {mustBeReal}
        b2 (:,:) double {mustBeReal}
        xs struct  % scale properties for x
        ys struct  % scale properties for fx
        
        tmp double      % temp storage for passing info
        EnableGPUComp char 
        MultiGPU char
    end
    
    methods
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = Proxy(x, fx, k, nbatches, nrepeats, nepochs, UseGPU, UseParallel)
            arguments
                x double;
                fx (1,:) double;
                k int32 {mustBeFinite,mustBePositive}=10;
                nbatches(1,1) double {mustBeFinite} = 1;
                nrepeats(1,1) double {mustBeFinite} = 1;
                nepochs(1,1) double {mustBeFinite} = 100;
                UseGPU char = 'no';
                UseParallel char = 'no' ;
            end
            obj.EnableGPUComp = UseGPU ;
            obj.MultiGPU = UseParallel ;
            
            
            batchsize = floor(size(x,2)/nbatches) ;
            nelem = size(x,2) ;
            startbatches = 1:batchsize:nelem ;
            
            obj.k = k ;
            obj.d = size(x,1) ;
            obj.fnet = feedforwardnet(double(k));
            disp(obj.fnet);
            obj.fnet.layers{1}.transferFcn = 'logsig';
            obj.fnet.divideParam.trainRatio = 100/100;
            obj.fnet.divideParam.valRatio = 0/100;
            obj.fnet.divideParam.testRatio = 0/100;
            obj.fnet.trainParam.min_grad = 1e-9 ;
            obj.fnet.trainParam.epochs = nepochs ;
            
            ngpus = gpuDeviceCount ;
            
%             if length(gpuDevice) & false
            if strcmp(obj.EnableGPUComp, 'yes') & ngpus
                disp('          Using GPU')                
                obj.fnet.trainFcn = 'trainscg';
                obj.fnet.input.processFcns = {'mapminmax'};
                obj.fnet.output.processFcns = {'mapminmax'};
            end
            
            if strcmp(UseParallel, 'yes') & (ngpus>1)
                disp('          Using parallel comp.')                                
            end
            
            [xn,obj.xs] = mapminmax(x);
            [yn,obj.ys] = mapminmax(fx);


            obj.fnet.trainParam.showWindow = true;
            
            % Train the Network
            for j=1:nrepeats
                for i=1:nbatches-1
                    [obj.fnet, tr] = train(network(obj.fnet),xn(:,startbatches(i):startbatches(i+1)),yn(:,startbatches(i):startbatches(i+1)), 'UseGPU', UseGPU, 'UseParallel', obj.MultiGPU);
                end
                [obj.fnet, tr] = train(network(obj.fnet),xn(:,startbatches(end):end),yn(:,startbatches(end):end), 'UseGPU', UseGPU, 'UseParallel', obj.MultiGPU);
            end
            
            obj.w1 = obj.fnet.IW{1} ;
            obj.b1 = obj.fnet.b{1} ;
            obj.w2 = obj.fnet.LW{2,1} ;
            obj.b2 = obj.fnet.b{2} ;
            
            if length(obj.w2)==0
                obj.w2 = ones(1, k) ;
            end
            if length(obj.b2)==0
                obj.b2 = 0 ;
            end
            obj.I = [] ;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = SetWeights(obj, w1, b1, w2, b2)
            obj.fnet.IW{1} = w1;
            obj.fnet.b{1} = b1;
            obj.fnet.LW{2,1} = w2 ;
            obj.fnet.b{2} = b2 ;
            obj.w1 = w1 ;
            obj.b1 = b1 ;
            obj.w2 = w2 ;
            obj.b2 = b2 ;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function ftx = EvalProxy(obj, x)
            N = network(obj.fnet) ;
            xmap = (mapminmax('apply', x, obj.xs)) ;
            xmap = x;
            ftx = mapminmax('reverse', N(xmap, 'UseGPU', obj.EnableGPUComp), obj.ys) ;
            ftx = N(xmap, 'UseGPU', obj.EnableGPUComp);
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Integral of proxy 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function [obj, mu]  = Integral (obj, ax, bx)
             % integrate over full range of inputs by default
            if nargin==1
                ax = obj.xs.xmin ;
                bx = obj.xs.xmax ;
            end
            ay = obj.ys.xmin ;
            by = obj.ys.xmax ;
            
            
            tpp = dec2bin(0:2^obj.d-1) -48 ;
            sP = (tpp==0)*-1 + tpp ;
            
            Qnet = feedforwardnet(2^obj.d) ;
            Qnet.inputs{1}.processFcns  = {};
            Qnet.outputs{2}.processFcns = {};
            
            Qnet.trainParam.showWindow = false;
            
            Qnet.trainParam.epochs= 1;
            [Qnet, tr2] = train(Qnet, rand(obj.d+1,10), rand(1,10)) ;
            Qnet.IW{1} = [sP repmat(-1, size(sP,1),1)] ;
            Qnet.LW{2,1} = -prod(Qnet.IW{1},2)' ;
            Qnet.b{1} = zeros(2^obj.d,1) ;
            Qnet.b{2} = 0 ;
            Qnet.Layers{1}.transferFcn = 'qnet' ;
            
            % vector (1D) vs tensor (>1D)
            %disp(size(obj.w1'));
            %disp(size(obj.b1'));
            %disp(size([obj.w1';obj.b1']));
            %disp([obj.w1';obj.b1']);
            if obj.d>1
                v = Qnet([obj.w1';obj.b1'])./prod(obj.w1') + 2^obj.d ;
            else
                v = Qnet([obj.w1';obj.b1'])./obj.w1' + 2^obj.d ;
            end
            
            Qvnet = obj.w2*real(v')+2^obj.d*obj.b2 ;
            mu = (by-ay)*prod(bx-ax)*(Qvnet/2^obj.d + 1)/2  + prod(bx-ax)*ay;
            obj.I = mu ;            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% The proxy is learned in [ax,bx]
        %%% but the integral calculated is in [suba, subb]
        %%% where suba and subb are in [ax, bx]
        %%% e.g. fctn over interval [4, 7] and integrate over [5, 6]
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function mu = IntegralSub (obj, ax, bx, suba, subb )
            % integrate over full range of inputs by default
            if nargin==1
                ax = obj.xs.xmin ;
                bx = obj.xs.xmax ;
                suba = -1 * ones(1,obj.d);
                subb = 1 * ones(1,obj.d);
            end
            
            aa = mapminmax('apply', ax+bx-subb, obj.xs) ;
            bb = mapminmax('apply', ax+bx-suba, obj.xs) ;
            vol = prod(bx-ax) ;
            volsub = prod(subb-suba) ;
            volsubn = prod(bb-aa) ;

            ay = obj.ys.xmin ;
            by = obj.ys.xmax ;
            
            tpp = dec2bin(0:2^obj.d-1) -48 ;
            sP = (tpp==0)*-1 + tpp ;
            outlw = prod(sP,2) ;
            
            for i=1:size(sP,2)
                loidx = find(sP(:,i)==-1) ;
                hiidx = find(sP(:,i)==1) ;
                sP(loidx,i) = aa(i) ;
                sP(hiidx,i) = bb(i) ;
            end
            
            
            Qnet = feedforwardnet(2^obj.d) ;
            Qnet.inputs{1}.processFcns  = {};
            Qnet.outputs{2}.processFcns = {};
            Qnet.trainParam.showWindow = false;
            Qnet.trainParam.epochs= 1;
            [Qnet, tr2] = train(Qnet, rand(obj.d+1,10), rand(1,10)) ;
            Qnet.IW{1} = [sP repmat(-1, size(sP,1),1)] ;
            Qnet.LW{2,1} = outlw' ;
            Qnet.b{1} = zeros(2^obj.d,1) ;
            Qnet.b{2} = 0 ;
            Qnet.Layers{1}.transferFcn = 'qnet' ;
            
            % vector (1D) vs tensor (>1D) 
            if obj.d>1
                v = Qnet([obj.w1';obj.b1'])./prod(obj.w1') + volsubn ;
            else
                v = Qnet([obj.w1';obj.b1'])./obj.w1' + volsubn;
            end
            
            Qvnet = obj.w2*real(v') + volsubn*obj.b2 ;
            mu = (by-ay)*vol*Qvnet/4  + volsub*(ay+by)/2;
        end
    end
end