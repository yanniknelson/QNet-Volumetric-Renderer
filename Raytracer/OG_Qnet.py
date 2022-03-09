from Ray import *
import scipy.io as scio
import matplotlib.pyplot as plt
import torch


device = torch.device("cuda:0")
type = torch.float64

class Poly1(torch.nn.Module):
    def __init__(self,dimension):
        super().__init__()
        self.dim = dimension

    def forward(self, x):
        # return poly(1, self.dim)
        return -torch.log(1 - (-torch.exp(x)))

class Qnet(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        with torch.no_grad():
            self.model = torch.nn.Sequential(torch.nn.Linear(dimension+1, 2**dimension, bias=False, dtype=type, device=device), 
            Poly1(dimension), 
            torch.nn.Linear(2**dimension, 1, bias=False, dtype=type,device=device))

        tpp = []
        for i in np.arange(0, 2**dimension, 1):
            tpp.append(list(np.binary_repr(i, width=dimension)))
        
        tpp = np.array(tpp).astype(float)
        S = torch.tensor(np.concatenate(((tpp == 0) * -1 + tpp, np.full((2**dimension, 1), -1)), axis=1), dtype=type, device=device)
        w3 = (-torch.prod(S, axis=1).T).reshape(1,S.size(0))

        self.model[0].weight = torch.nn.Parameter(S)
        self.model[2].weight = torch.nn.Parameter(w3)

    def forward(self, x):
        return self.model(x)

class Intergrator():
    def __init__(self, W1 = None, B1 = None, W2 = None, B2 = None):
        self.qnet = Qnet(1)
        if (W1 is None) or (B1 is None) or (W2 is None) or (B2 is None):
            return
        self.train(W1, B1, W2, B2)

    def train(self, W1, B1, W2, B2):
        self.baseW1 = W1
        self.baseB1 = B1
        self.baseW2 = W2
        self.baseB2 = B2

        self.W1 = W1
        self.B1 = B1
        self.W2 = W2
        self.B2 = B2

    def Transform(self, startpos, dir):
        dir = dir/np.linalg.norm(dir)
        b = np.array([1,0,0])
        v = np.cross(b, dir)
        c = dir[0] # dir dot b (will select the first element of b, so skip calculation)
        skew = np.array([[0, -v[2], v[1]],[v[2],0,-v[0]],[-v[1], v[0], 0]])
        self.rot = torch.tensor(np.eye(3) + skew + np.dot(skew, skew)/(1+c), dtype=type, device=device)
        self.c = torch.tensor(startpos, dtype=type, device=device)
        self.B1 = self.baseB1 + torch.matmul(self.baseW1, self.c)
        self.W1 = torch.matmul(self.baseW1, self.rot)

    def TransformRay(self, ray):
        self.Transform(ray.o, ray.dir)

    def apply(self, start = -1, end = 1):
        # Slicing the Z and Y dimensions
        xDim = self.W1[:,:1] # get the x weights 
        yzDims = self.W1[:, 1:] # get the z and y weights
        newb1 = self.B1 + yzDims.matmul(torch.tensor([0,0],dtype=type, device=device)) # update bais
        self.qnet.model[0].weight[0][0] = -end
        self.qnet.model[0].weight[1][0] = -start
        y = torch.cat((xDim, newb1.reshape(xDim.size(0),1)), axis=1)
        res = torch.div(self.qnet(y), torch.prod(xDim, axis = 1).reshape(xDim.size(0),1)) + end-start
        self.model = torch.nn.Linear(self.W2.size(1),1, dtype=type,device=device)
        self.model.weight = torch.nn.Parameter(self.W2)
        self.model.bias = torch.nn.Parameter(self.B2*(end-start))#2**self.dim))

        return self.model(res.T)

