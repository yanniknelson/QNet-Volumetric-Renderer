from Ray import *
import scipy.io as scio
import matplotlib.pyplot as plt
import torch


type = torch.float64

class Poly1(torch.nn.Module):
    def __init__(self,dimension):
        super().__init__()
        self.dim = dimension

    def forward(self, x):
        # return poly(1, self.dim)
        return -torch.log(1 - (-torch.exp(x)))

class Qnet(torch.nn.Module):
    def __init__(self, dimension, device):
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
    def __init__(self, W1 = None, B1 = None, W2 = None, B2 = None, map = False, yoffset = None, ymin = None, yrange = None, gpu = False):
        if (W1 is None) or (B1 is None) or (W2 is None) or (B2 is None):
            return
        if gpu:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.qnet = Qnet(1, self.device)
        self.train(W1, B1, W2, B2, map, yoffset, ymin, yrange)

    def train(self, W1, B1, W2, B2, map = False, yoffset = None, ymin = None, yrange = None):
        self.map = map         
        self.yoffset = yoffset
        self.ymin = ymin
        self.yrange = yrange

        W2 = torch.tensor(W2, dtype=type, device=self.device)
        self.baseW1 = torch.tensor(W1, dtype=type, device=self.device)
        self.baseB1 = torch.squeeze(torch.tensor(B1, dtype=type, device=self.device))
        self.baseB2 = torch.squeeze(torch.tensor(B2, dtype=type, device=self.device))
        self.model = torch.nn.Linear(W2.size(1),1, dtype=type,device=self.device)
        self.model.weight = torch.nn.Parameter(W2)
        self.model.bias = torch.nn.Parameter(self.baseB2)

    def IntegrateRay(self, ray, t0, t1):
        segment_length = t1-t0
        #transform weights
        dir = ray.d/np.linalg.norm(ray.d)
        b = np.array([1,0,0])
        rot = np.array([[-1, 0, 0],[0,-1,0],[0,0,1]])
        if np.any(dir != -b):
            v = np.cross(b, dir)
            c = dir[0] # dir dot b (will select the first element of b, so skip calculation)
            skew = np.array([[0, -v[2], v[1]],[v[2],0,-v[0]],[-v[1], v[0], 0]])
            rot = np.eye(3) + skew + np.dot(skew, skew)/(1+c)
        rot = torch.tensor(rot, dtype=type, device=self.device)
        c = torch.tensor(ray.o + ray.d * t0, dtype=type, device=self.device)
        B1 = self.baseB1 + torch.matmul(self.baseW1, c)
        W1 = torch.matmul(self.baseW1, rot)
        #slice
        xDim = W1[:,:1] # get the x weights 
        yzDims = W1[:, 1:] # get the z and y weights
        newb1 = B1 + yzDims.matmul(torch.tensor([0,0],dtype=type, device=self.device)) # update bais
        #intergrate over interval [0, t1-t0]
        self.qnet.model[0].weight[0][0] = -(segment_length)
        self.qnet.model[0].weight[1][0] = 0
        #integrate
        y = torch.cat((xDim, newb1.reshape(xDim.size(0),1)), axis=1)
        res = torch.div(self.qnet(y), torch.prod(xDim, axis = 1).reshape(xDim.size(0),1)) + (segment_length)
        self.model.bias = torch.nn.Parameter(self.baseB2*(segment_length))
        return (self.model(res.T) + segment_length)/self.yrange

