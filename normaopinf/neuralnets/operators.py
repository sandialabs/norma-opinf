import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
torch.set_default_dtype(torch.float64)


class CompositeOperator(nn.Module):
    def __init__(self,state_operators):
        super(CompositeStateOperator, self).__init__()
        self.state_operators = state_operators
    def forward(self,x,parameters):
        result = self.state_operators[0].forward(x,parameters)
        for state_operator in self.state_operators[1::]:
          result += state_operator.forward(x,parameters)
        return result 

 

class NpdOperator(nn.Module):
    def __init__(self,n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs):
        super(NpdOperator, self).__init__()
        self.SpdOperator = SpdOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)

    def forward(self,x,params):
        result = self.SpdOperator.forward(x,params)
        return -result

class SpdOperator(nn.Module):
    def __init__(self,n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs):
        super(SpdOperator, self).__init__()
        forward_list = []
        idx = np.tril_indices(n_outputs)
        self.idx = idx

        self.n_outputs = n_outputs
        networkOutputSize = idx[0].size
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        dim = np.zeros(n_hidden_layers+2,dtype='int')
        dim[0] = n_inputs
        for i in range(1,n_hidden_layers+1):
         dim[i] = n_neurons_per_layer
        dim[-1] = networkOutputSize
        self.dim = dim
        input_dim = dim[0:-1]
        output_dim = dim[1::]

        for i in range(0,n_hidden_layers+1):
          forward_list.append(nn.Linear(input_dim[i], output_dim[i]))

        self.forward_list = nn.ModuleList(forward_list)

        self.activation = F.relu

    def forward(self,x,parameters):
      y = torch.cat(( x , parameters ) , 1)

      for i in range(0,self.num_layers-1):
        y = self.activation(self.forward_list[i](y))

      y = self.forward_list[-1](y)

      K = torch.zeros(y.shape[0],self.n_outputs,self.n_outputs)

      K[:,self.idx[0],self.idx[1]] = y[:,0:self.idx[0].size]

      KT = torch.transpose(K,2,1)
      K = torch.einsum('ijk,ikl->ijl',K,KT)

      self.system_matrix_ =  K

      result = torch.einsum('ijk,ik->ij',K,x[:,0:self.n_outputs])
      return result[:,:]


class SkewOperator(nn.Module):
    def __init__(self,n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs):
        super(SkewOperator, self).__init__()
        forward_list = []
        idx = np.tril_indices(n_outputs)
        self.idx = idx

        self.n_outputs = n_outputs
        networkOutputSize = idx[0].size
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        dim = np.zeros(n_hidden_layers+2,dtype='int')
        dim[0] = n_inputs
        for i in range(1,n_hidden_layers+1):
         dim[i] = n_neurons_per_layer
        dim[-1] = networkOutputSize
        self.dim = dim
        input_dim = dim[0:-1]
        output_dim = dim[1::]

        for i in range(0,n_hidden_layers+1):
          forward_list.append(nn.Linear(input_dim[i], output_dim[i]))

        self.forward_list = nn.ModuleList(forward_list)
        self.activation = F.relu

    def forward(self,x , parameters):
      y = torch.cat(( x , parameters ) , 1)

      for i in range(0,self.num_layers-1):
        y = self.activation(self.forward_list[i](y))

      y = self.forward_list[-1](y)

      S = torch.zeros(y.shape[0],self.n_outputs,self.n_outputs)

      S[:,self.idx[0],self.idx[1]] = y[:,0:self.idx[0].size]

      ST = torch.transpose(S,2,1)
      R = S - ST 

      self.system_matrix_ = R

      result = torch.einsum('ijk,ik->ij',self.system_matrix,x[:,0:self.n_outputs])
      return result[:,:]



class StandardOperator(nn.Module):
    def __init__(self,n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs):
        super(StandardOperator, self).__init__()
        forward_list = []
        self.n_outputs = n_outputs
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        dim = np.zeros(n_hidden_layers+2,dtype='int')
        dim[0] = n_inputs
        for i in range(1,n_hidden_layers+1):
         dim[i] = n_neurons_per_layer
        dim[-1] = self.n_outputs 
        self.dim = dim
        input_dim = dim[0:-1]
        output_dim = dim[1::]

        for i in range(0,n_hidden_layers+1):
          forward_list.append(nn.Linear(input_dim[i], output_dim[i]))

        self.forward_list = nn.ModuleList(forward_list)

        self.activation = F.relu

    def forward(self,x , parameters):
      y = torch.cat(( x , parameters ) , 1)
      for i in range(0,self.num_layers-1):
        y = self.activation(self.forward_list[i](y))
      result = self.forward_list[-1](y)
      return result[:,:]


if __name__ == "__main__":
    n_hidden_layers = 2
    n_neurons_per_layer = 5
    n_inputs = 5
    n_outputs = 5
    inputs = torch.tensor(np.random.normal(size=(10,n_inputs)))
    parameters = torch.tensor(np.random.normal(size=(10,0)))

    
    StandardMlp = StandardOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    SpdMlp = SpdOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    NpdMlp = NpdOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    SkewMlp = SkewOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)

    ops = [StandardMlp,NpdMlp,SkewMlp]
    CompositeMlp = CompositeOperator(ops) 
    r1 = StandardMlp.forward(inputs,parameters)
    r2 = NpdMlp.forward(inputs,parameters)
    r3 = SkewMlp.forward(inputs,parameters)
    r4 = CompositeMlp(inputs,parameters)
    assert np.allclose( r4.detach().numpy(), (r1 + r2 + r3).detach().numpy())
