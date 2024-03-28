import torch 
from torch import nn
import torch.nn.functional as F 
import numpy as np 
from torch.nn import Dropout
from torch_geometric.nn import GCNConv, GATv2Conv, GIN, global_add_pool, aggr, global_mean_pool, GatedGraphConv, GraphConv
from torch_geometric_temporal.nn.recurrent import GConvGRU, DCRNN, AGCRN, GCLSTM, TGCN
from torch_geometric_temporal.nn.attention import STConv, TemporalConv, GMAN


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
    
    def forward(self, obs):
        # convert observation tensor to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
    
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        # print(output.shape)
        return output

class MLP(nn.Module):
    """Multilayer Perceptron Network."""

    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.aggregation = aggr.MeanAggregation()
        self.layer1 = nn.Linear(in_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, out_dim)
        

    def forward(self, obs):
        # convert observation tensor to tensor if it's a numpy array
        
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif isinstance(obs, list):
            obs = torch.tensor(np.array(obs), dtype=torch.float)
            # obs = torch.tensor(obs, dtype=torch.float)
        index = torch.tensor([0])
        obs = self.aggregation(obs, index)
        h1 = F.relu(self.layer1(obs))
        h2 = F.relu(self.layer2(h1))
        
        output = self.layer3(h2)
        # print("MLP output[-1] shape: ", output[-1].shape)
        # print("MLP output shape: ", output.shape)
        return output[-1]


class GCN(nn.Module):
    """ Graph Convolutional Network Architecture"""
    def __init__(self, in_dim, out_dim):
        
        super(GCN, self).__init__()
       
        self.layer1 = GCNConv(in_dim, 64)
        self.layer2 = GCNConv(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
        

    def forward(self, obs, edge_index):
        # convert observation tensor to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif isinstance(obs, list):
            obs = torch.tensor(np.array(obs), dtype=torch.float)

        h1 = F.relu(self.layer1(obs, edge_index))
        h2 = F.relu(self.layer2(h1, edge_index))
        
        output = self.layer3(h2)
        # print(output[-1].shape)
        return output[-1]


class GCN_mean(nn.Module):
    """ Graph Convolutional Network Architecture"""
    def __init__(self, in_dim, out_dim):
        
        super(GCN_mean, self).__init__()
       
        self.layer1 = GCNConv(in_dim, 64)
        self.layer2 = GCNConv(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
        

    def forward(self, obs, edge_index):
        # convert observation tensor to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif isinstance(obs, list):
            obs = torch.tensor(np.array(obs), dtype=torch.float)

        h1 = F.relu(self.layer1(obs, edge_index))
        h2 = F.relu(self.layer2(h1, edge_index))
        
        output = self.layer3(h2)
        # print(output.shape)
        output = torch.mean(output, dim=0)
        # print(output.shape)
        return output

class GCN_new(nn.Module):
    """ Graph Convolutional Network Architecture"""
    def __init__(self, in_dim, out_dim):
        
        super(GCN_new, self).__init__()
       
        self.layer1 = GCNConv(in_dim, 64)
        self.layer2 = GCNConv(64, 64)
        self.layer3 = nn.Linear(64, 1)
        

    def forward(self, obs, edge_index):
        # convert observation tensor to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif isinstance(obs, list):
            obs = torch.tensor(np.array(obs), dtype=torch.float)

        h1 = F.relu(self.layer1(obs, edge_index))
        h2 = F.relu(self.layer2(h1, edge_index))
        
        output = self.layer3(h2)
        # print(output.shape)
        # retrieve every 12 predictions, excluding first 3 for root nodes
        output = output.reshape(-1,15)[:,3:].reshape(-1)
        # output = output[3:,:]
        output = torch.flatten(output)
        # print(output.shape)
        return output
        # return output[3:]

class GConvN(nn.Module):
    """ Graph Convolutional Network Architecture"""
    def __init__(self, in_dim, out_dim):
        
        super(GConvN, self).__init__()
       
        self.layer1 = GCNConv(in_dim, 64)
        self.layer2 = GCNConv(64, 1)
        #self.layer3 = nn.Linear(64, 1)
        

    def forward(self, obs, edge_index):
        # convert observation tensor to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif isinstance(obs, list):
            obs = torch.tensor(np.array(obs), dtype=torch.float)

        h1 = F.relu(self.layer1(obs, edge_index))
        h2 = F.relu(self.layer2(h1, edge_index))
        
        output = h2 #self.layer3(h2)
        # print(output.shape)
        # retrieve every 12 predictions, excluding first 3 for root nodes
        output = output.reshape(-1,15)[:,3:].reshape(-1)
        # output = output[3:,:]
        output = torch.flatten(output)
        # print(output.shape)
        return output
        # return output[3:]

class GAT(nn.Module):
  """Graph Attention Network"""
  def __init__(self, in_dim, out_dim):
    super(GAT, self).__init__()
    
    self.gat1 = GATv2Conv(in_dim, 128)
    self.gat2 = GATv2Conv(128, 64)
    self.layer3 = nn.Linear(64, out_dim)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.005,
                                      weight_decay=5e-4)

  def forward(self, obs, edge_index):
    if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
    elif isinstance(obs, list):
        obs = torch.tensor(np.array(obs), dtype=torch.float)
    h = F.dropout(obs, p=0.6)
    h = self.gat1(h, edge_index)
    h = F.relu(h)
    h = F.dropout(h, p=0.6)
    h = self.gat2(h, edge_index)
    output = self.layer3(h)
    # print(output.shape)
    return output[-1,:]

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(dataset.num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h, F.log_softmax(h, dim=1)

class rGIN(nn.Module):
  """Graph Isomorphism Network"""
  def __init__(self, in_dim, out_dim, num_layers: int = 2):
    super(rGIN, self).__init__()
    
    self.gin = GIN(in_dim, 64, num_layers)
    
    self.layer2 = nn.Linear(64, out_dim)

  def forward(self, obs, edge_index):
    if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
    elif isinstance(obs, list):
        obs = torch.tensor(np.array(obs), dtype=torch.float)
    
    h = self.gin(obs, edge_index)
    h = global_add_pool(h, None)
    output = self.layer2(h)
    # print(output.shape)
    return output

class RecurrentGCN(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(RecurrentGCN, self).__init__()
    self.recurrent1 = GConvGRU(in_dim, 32, 3)
    self.recurrent2 = GConvGRU(32, 32, 3)
    self.linear = nn.Linear(32, out_dim)

  def forward(self, obs, edge_index):
    if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
    elif isinstance(obs, list):
        obs = torch.tensor(np.array(obs), dtype=torch.float)
    
    h = self.recurrent1(obs, edge_index)
    h = F.relu(h)
    h = self.recurrent2(h, edge_index)
    h = F.relu(h)
    h = global_mean_pool(h)
    output = self.linear(h)
    #output = output.reshape(-1,15)[:,3:].reshape(-1)
    output = output.reshape(-1,11)[:,1:].reshape(-1)
    #output = output.reshape(-1,23)[:,1:].reshape(-1)
    #output = output.reshape(-1, 27)[:, 3:].reshape(-1)
    # output = output.reshape(-1, 57)[:, 3:].reshape(-1)
    output = torch.flatten(output)
    return output


class GatedGraphNN(nn.Module):
    """ Gated Graph Sequential Neural Network Architecture"""
    def __init__(self, in_dim, out_dim):
        
        super(GatedGraphNN, self).__init__()
       
        self.layer1 = GatedGraphConv(in_dim, 64)
        # self.layer2 = GatedGraphConv(64, 64)
        self.layer3 = nn.Linear(in_dim, out_dim)
        

    def forward(self, obs, edge_index):
        # convert observation tensor to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif isinstance(obs, list):
            obs = torch.tensor(np.array(obs), dtype=torch.float)

        h1 = F.relu(self.layer1(obs, edge_index))
        # h2 = F.relu(self.layer2(h1, edge_index))
        
        output = self.layer3(h1)
        
        return output[-1]

class RecurrentGCN_new(nn.Module):
    """ Recurrent Graph Convolutional Network Architecture"""
    def __init__(self, in_dim, out_dim):
        
        super(RecurrentGCN_new, self).__init__()
       
        self.recurrent = GCLSTM(in_dim, 32, 3)
        self.linear = nn.Linear(32, 1)
        

    def forward(self, obs, edge_index):
        # convert observation tensor to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif isinstance(obs, list):
            obs = torch.tensor(np.array(obs), dtype=torch.float)

        h = F.relu(self.recurrent(obs, edge_index)[0])
        
        output = self.linear(h)

        output = output.reshape(-1,11)[:,1:].reshape(-1)

        output = torch.flatten(output)

        return output


class RecurrentTGCN(nn.Module):
    """ Recurrent Graph Convolutional Network Architecture"""

    def __init__(self, in_dim, out_dim):

        super(RecurrentTGCN, self).__init__()

        self.recurrent1 = TGCN(in_dim, 32)
        self.recurrent2 = TGCN(32, 32)
        self.linear = nn.Linear(32, 1)

    def forward(self, obs, edge_index, nodes_num):
        # convert observation tensor to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif isinstance(obs, list):
            obs = torch.tensor(np.array(obs), dtype=torch.float)

        h = F.relu(self.recurrent1(obs, edge_index))
        h = F.relu(self.recurrent2(h, edge_index))
        output = self.linear(h)
        output = output.reshape(-1, nodes_num)[:, 1:].reshape(-1)

        output = torch.flatten(output)

        return output

class AttentionSTGCN(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(AttentionSTGCN, self).__init__()
    
    self.attention1 = STConv(11, in_dim, 32, 2, 1, 3)
    self.attention2 = STConv(11, in_dim, 32, 2, 1, 3)
    self.linear = nn.Linear(2, 1)

  def forward(self, obs, edge_index):
    if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
    elif isinstance(obs, list):
        obs = torch.tensor(np.array(obs), dtype=torch.float)
    obs = torch.unsqueeze(obs, 0)
    obs = torch.unsqueeze(obs, 0)
    print(obs.shape)
    h = self.attention1(obs, edge_index)
    h = F.relu(h)
    h = self.attention2(h, edge_index)
    h = F.relu(h)
    output = self.linear(h)
    #output = output.reshape(-1,15)[:,3:].reshape(-1)
    #output = output.reshape(-1,23)[:,1:].reshape(-1)
    output = output.reshape(-1, 11)[:, 1:].reshape(-1)
    output = torch.flatten(output)
    return output


class GCN_test(nn.Module):
    """ Graph Convolutional Network Architecture"""

    def __init__(self, in_dim, out_dim):

        super(GCN_test, self).__init__()

        self.layer1 = GCNConv(in_dim, 64)
        self.layer2 = GCNConv(64, out_dim)
        #self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs, edge_index):
        # convert observation tensor to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif isinstance(obs, list):
            obs = torch.tensor(np.array(obs), dtype=torch.float)

        h1 = F.relu(self.layer1(obs, edge_index))
        output = F.relu(self.layer2(h1, edge_index))
        output = output.reshape(-1, 11)[:, -1] #.reshape(-1)
        #output = self.layer3(h2)
        #print(output.reshape(-1).shape)
        return output
