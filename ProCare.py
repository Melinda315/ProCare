
# coding: utf-8

# ## **Preparation**

# In[ ]:


import math
import pickle as pickle
import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter
from torch_geometric.utils import softmax
from torchdiffeq import odeint_adjoint as odeint
import random
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# In[ ]:


with open('./binary_train_codes_x.pkl', 'rb') as f0:
  binary_train_codes_x = pickle.load(f0)

with open('./binary_test_codes_x.pkl', 'rb') as f1:
  binary_test_codes_x = pickle.load(f1)

train_codes_y = np.load('./train_codes_y.npy')
train_visit_lens = np.load('./train_visit_lens.npy')
test_codes_y = np.load('./test_codes_y.npy')
test_visit_lens = np.load('./test_visit_lens.npy')
code_levels = np.load('./code_levels.npy')

train_pids = np.load('./train_pids.npy')
test_pids = np.load('./test_pids.npy')
with open('./patient_time_duration_encoded.pkl', 'rb') as f80:
  patient_time_duration_encoded = pickle.load(f80)

# In[ ]:


# Prepare padded hypergraphs for batching
def transform_and_pad_input(x):
  tempX = []
  for ele in x:
    tempX.append(torch.tensor(ele).to(torch.float32))
  x_padded = pad_sequence(tempX, batch_first=True, padding_value=0)
  return x_padded

trans_y_train = torch.tensor(train_codes_y)
trans_y_test = torch.tensor(test_codes_y)
padded_X_train = torch.transpose(transform_and_pad_input(binary_train_codes_x), 1, 2)
padded_X_test = torch.transpose(transform_and_pad_input(binary_test_codes_x), 1, 2)
class_num = train_codes_y.shape[1]

total_pids = list(train_pids) + list(test_pids)
cur_max = 0
for pid in total_pids:
  duration = patient_time_duration_encoded[pid]
  ts = [sum(duration[0:gap+1]) for gap in range(len(duration))]
  if cur_max < max(ts):
    cur_max = max(ts)


# ## **Model**

# ### **Dataloader and Temporal Encoding**

# In[ ]:


def prepare_temporal_encoding(H, pid, duration_dict, Dv):
  TE = []
  X_i_idx = torch.unique(torch.nonzero(H, as_tuple=True)[0])
  H_i = H[X_i_idx, :]
  for code in X_i_idx:
    TE_code = torch.zeros(Dv * 2)
    visits = torch.nonzero(H[code.item()]).tolist()
    temp = duration_dict[pid][:-1]
    code_duration = [sum(temp[0:gap+1]) for gap in range(len(temp))]
    visits.append([len(code_duration) - 1])
    pre_delta = [code_duration[visits[j][0]] - code_duration[visits[j - 1][0]] for j in range(1, len(visits))]
    delta = sum(pre_delta) / len(pre_delta)
    T_m = sum(code_duration)
    if T_m == 0:
      T_m += 1
    for k in range(len(TE_code)):
      if k < Dv:
        TE_code[k] = math.sin((k * delta) / (T_m * Dv))
      else:
        TE_code[k] = math.cos(((k - Dv) * delta) / (T_m * Dv))
    TE.append(TE_code)
  TE_i = torch.stack(TE)
  return TE_i


# In[ ]:


def load_prepared_TE(pids, te_dict):
  TE_list = []
  for pid in pids:
    one_patient = []
    patient_dict = te_dict[pid]
    for i, (k, v) in enumerate(patient_dict.items()):
      one_patient.append(torch.tensor(v))
    TE_list.append(torch.stack(one_patient))
  return TE_list


# In[ ]:


class ProCare_Dataset(data.Dataset):
    def __init__(self, hyperG, data_label, pid, duration_dict, data_len, te_location, Dv):
        self.hyperG = hyperG
        self.data_label = data_label
        self.pid = pid
        self.data_len = data_len
        if te_location == None:
          TE_list = [prepare_temporal_encoding(hyperG[j], pid[j], duration_dict, Dv) for j in range(len(hyperG))]
          self.TE = pad_sequence(TE_list, batch_first=True, padding_value=0)
        else:
          with open(te_location, 'rb') as f250:
            TE_dict = pickle.load(f250)
          self.TE = pad_sequence(load_prepared_TE(pid, TE_dict), batch_first=True, padding_value=0)
 
    def __len__(self):
        return len(self.hyperG)
 
    def __getitem__(self, idx):
        return self.hyperG[idx], self.data_label[idx], self.pid[idx], self.TE[idx], self.data_len[idx]


# ### **Hierarchical Embedding for Medical Codes**

# In[ ]:


def glorot(tensor):
  if tensor is not None:
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    tensor.data.uniform_(-stdv, stdv)


# In[ ]:


class HierarchicalEmbedding(nn.Module):
    def __init__(self, code_levels, code_num_in_levels, code_dims):
        super(HierarchicalEmbedding, self).__init__()
        self.level_num = len(code_num_in_levels)
        self.code_levels = code_levels
        self.level_embeddings = nn.ModuleList([nn.Embedding(code_num, code_dim) for level, (code_num, code_dim) in enumerate(zip(code_num_in_levels, code_dims))])

    def forward(self, input=None):
        embeddings = [self.level_embeddings[level](self.code_levels[:, level] - 1) for level in range(self.level_num)]
        embeddings = torch.cat(embeddings, dim=1)
        return embeddings # return: (code_num, one_code_dim * 4)


# ### **Hypergraph Modelling**

# In[ ]:


class UniGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout=0., negative_slope=0.2):
        super(UniGATConv, self).__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_v = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop  = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)

    def reset_parameters(self):
        glorot(self.att_v)
        glorot(self.att_e)

    def forward(self, X, vertex, edges):
        H, C, N = self.heads, self.out_channels, X.shape[0]
        X0 = self.W(X)
        X = X0.view(N, H, C)
        Xve = X[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce='mean')
        alpha_e = (Xe * self.att_e).sum(-1)
        a_ev = alpha_e[edges]
        alpha = a_ev
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, vertex, num_nodes=N)
        alpha = self.attn_drop(alpha)
        alpha = alpha.unsqueeze(-1)
        Xev = Xe[edges]
        Xev = Xev * alpha 
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)
        X = Xv.view(N, H * C)
        return X + X0


# In[ ]:


class UniGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, nhead):
        super(UniGNN, self).__init__()
        self.conv_out = UniGATConv(nhid * nhead, nclass, heads=1, dropout=0.)
        self.convs = nn.ModuleList(
            [UniGATConv(nfeat, nhid, heads=nhead, dropout=0.)] +
            [UniGATConv(nhid * nhead, nhid, heads=nhead, dropout=0.) for _ in range(nlayer - 2)]
        )
        self.act = nn.ReLU()
        self.input_drop = nn.Dropout(0.)
        self.dropout = nn.Dropout(0.)

    def forward(self, X, V, E):
        X = self.input_drop(X)
        for conv in self.convs:
            X = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)
        X = self.conv_out(X, V, E)      
        return F.log_softmax(X, dim=1)


# ### **Encoder and Decoder**

# In[ ]:


class Encoder(nn.Module):
  def __init__(self, code_levels, single_dim, visit_dim, hdim, Dv, alpha_dim, gnn_dim, gnn_layer, nhead, personal_gate, hyperG_gate, PGatt_gate):
    super(Encoder, self).__init__()
    # Hierarchical embedding for medical codes
    code_num_in_levels = (np.max(code_levels, axis=0)).tolist()
    code_levels = torch.from_numpy(code_levels).to(device)
    code_dims = [single_dim] * code_levels.shape[1]
    self.hier_embed_layer = HierarchicalEmbedding(code_levels, code_num_in_levels, code_dims)
    # Visit representation learning
    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.Softmax()
    self.pgate = personal_gate
    self.Dv = Dv
    if self.pgate:
      self.W_t = nn.Linear(sum(code_dims) + Dv * 2, sum(code_dims), bias=True)
      self.agate = PGatt_gate
      if self.agate:
        self.W_F = nn.Linear(sum(code_dims), alpha_dim, bias=False)
        self.z = nn.Linear(alpha_dim, 1, bias=False)
    self.hgate = hyperG_gate
    if not self.hgate:
      self.map_visit = nn.Linear(sum(code_dims), visit_dim, bias=True)
    else:
      self.unignn = UniGNN(sum(code_dims), gnn_dim, visit_dim, gnn_layer, nhead)
    # Aggregate visit embeddings sequentially with attention
    self.temporal_edge_aggregator = nn.GRU(visit_dim, hdim, 1, batch_first=True)
    self.attention_context = nn.Linear(hdim, 1, bias=False)

  def forward(self, H, TE):
    code_idx = torch.unique(torch.nonzero(H, as_tuple=True)[0])
    X_G = self.hier_embed_layer(None)
    personal_TE = torch.zeros(H.shape[0], self.Dv * 2).to(device)
    personal_TE[code_idx.tolist(), :] = TE[:len(code_idx), :]
    if self.pgate:
      X_0 = self.sigmoid(self.W_t(torch.cat((X_G, personal_TE), 1)))
      if self.hgate:
        V = torch.nonzero(H)[:, 0]
        E = torch.nonzero(H)[:, 1]
        X_P = self.unignn(X_0, V, E)
      else:
        X_P = self.map_visit(X_0)
      if self.agate:
        nom = math.e ** (torch.squeeze(self.z(self.sigmoid(self.W_F(X_P))), 1))
        den = nom + math.e ** (torch.squeeze(self.z(self.sigmoid(self.W_F(X_G))), 1))
        alpha0 = nom / den
        X = torch.matmul(torch.diag(alpha0), X_P) + torch.matmul(torch.diag(1 - alpha0), X_G)
      else:
        X = 0.8*X_P + X_G
    else:
      if self.hgate:
        V = torch.nonzero(H)[:, 0]
        E = torch.nonzero(H)[:, 1]
        X = self.unignn(X_G, V, E)
      else:
        X = X_G
    visit_emb = torch.matmul(H.T.to(torch.float32), X)
    hidden_states, _ = self.temporal_edge_aggregator(visit_emb)
    alpha1 = self.softmax(torch.squeeze(self.attention_context(hidden_states), 1))
    h = torch.sum(torch.matmul(torch.diag(alpha1), hidden_states), 0)
    return h


# In[ ]:


class GRUODECell_Autonomous(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.bias        = bias

        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)


    def forward(self, t, h):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        The step size is given by delta_t.
        Args:
            t        time
            h        hidden state (current)
        Returns:
            Updated h
        """
        x = torch.zeros_like(h)
        z = torch.sigmoid(x + self.lin_hz(h))
        n = torch.tanh(x + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh

class ODEFunc(nn.Module):
  def __init__(self, hdim, ode_hid):
    super().__init__()
    self.func = nn.Sequential(nn.Linear(hdim, ode_hid),
                              nn.Tanh(),
                              nn.Linear(ode_hid, hdim))
    for m in self.func.modules():
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1)
        nn.init.constant_(m.bias, val=0)

  def forward(self, t, y):
    output = self.func(y)
    return output


# In[ ]:


class ODE_VAE_Decoder(nn.Module):
  def __init__(self, hdim, dist_dim, ode_hid, nclass, ODE_Func):
    super(ODE_VAE_Decoder, self).__init__()
    self.fc_mu = nn.Linear(hdim, dist_dim)
    self.fc_var = nn.Linear(hdim, dist_dim)
    self.map_back = nn.Linear(dist_dim, hdim)
    self.relu = nn.ReLU()
    self.odefunc = ODE_Func
    self.final_layer = nn.Linear(hdim, nclass)
    self.softmax = nn.Softmax(dim=-1)

  def reparameterize(self, mu, log_var):
    std = torch.exp(0.5 * log_var)
    q = torch.distributions.Normal(mu, std)
    return q.rsample()

  def forward(self, h, timestamps):
    mu = self.fc_mu(h)
    log_var = self.fc_var(h)
    z = self.map_back(self.reparameterize(mu, log_var))
    pred_z = odeint(func = self.odefunc, y0 = z, t = timestamps, method = 'rk4', options=dict(step_size=0.1))
    output = self.softmax(self.final_layer(pred_z))
    return output, mu, log_var


# In[ ]:


class MLP_Decoder(nn.Module):
  def __init__(self, hdim, nclass):
    super(MLP_Decoder, self).__init__()
    self.final_layer = nn.Linear(hdim, nclass)
    self.softmax = nn.Softmax()

  def forward(self, h, timestamps):
    output = self.softmax(self.final_layer(h))
    return output, 0, 0


# ### **ProCare**

# In[ ]:


class ProCare_VAE(nn.Module):
  def __init__(self, code_levels, single_dim, visit_dim, hdim, Dv, alpha_dim, gnn_dim, gnn_layer, nhead, nclass, dist_dim, ode_hid, personal_gate, hyperG_gate, PGatt_gate, ODE_gate):
    super(ProCare_VAE, self).__init__()
    self.encoder = Encoder(code_levels, single_dim, visit_dim, hdim, Dv, alpha_dim, gnn_dim, gnn_layer, nhead, personal_gate, hyperG_gate, PGatt_gate)
    self.ogate = ODE_gate
    if not ODE_gate:
      self.decoder = MLP_Decoder(hdim, nclass)
    else:
      #self.ODE_Func = ODEFunc(hdim, ode_hid)
      self.ODE_Func = GRUODECell_Autonomous(hdim)
      self.decoder = ODE_VAE_Decoder(hdim, dist_dim, ode_hid, nclass, self.ODE_Func)

  def forward(self, Hs, TEs, timestamps, seq_lens):
    h = torch.stack([self.encoder(Hs[ii][:, 0:int(seq_lens[ii])], TEs[ii]) for ii in range(len(Hs))])
    pred, mu, log_var = self.decoder(h, timestamps)
    if self.ogate:
      pred = torch.swapaxes(pred, 0, 1)
    return pred, mu, log_var


# ## **Training**

# ### **Evaluation Utils Functions**

# In[ ]:


def IDCG(ground_truth, topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    idcg = 0
    for i in range(topn):
        idcg += ((2 ** t[i]) - 1) / math.log(i + 2, 2)
    return idcg


def nDCG(ranked_list, ground_truth, topn):
    dcg = 0
    idcg = IDCG(ground_truth, topn)
    for i in range(topn):
        idx = ranked_list[i]
        dcg += ((2 ** ground_truth[idx]) - 1)/ math.log(i + 2, 2)
    return dcg / idcg


# In[ ]:


def evaluate_model(pred, label, k1, k2, k3, k4, k5, k6):
    # Below is for nDCG
    ks = [k1, k2, k3, k4, k5, k6]
    y_pred = np.array(pred.cpu().detach().tolist())
    y_true_hot = np.array(label.cpu().detach().tolist())
    ndcg = np.zeros((len(ks), ))
    for i, topn in enumerate(ks):
        for pred2, true_hot in zip(y_pred, y_true_hot):
            ranked_list = np.flip(np.argsort(pred2))
            ndcg[i] += nDCG(ranked_list, true_hot, topn)
    n_list = ndcg / len(y_true_hot)
    metric_n_1 = n_list[0]; metric_n_2 = n_list[1]; metric_n_3 = n_list[2]; metric_n_4 = n_list[3]; metric_n_5 = n_list[4]; metric_n_6 = n_list[5]
    # Below is for precision and recall
    a = np.zeros((len(ks), )); r = np.zeros((len(ks), ))
    for pred2, true_hot in zip(y_pred, y_true_hot):
        pred2 = np.flip(np.argsort(pred2))
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred2[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            r[i] += len(it) / len(t)
    p_list = a / len(y_true_hot); r_list = r / len(y_true_hot)
    metric_p_1 = p_list[0]; metric_p_2 = p_list[1]; metric_p_3 = p_list[2]; metric_p_4 = p_list[3]; metric_p_5 = p_list[4]; metric_p_6 = p_list[5]
    metric_r_1 = r_list[0]; metric_r_2 = r_list[1]; metric_r_3 = r_list[2]; metric_r_4 = r_list[3]; metric_r_5 = r_list[4]; metric_r_6 = r_list[5]
    return metric_p_1, metric_r_1, metric_n_1, metric_p_2, metric_r_2, metric_n_2, metric_p_3, metric_r_3, metric_n_3, metric_p_4, metric_r_4, metric_n_4, metric_p_5, metric_r_5, metric_n_5, metric_p_6, metric_r_6, metric_n_6


# ### **Loss Function**

# In[ ]:


def ProCare_loss(pred, truth, past, pids, mu, log_var, duration_dict, timestamps, ode_gate, balance, cur_max):
  criterion = nn.BCELoss()
  if not ode_gate:
    loss = criterion(pred, truth)
  else:
    reconstruct_loss = 0
    last_visits = []
    for i, traj in enumerate(pred):
      duration = duration_dict[pids[i].item()]
      temp = [sum(duration[0:gap+1]) for gap in range(len(duration))]
      ts = [stamp / cur_max for stamp in temp]
      idx = [(timestamps == m).nonzero(as_tuple=True)[0].item() for m in ts]
      visit_lens = len(ts)
      last_visits.append(traj[idx[-1], :])
      reconstruct_loss += criterion(traj[idx[:-1], :], torch.swapaxes(past[i][:, 0:(visit_lens-1)], 0, 1))
    ELBO=torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))
    reconstruct_loss=(reconstruct_loss / len(pred))
    elbo=reconstruct_loss/ELBO
    ELBO = elbo*ELBO + reconstruct_loss
    #print("ELBO:",ELBO)
    last_visits = torch.stack(last_visits)
    pred_loss = criterion(last_visits, truth)
    loss = balance * pred_loss + (1 - balance) * ELBO
  return loss


# ### **Main Loop**

# In[ ]:


def train(model, lrate, num_epoch, train_loader, test_loader, model_directory, ode_gate, duration_dict, early_stop_range, balance, cur_max):
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr = lrate)
  best_metric_r1=0
  test_loss_per_epoch = []; train_average_loss_per_epoch = []
  p1_list = []; p2_list = []; p3_list = []; p4_list = []; p5_list = []; p6_list = []
  r1_list = []; r2_list = []; r3_list = []; r4_list = []; r5_list = []; r6_list = []
  n1_list = []; n2_list = []; n3_list = []; n4_list = []; n5_list = []; n6_list = []
  for epoch in range(num_epoch):
    one_epoch_train_loss = []
    for i, (hyperGs, labels, pids, TEs, seq_lens) in enumerate(train_loader):
      hyperGs = hyperGs.to(device); labels = labels.to(device); TEs = TEs.to(device)
      hyperGs_vae=[]
      labels_vae=[]
      TEs_vae=[]
      pids_vae=[]
      seq_lens_vae=[]
      for patient_num in range(len(labels)):
        if seq_lens[patient_num]>1:
          hyperGs_vae.append(hyperGs[patient_num])
          labels_vae.append(labels[patient_num])
          TEs_vae.append(TEs[patient_num])
          pids_vae.append(pids[patient_num])
          seq_lens_vae.append(seq_lens[patient_num])
        
      hyperGs_vae=torch.stack(hyperGs_vae).to(device)
      labels_vae=torch.stack(labels_vae).to(device)
      TEs_vae=torch.stack(TEs_vae).to(device)
      timestamps = []
      for pid in pids_vae:
        duration = duration_dict[pid.item()]
        timestamps += [sum(duration[0:gap+1]) for gap in range(len(duration))]
      temp = [stamp / cur_max for stamp in list(set(timestamps))]
      timestamps = torch.tensor(temp).to(torch.float32).sort()[0]
      pred, mu, log_var = model(hyperGs_vae, TEs_vae, timestamps, seq_lens_vae)
      loss = ProCare_loss(pred, labels_vae.to(torch.float32), hyperGs_vae,pids_vae, mu, log_var, duration_dict, timestamps, ode_gate, balance, cur_max)
      one_epoch_train_loss.append(loss.item())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    train_average_loss_per_epoch.append(sum(one_epoch_train_loss) / len(one_epoch_train_loss))
    print('Epoch: [{}/{}], Average Loss: {}'.format(epoch+1, num_epoch, round(train_average_loss_per_epoch[-1], 9)))
    model.eval()
    one_epoch_test_loss = []
    test_data_len = 0
    pred_list = []
    truth_list = []
    for (hyperGs, labels, pids, TEs, seq_lens) in test_loader:
      hyperGs = hyperGs.to(device); labels = labels.to(device); TEs = TEs.to(device)
      hyperGs_vae=[]
      labels_vae=[]
      TEs_vae=[]
      pids_vae=[]
      seq_lens_vae=[]
      for patient_num in range(len(labels)):
        if seq_lens[patient_num]>1:
          hyperGs_vae.append(hyperGs[patient_num])
          labels_vae.append(labels[patient_num])
          TEs_vae.append(TEs[patient_num])
          pids_vae.append(pids[patient_num])
          seq_lens_vae.append(seq_lens[patient_num])
      hyperGs_vae=torch.stack(hyperGs_vae).to(device)
      labels_vae=torch.stack(labels_vae).to(device)
      TEs_vae=torch.stack(TEs_vae).to(device)
      with torch.no_grad():
        timestamps = []
        for pid in pids_vae:
          duration = duration_dict[pid.item()]
          timestamps += [sum(duration[0:gap+1]) for gap in range(len(duration))]
        temp = [stamp / cur_max for stamp in list(set(timestamps))]
        timestamps = torch.tensor(temp).to(torch.float32).sort()[0]
        pred, mu, log_var = model(hyperGs_vae, TEs_vae, timestamps, seq_lens_vae)
        test_loss = ProCare_loss(pred, labels_vae.to(torch.float32),hyperGs_vae, pids_vae, mu, log_var, duration_dict, timestamps, ode_gate, balance, cur_max)
      one_epoch_test_loss.append(test_loss.item() * len(pids))
      test_data_len += len(pids_vae)
      truth_list.append(labels_vae)
      if ode_gate:
        for jj, traj in enumerate(pred):
          duration = duration_dict[pids_vae[jj].item()]
          ts1 = [sum(duration[0:gap+1]) for gap in range(len(duration))]
          ts = [stamp / cur_max for stamp in ts1]
          idx = [(timestamps == m).nonzero(as_tuple=True)[0].item() for m in ts]
          pred_list.append(traj[idx[-1], :])
      else:
        pred_list.append(pred)
    pred = torch.vstack(pred_list)
    truth = torch.vstack(truth_list)
    metric_p1, metric_r1, metric_n1, metric_p2, metric_r2, metric_n2, metric_p3, metric_r3, metric_n3, metric_p4, metric_r4, metric_n4, metric_p5, metric_r5, metric_n5, metric_p6, metric_r6, metric_n6, = evaluate_model(pred, truth, 5, 10, 15, 20, 25, 30)
    p1_list.append(metric_p1); p2_list.append(metric_p2); p3_list.append(metric_p3); p4_list.append(metric_p4); p5_list.append(metric_p5); p6_list.append(metric_p6)
    r1_list.append(metric_r1); r2_list.append(metric_r2); r3_list.append(metric_r3); r4_list.append(metric_r4); r5_list.append(metric_r5); r6_list.append(metric_r6)
    n1_list.append(metric_n1); n2_list.append(metric_n2); n3_list.append(metric_n3); n4_list.append(metric_n4); n5_list.append(metric_n5); n6_list.append(metric_n6)
    test_loss_per_epoch.append(sum(one_epoch_test_loss) / test_data_len)
    if metric_r1 > best_metric_r1:
        best_metric_r1 = metric_r1
        best_index = len(r1_list)-1
        if epoch >= 20:
          torch.save(model.state_dict(), f'{model_directory}/no_Hyper_gru_allopen__epoch_{epoch+1}.pth')
    print(f'Test Epoch {epoch+1}: {round(test_loss_per_epoch[-1], 9)}; {round(metric_r1, 9)}; {round(metric_r4, 9)}; {round(metric_r6, 9)}; {round(metric_n1, 9)}; {round(metric_n4, 9)}; {round(metric_n6, 9)}')
    print("best:", "metric_r1_list:",r1_list[best_index],"metric_r2_list:", r2_list[best_index],"metric_n1_list:",n1_list[best_index],"metric_n2_list:",n2_list[best_index])
    #if epoch >= 80 and test_loss_per_epoch[-1] < min(test_loss_per_epoch[0:-1]):
      #torch.save(model.state_dict(), f'{model_directory}/procare_epoch_{epoch+1}.pth')
    #early_stop = (-1) * early_stop_range
    #last_loss = test_loss_per_epoch[early_stop:]
    #if epoch >= 80 and sorted(last_loss) == last_loss:
      #break
    model.train()
  return p1_list, p2_list, p3_list, p4_list, p5_list, p6_list, r1_list, r2_list, r3_list, r4_list, r5_list, r6_list, n1_list, n2_list, n3_list, n4_list, n5_list, n6_list, test_loss_per_epoch, train_average_loss_per_epoch


# ### **Start Training**

# In[ ]:


model = ProCare_VAE(code_levels, 32, 128, 256, 8, 64, 256, 2, 8, class_num, 64, 128, True,True, True, True).to(device)
dict_model=torch.load('./Procare_models/2visit_gru_allopen__epoch_76.pth')
#dict_model.pop('decoder.final_layer.weight')
#dict_model.pop('decoder.final_layer.bias')
model.load_state_dict(dict_model)
#model.load_state_dict(dict_model)
print(f'Number of parameters of this model: {sum(param.numel() for param in model.parameters())}')
te_directory = './sum_TE2.0.pkl'
training_data = ProCare_Dataset(padded_X_train, trans_y_train, train_pids, patient_time_duration_encoded, train_visit_lens, te_directory, 32)
train_loader = DataLoader(training_data, batch_size=128, shuffle=True)
test_data = ProCare_Dataset(padded_X_test, trans_y_test, test_pids, patient_time_duration_encoded, test_visit_lens, te_directory, 32)
test_loader = DataLoader(test_data, batch_size=int(len(test_data) / 7), shuffle=False)
model_directory = './Procare_models'
p1_list, p2_list, p3_list, p4_list, p5_list, p6_list, r1_list, r2_list, r3_list, r4_list, r5_list, r6_list, n1_list, n2_list, n3_list, n4_list, n5_list, n6_list, test_loss_per_epoch, train_average_loss_per_epoch = train(model, 0.0001, 500, train_loader, test_loader, model_directory, True, patient_time_duration_encoded, 10, 0.5, cur_max)


# In[ ]:


plt.plot(train_average_loss_per_epoch,'r',label="Train")
plt.plot(test_loss_per_epoch,'b',label="Test")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# ### **Trash -- No need to run this part**

# In[ ]:


# def train(model, lrate, num_epoch, train_loader, test_loader, model_directory, ode_gate, duration_dict, early_stop_range, balance, cur_max):
#   model.train()
#   optimizer = torch.optim.Adam(model.parameters(), lr = lrate)
#   test_loss_per_epoch = []; train_average_loss_per_epoch = []
#   p1_list = []; p2_list = []; p3_list = []; p4_list = []; p5_list = []; p6_list = []
#   r1_list = []; r2_list = []; r3_list = []; r4_list = []; r5_list = []; r6_list = []
#   n1_list = []; n2_list = []; n3_list = []; n4_list = []; n5_list = []; n6_list = []
#   for epoch in range(num_epoch):
#     one_epoch_train_loss = []
#     for i, (hyperGs, labels, pids, TEs, seq_lens) in enumerate(train_loader):
#       hyperGs = hyperGs.to(device); labels = labels.to(device); TEs = TEs.to(device)
#       timestamps = []
#       for pid in pids:
#         duration = duration_dict[pid.item()]
#         timestamps += [sum(duration[0:gap+1]) for gap in range(len(duration))]
#       timestamps1 = list(set(timestamps))
#       timestamps = [stamp / cur_max for stamp in timestamps1]
#       timestamps = torch.tensor(timestamps).to(torch.float32).sort()[0]
#       pred, mu, log_var = model(hyperGs, TEs, timestamps, seq_lens)
#       loss = ProCare_loss(pred, labels.to(torch.float32), hyperGs, pids, mu, log_var, duration_dict, timestamps, ode_gate, balance, cur_max)
#       one_epoch_train_loss.append(loss.item())
#       optimizer.zero_grad()
#       loss.backward()
#       optimizer.step()
#     train_average_loss_per_epoch.append(sum(one_epoch_train_loss) / len(one_epoch_train_loss))
#     print('Epoch: [{}/{}], Average Loss: {}'.format(epoch+1, num_epoch, train_average_loss_per_epoch[-1]))
#     model.eval()
#     for (hyperGs, labels, pids, TEs, seq_lens) in test_loader:
#       hyperGs = hyperGs.to(device); labels = labels.to(device); TEs = TEs.to(device)
#       with torch.no_grad():
#         timestamps = []
#         for pid in pids:
#           duration = duration_dict[pid.item()]
#           timestamps += [sum(duration[0:gap+1]) for gap in range(len(duration))]
#         timestamps1 = list(set(timestamps))
#         timestamps = [stamp / cur_max for stamp in timestamps1]
#         timestamps = torch.tensor(timestamps).to(torch.float32).sort()[0]
#         pred, mu, log_var = model(hyperGs, TEs, timestamps, seq_lens)
#         test_loss = ProCare_loss(pred, labels.to(torch.float32), hyperGs, pids, mu, log_var, duration_dict, timestamps, ode_gate, balance, cur_max)
#         test_loss_per_epoch.append(test_loss.item())
#       if ode_gate:
#         pred_list = []
#         for jj, traj in enumerate(pred):
#           duration = duration_dict[pids[jj].item()]
#           ts1 = [sum(duration[0:gap+1]) for gap in range(len(duration))]
#           ts = [stamp / cur_max for stamp in ts1]
#           idx = [(timestamps == m).nonzero(as_tuple=True)[0].item() for m in ts]
#           pred_list.append(traj[idx[-1], :])
#         pred = torch.stack(pred_list)
#       metric_p1, metric_r1, metric_n1, metric_p2, metric_r2, metric_n2, metric_p3, metric_r3, metric_n3, metric_p4, metric_r4, metric_n4, metric_p5, metric_r5, metric_n5, metric_p6, metric_r6, metric_n6, = evaluate_model(pred, labels, 5, 10, 15, 20, 25, 30)
#       p1_list.append(metric_p1); p2_list.append(metric_p2); p3_list.append(metric_p3); p4_list.append(metric_p4); p5_list.append(metric_p5); p6_list.append(metric_p6)
#       r1_list.append(metric_r1); r2_list.append(metric_r2); r3_list.append(metric_r3); r4_list.append(metric_r4); r5_list.append(metric_r5); r6_list.append(metric_r6)
#       n1_list.append(metric_n1); n2_list.append(metric_n2); n3_list.append(metric_n3); n4_list.append(metric_n4); n5_list.append(metric_n5); n6_list.append(metric_n6)
#       print(f'Test Epoch {epoch+1}: {test_loss_per_epoch[-1]}; {metric_r1}; {metric_r4}; {metric_r6}; {metric_n1}; {metric_n4}; {metric_n6}')
#     if epoch > 80 and test_loss_per_epoch[-1] < min(test_loss_per_epoch[0:-1]):
#       torch.save(model.state_dict(), f'{model_directory}/procare_epoch_{epoch+1}.pth')
#     ES = (-1) * early_stop_range
#     last_loss = test_loss_per_epoch[ES:]
#     if epoch > 80 and sorted(last_loss) == last_loss:
#       break
#     model.train()
#   return p1_list, p2_list, p3_list, p4_list, p5_list, p6_list, r1_list, r2_list, r3_list, r4_list, r5_list, r6_list, n1_list, n2_list, n3_list, n4_list, n5_list, n6_list, test_loss_per_epoch, train_average_loss_per_epoch

