"""
@Project   : MvGCN
@Time      : 2021/10/4
@Author    : Zhihao Wu
@File      : model.py
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
from args import parameter_parser
from utils import tab_printer, get_evaluation_results, compute_renormalized_adj
from Dataloader import load_data, construct_laplacian,load_data_Isogram
import tqdm
import random
import scipy.sparse as ss
import warnings
import time
warnings.filterwarnings("ignore")
from tqdm import tqdm
import os
import argparse
import logging
#from torchvision import datasets, transforms
import torch.multiprocessing as mp
import struct
#from plotheatmap import visualize_node_trajectories

args=parameter_parser()
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size, device):
        super(FusionLayer, self).__init__()
        self.device=device
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 64, bias=False),
                nn.Tanh(),
                nn.Linear(64, 1, bias=False)
            ).to(device)

    def forward(self, emb_list):
        emb_list=[emb.to(self.device) for emb in emb_list]

        if self.fusion_type == "average":
            common_emb = sum(emb_list) / len(emb_list)
        elif self.fusion_type == "weight":
            weight = F.softmax(self.weight, dim=0)
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_ = torch.stack(emb_list, dim=1).to(self.device)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")

        #common_emb=(common_emb+common_emb.t())/2
        return common_emb

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim,output_dim)*2*init_range - init_range
    return nn.Parameter(initial)



class ENcoder(nn.Module):
    def __init__(self, in_size):
        super(ENcoder, self).__init__()
        self.in_size = in_size
        self.liner1 = nn.Linear(in_size, 1024)
        self.liner2 = nn.Linear(1024, 512)
        self.liner3 = nn.Linear(512, 1024)
        self.liner4 = nn.Linear(1024, in_size)
        self.en = nn.Sequential(self.liner1,
                                nn.ReLU(),
                                self.liner2,
                                nn.ReLU(),
                                self.liner3,
                                nn.ReLU(),
                                self.liner4,
                                nn.Softmax(dim=1))
    def forward(self, x):
        x = self.en(x)
        return x

class evaluator(nn.Module):
    def __init__(self, in_size):
        super(evaluator, self).__init__()
        self.dc1 = nn.Linear(in_size, 1024)
        self.dc2 = nn.Linear(1024, 512)
        self.dc3 = nn.Linear(512, 128)
        self.dc4 = nn.Linear(128, 1)
        self.sigmoid = nn.ReLU()
    def forward(self, A):
        A = F.relu(self.dc1(A))
        A = F.relu(self.dc2(A))
        A = F.relu(self.dc3(A))
        A = F.sigmoid(self.dc4(A))
        # A = F.sigmoid(A)
        return A

class Adjgenerator(nn.Module):
    def __init__(self, in_size):
        super(Adjgenerator, self).__init__()
        self.in_size=in_size
        self.encoder = ENcoder(in_size)
        self.evaluator=evaluator(in_size)
    def forward(self, I,adj_list):
        A = self.encoder(I)
        real = 0
        for adj in adj_list:
            real += self.evaluator(adj)
        fake=self.evaluator(A)
        return A,real/len(adj_list),fake

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ResBlock(nn.Module):#表示神经网络中的残差模块。用于缓解深层网络训练中的梯度消失问题
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)#用于对输入特征进行标准化，以加速模型收敛并稳定训练过程
        self.relu = nn.ReLU(inplace=True)#激活函数：用来引入非线性，inplace=True 表示在输入张量上进行操作，节省内存
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)#是一个 3x3 的卷积核函数，用于提取特征。卷积核大小为 3x3 是 ResNet 经典设计，步幅为 stride。卷积可以对输入特征进行缩放和转变。
        self.norm2 = norm(planes)#第二个批归一化层，用于标准化经过卷积后的特征。
        self.conv2 = conv3x3(planes, planes)#第二个 3x3 的卷积，用于进一步处理特征。输入和输出的通道数一致。

    def forward(self, x):
        shortcut = x#保存输入 x，后续用于跳跃连接。残差块的主要思想就是通过将输入直接加到输出上，保留了原始信息。

        out = self.relu(self.norm1(x))#对输入 x 进行批归一化，然后通过 ReLU 激活函数引入非线性。

        if self.downsample is not None:
            shortcut = self.downsample(out)#如果 downsample 不为 None，则表示需要对 shortcut 进行调整（例如，当输入和输出的通道数或空间维度不一致时），通过 downsample 使 shortcut 和主分支输出的形状一致。

        out = self.conv1(out)#进行第一次 3x3 卷积，将输入变换到中间层。
        out = self.norm2(out)#进行第二次批归一化。
        out = self.relu(out)#再次应用 ReLU 激活函数。
        out = self.conv2(out)#第二次 3x3 卷积。

        return out + shortcut#将卷积输出 out 与捷径连接 shortcut 相加，得到残差块的输出。这就是残差学习的核心，通过引入跳跃连接保留了原始的输入信息，使得网络层数增加时不会产生梯度消失的问题。


# class SequentialWithAdj(nn.Module):
#     def __init__(self, *args):
#         super(SequentialWithAdj, self).__init__()
#         self.layers = nn.ModuleList(args)
#
#     def forward(self, x, adj):
#         for layer in self.layers:
#             # 检查该层是否支持 adj 参数
#             if isinstance(layer, (GraphConvSparse, ODEBlockGNN)):
#                 x = layer(x, adj)  # 传递 adj
#             else:
#                 x = layer(x)
#         return x

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, device, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.device = device
        #self.weight = glorot_init(input_dim, output_dim).to(device)
        self.weight=nn.Parameter(torch.randn(input_dim,output_dim,device=device)*0.01)
        self.activation = activation
        #self.adj=adj

    def forward(self, x, adj):
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        if self.activation==None:
            return x
        else:
            return self.activation(x)

class ODEfuncGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, adj, device, alpha1, alpha2):
        super(ODEfuncGNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim=hidden_dim
        self.device = device

        self.adj = adj
        # self.num_views=len(input_dim)
        # self.mu_module=nn.ModuleList()
        # self.view_num = len(adj)
        # self.pai = nn.Parameter(torch.ones(self.view_num) / self.view_num, requires_grad=True)
        #self.hidden_dim=sum(input_dim) // len(input_dim)

        self.gc1 = GraphConvSparse(sum(input_dim), self.hidden_dim, self.device)
        self.gc2 = GraphConvSparse(self.hidden_dim, sum(input_dim), self.device)
        # self.encoder=ENcoder(sample_size)
        # self.evaluator=evaluator(sample_size)
        # #self.I = torch.eye(sample_size).to(device)

        # self.S = nn.Parameter(torch.randn_like(self.adjs[0]), requires_grad=True)
        # self.theta = nn.Parameter(torch.FloatTensor([-5]).repeat(self.adjs[0].shape[0], 1), requires_grad=True)
        self.relu = nn.ReLU()
        self.nfe = 0
        # 添加线性层调整残差形状
        self.alpha1 = nn.Parameter(torch.tensor(alpha1),requires_grad=False)
        self.alpha2 = nn.Parameter(torch.tensor(alpha2),requires_grad=False)
        # self.attn1 = nn.Parameter(torch.randn(64, device=device))
        # self.attn2 = nn.Parameter(torch.randn(sum(input_dim), device=device))
        self.residual_transform1 = nn.Linear(sum(input_dim), self.hidden_dim).to(device)
        self.residual_transform2 = nn.Linear(self.hidden_dim, sum(input_dim)).to(device)
        self.bn1=nn.BatchNorm1d(self.hidden_dim)
        self.bn2=nn.BatchNorm1d(sum(input_dim))

    def forward(self, t, x):
        # print("x=",X)
        # print("adj1=",adj1)
        # exp_sum_pai = 0
        # for i in range(self.view_num):
        #     exp_sum_pai += torch.exp(self.pai[i])
        #
        # weight = torch.zeros_like(self.pai)
        # for i in range(self.view_num):
        #     weight[i] = torch.exp(self.pai[i]) / exp_sum_pai
        #
        # adj = weight[0] * self.adjs[0]
        # for i in range(1, self.view_num):
        #     adj = adj + weight[i] * self.adjs[i]
        #
        # theta_sigmoid_tri = self.thred_proj(self.theta)
        #
        # S_add_ST = torch.sigmoid((self.S + self.S.t()) / 2)
        # adj_S = adj * torch.relu(S_add_ST - theta_sigmoid_tri)

        self.nfe += 1
        x_residual = self.residual_transform1(x)
        x = self.gc1(x,self.adj)
        #x = self.gc1(x, adj)
        x=self.relu(x)
        #x=F.dropout(x,p=0.3)
        x=self.bn1(x)
        x=self.alpha1 * torch.tanh(x) + (1 - self.alpha1) * torch.tanh(x_residual)

        x_residual=self.residual_transform2(x)
        x = self.gc2(x,self.adj)
        #x = self.gc2(x, self.adj)
        #x = self.gc2(x, self.adj)
        x=self.bn2(x)
        x=self.alpha2 * torch.tanh(x) + (1 - self.alpha2) * torch.tanh(x_residual)

        return x
    def thred_proj(self, theta):
        theta_sigmoid = torch.sigmoid(theta)
        theta_sigmoid_mat = theta_sigmoid.repeat(1, theta_sigmoid.shape[0])
        theta_sigmoid_triu = torch.triu(theta_sigmoid_mat)
        theta_sigmoid_diag = torch.diag(theta_sigmoid_triu.diag())
        theta_sigmoid_tri = theta_sigmoid_triu + theta_sigmoid_triu.t() - theta_sigmoid_diag
        return theta_sigmoid_tri

class DeepMvNMF(nn.Module):
    def __init__(self, input_dims, en_hidden_dims, num_views, device):
        super(DeepMvNMF, self).__init__()
        self.encoder = nn.ModuleList()
        self.mv_decoder = nn.ModuleList()
        self.device = device
        for i in range(len(en_hidden_dims)-1):
            self.encoder.append(nn.Linear(en_hidden_dims[i], en_hidden_dims[i+1]))
        for i in range(num_views):
            decoder = nn.ModuleList()
            de_hidden_dims = [input_dims[i]]
            for k in range(1, len(en_hidden_dims)):
                de_hidden_dims.insert(0, en_hidden_dims[k])
            # print(de_hidden_dims)
            for j in range(len(de_hidden_dims)-1):
                decoder.append(nn.Linear(de_hidden_dims[j], de_hidden_dims[j+1]))
            self.mv_decoder.append(decoder)
        print(self.mv_decoder)

    def forward(self, input):
        z = input
        for layer in self.encoder:
            z = F.relu(layer(z))
        x_hat_list = []
        for de in self.mv_decoder:
            x_hat = z
            for layer in de:
                x_hat = F.relu(layer(x_hat))
            x_hat_list.append(x_hat)
        return z, x_hat_list

class ODEBlockGNN(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlockGNN, self).__init__()
        self.odefunc = odefunc
        self.integration_time = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.dt = 0.2

    def forward(self, x, return_trajectory=False):
        # t0, t1 = torch.tensor([0.0]), self.integration_time
        # t_seq = torch.arange(t0.item(), t1.item(), self.dt).to(x.device)
        # t_seq = torch.cat((t_seq, t1.view(1).to(x.device)), dim=0)
        t_seq = torch.arange(0, self.integration_time.item() + self.dt, self.dt).to(x.device)
        x_residual=x
        #self.odefunc.adj=self.adj
        out = odeint(self.odefunc, x, t_seq, method='rk4')
        #visualize_node_trajectories(out.cpu().detach().numpy(), node_indices=[0, 5, 10], method="tsne")
        if return_trajectory:
            # out: [T, N, D]
            return out
        x=out[-1]+x_residual
        return x
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self,value):
        self.odefunc.nfe=value

# class ContrastiveGraph(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, adj, device):
#         super(ContrastiveGraph, self).__init__()
#         self.device = device
#         self.gcn = GraphConvSparse(input_dim, hidden_dim, adj, self.device)  # 图卷积层
#         self.adj=adj
#
#     def forward(self, x):
#         return self.gcn(x)
#
# def contrastive_loss(embeddings, adj, temperature=0.5):
#     # 计算正样本对和负样本对
#     positive_pairs = (adj > 0).float()
#     negative_pairs = (adj == 0).float()
#
#     sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
#
#     positive_sim = torch.exp(sim_matrix / temperature) * positive_pairs
#     negative_sim = torch.exp(sim_matrix / temperature) * negative_pairs
#
#     pos_sum = positive_sim.sum(dim=-1)
#     neg_sum = negative_sim.sum(dim=-1)
#
#     loss = -torch.log(pos_sum / (pos_sum + neg_sum)).mean()
#     return loss
#
# # 构造优化过程
# def optimize_adjacency(features, adj, input_dim, hidden_dim, epochs=100):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = ContrastiveGraph(input_dim, hidden_dim, adj, device).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#     features = torch.tensor(features, dtype=torch.float32, device=device)
#     adj = torch.tensor(adj, dtype=torch.float32, device=device)
#
#     with tqdm(total=100, desc="Pretraining2") as pbar:
#         for epoch in range(epochs):
#             embeddings = model(features)  # 学习节点表示
#             loss = contrastive_loss(embeddings, adj)  # 计算对比学习损失
#
#             optimizer.zero_grad()
#             loss.backward(retain_graph=True)
#             optimizer.step()
#             pbar.set_postfix({'Loss': '{:.6f}'.format(loss.item())})
#             pbar.update(1)
#
#             # 更新邻接矩阵
#             with torch.no_grad():
#                 similarity = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
#                 adj = (similarity > 0.5).float()  # 阈值法重新生成邻接矩阵
#
#     return adj.cpu().numpy()

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(lr, batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    #它在训练深度学习模型时动态调整学习率，以便在不同的训练阶段使用不同的学习率
    initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader, device):
    total_correct = 0
    #print(dataset_loader)
    for data in dataset_loader:
        # x = x.to(device)
        # y = one_hot(np.array(y.cpu().numpy()), 10)
        x = data.x.to(device)  # 获取节点特征
        y = data.y.to(device)  # 获取标签

        # x=x.reshape(128,48)

        y_np = y.cpu().numpy()  # 转换为 numpy 数组，确保兼容性
        y_one_hot = one_hot(y_np, 10)  # 生成 one-hot 标签

        #target_class = np.argmax(y, axis=1)
        target_class = np.argmax(y_one_hot, axis=1)
        print("x's shape", x.shape)
        # if x.dim()>2:
        #     x = x.view(-1, x.shape[-1])
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# if __name__ == '__main__':
#     args=parameter_parser()
#     save_direction = 'pretrain/'
#     direction_judge=save_direction + 'pretrain' + args.dataset + '_adj.npz'
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     random.seed(args.seed)
#
#     # device = 'cuda:0'
#     device='cpu'
#     feature_list, adj_list, labels, idx_labeled, idx_unlabeled = load_data(args, device)
#     num_classes = len(np.unique(labels))
#     labels = labels.to(device)
#     hidden_dims = [args.dim1, args.dim2]
#     num_view = len(feature_list)
#     loss_function1 = torch.nn.NLLLoss()
#     input_dims = []
#     for i in range(num_view):
#         input_dims.append(feature_list[i].shape[1])
#     sample_size = feature_list[0].shape[0]
#
#     mask = torch.zeros_like(adj_list[0]).bool().to(device)
#     for adj in adj_list:
#         mask = mask | adj.bool()
#
#     I = torch.eye(sample_size).to(device)
#     model = MvGCN(input_dims, num_classes, 0, hidden_dims, device).to(device)
#     optimizer2 = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
#     StepLR1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=200, gamma=0.5)
#     StepLR2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=500, gamma=0.5)
#     Best_Acc=0
#     Loss_list = []
#     ACC_list = []
#     F1_list = []
#     # with tqdm(total=args.num_epoch, desc="Training") as pbar:
#     for epoch in range(args.num_epoch):
#         result, adj,fakeData_result,realData_result = model(feature_list, I)
#         # result, adj= model(feature_list, I)
#         # print(adj)
#         # print(adj)
#         output = F.log_softmax(result, dim=1)
#         loss0=0
#         for i in range(len(adj_list)):
#             loss0 = loss0 + torch.norm(adj - adj_list[i])**2
#         loss0=loss0-torch.mean(torch.log(realData_result+0.001) + torch.log(1 - fakeData_result+0.001))
#         loss1 = loss_function1(output[idx_labeled], labels[idx_labeled])
#         optimizer1.zero_grad()
#         # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
#         loss0.requires_grad_(True)
#         loss0.backward(retain_graph=True)
#         # StepLR1.step()
#         optimizer2.zero_grad()
#         # loss=loss0*0.01+loss1
#         loss1.requires_grad_(True)
#         loss1.backward()
#         optimizer1.step()
#         optimizer2.step()
#         # StepLR2.step()
#         with torch.no_grad():
#             model.eval()
#             # output, _, _ = model(feature_list, lp_list, args.Lambda, args.ortho)
#             pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
#             ACC, P, R, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])
#             if ACC>Best_Acc:
#                 Best_Acc=ACC
#             # pbar.set_postfix({'Loss': '{:.6f}'.format((loss).item()),'ACC': '{:.2f}'.format(ACC * 100), 'F1': '{:.2f}'.format(F1 * 100),'Best acc': '{:.4f}'.format(Best_Acc*100)})
#             # pbar.update(1)
#             print({'epoch':'{}'.format(epoch+1),'Loss': '{:.6f}'.format(loss1.item()+loss0.item()),'ACC': '{:.2f}'.format(ACC * 100), 'F1': '{:.2f}'.format(F1 * 100),'Best acc': '{:.4f}'.format(Best_Acc*100)})
#             Loss_list.append(float(loss1.item()+loss0.item()))
#             ACC_list.append(ACC)
#             F1_list.append(F1)
    # if args.save_results:
    #     experiment_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #     results_direction = 'results/' + args.dataset + '_results.txt'
    #     fp = open(results_direction, "a+", encoding="utf-8")
    #     # fp = open("results_" + args.dataset_name + ".txt", "a+", encoding="utf-8")
    #     fp.write(format(experiment_time))
    #     fp.write("\ndataset_name: {}\n".format(args.dataset))
    #     fp.write("knn: {}  |  ".format(args.knns))
    #     fp.write("ratio: {}  |  ".format(args.ratio))
    #     fp.write("epochs: {}  |  ".format(args.num_epoch))
    #     fp.write("lr: {}  |  ".format(args.lr))
    #     fp.write("wd: {}\n".format(args.weight_decay))
    #     # fp.write("lambda: {}  |  ".format(args.Lambda))
    #     # fp.write("alpha: {}\n".format(args.alpha))
    #     # fp.write("layer: {}\n".format(str_layers))
    #     fp.write("ACC:  {:.4f} ".format(Best_Acc*100))
    #     # fp.write("F1 :  {:.2f}".format(np.mean(F1_list) * 100))
    #     fp.close()
    #
    # if args.save_all:
    #     if args.save_loss:
    #         fp2 = open("results/loss/" + str(args.dataset) + ".txt", "a+", encoding="utf-8")
    #         fp2.seek(0)
    #         fp2.truncate()
    #         for i in range(len(Loss_list)):
    #             fp2.write(str(Loss_list[i]) + '\n')
    #         fp2.close()
    #
    #     if args.save_ACC:
    #         fp3 = open("results/ACC/" + str(args.dataset) + ".txt", "a+", encoding="utf-8")
    #         fp3.seek(0)
    #         fp3.truncate()
    #         for i in range(len(ACC_list)):
    #             fp3.write(str(ACC_list[i]) + '\n')
    #         fp3.close()
    #
    #     if args.save_F1:
    #         fp4 = open("results/F1/" + str(args.dataset) + ".txt", "a+", encoding="utf-8")
    #         fp4.seek(0)
    #         fp4.truncate()
    #         for i in range(len(F1_list)):
    #             fp4.write(str(F1_list[i]) + '\n')
    #         fp4.close()