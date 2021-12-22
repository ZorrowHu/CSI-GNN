import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import scipy.sparse as sp

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)

        del input_in
        del input_out

        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)

        del gi
        del gh

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n) 

        del i_r
        del i_i
        del i_n
        del h_r
        del h_i
        del h_n

        hy = newgate + inputgate * (hidden - newgate)

        del resetgate
        del inputgate
        del newgate
        
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid

        self.sgc_embed = opt.sgc_embed
        self.gnn_embed = opt.gnn_embed

        self.alpha = opt.alpha
        self.beta = opt.beta

        self.a = opt.a
        self.b = opt.b

        self.degree = opt.degree
        self.normalize = opt.normalize

        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)                         #在这里用了GNN
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  #target attention
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # (b,s,1)

        # alpha = torch.sigmoid(alpha) # B,S,1
        alpha = F.softmax(alpha, 1) # B,S,1
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)  # (b,d)

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        a = a.view(ht.shape[0], 1, ht.shape[1])  # b,1,d

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        # target attention: sigmoid(hidden M b)
        # mask  # batch_size x seq_length
        hidden = hidden * mask.view(mask.shape[0], -1, 1).float()  # batch_size x seq_length x latent_size
        qt = self.linear_t(hidden)  # batch_size x seq_length x latent_size
        # beta = torch.sigmoid(b @ qt.transpose(1,2))  # batch_size x n_nodes x seq_length
        beta = F.softmax(b @ qt.transpose(1,2), -1)  # batch_size x n_nodes x seq_length

        target = beta @ hidden  # batch_size x n_nodes x latent_size

        a = a + target  # b,n,d

        scores = torch.sum(a * b, -1)  # b,n
        # scores = torch.matmul(a, b.transpose(1, 0))

        return scores

    def forward(self, inputs, A, A_in_out):

        hidden = self.embedding(inputs)
        batch_size, item_num, hidden_size = hidden.shape
        
        alpha = self.alpha
        beta = self.beta
        degree = self.degree
        I = torch.eye(item_num).cuda()
            
        ###############################################################
        # 从这里开始SGC处理特征
        # 使用实际邻接矩阵A，这个方法比前两个都work
        ################################################################
        if self.normalize == False:    
            #Version 1.0 没有对特征矩阵A进行归一化的操作
            #Version 1.1 在暴力相乘的基础上引入alpha和degree进行控制
            for i in range(len(A)):
                #A[i] = alpha * norm_adj + (1 - alpha) * torch.eye(item_num).cuda()
                #下面这行效果比较好，估计是因为考虑了重复点击的缘故
                if self.sgc_embed == 1:
                    A[i] = alpha * A[i] + beta * (1 - A[i]) * I
                elif self.sgc_embed == 2:
                    A[i] = alpha * A[i] + (1 - alpha) * I + beta * (1 - A[i]) * I
                elif self.sgc_embed == 3:
                    A[i] = alpha * (A[i] + (1 - A[i]) * I) + (1 - alpha) * I

        if self.normalize == True:
            #Version 2.0 对 特征矩阵A进行归一化的操作 A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2 
            for i in range(len(A)):

                # 0代表了只做attention不做sgc
                if self.sgc_embed == 0:
                    A[i] = alpha * (A[i] + (1 - A[i]) * I) + beta * I
                # 进入sgc
                else:
                    adj = A[i]
                    adj = adj + I
                    row_sum = adj.sum(1)
                    d_inv_sqrt = np.power(row_sum.tolist(), -0.5).flatten()
                    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                    d_mat_inv_sqrt = sp.diags(d_inv_sqrt).toarray()
                    d_mat_inv_sqrt = torch.Tensor(d_mat_inv_sqrt).float()
                    d_mat_inv_sqrt = trans_to_cuda(d_mat_inv_sqrt)
                    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

                    del adj
                    del row_sum
                    del d_inv_sqrt
                    del d_mat_inv_sqrt

                    if self.sgc_embed == 1:        #基础操作
                        A[i] = norm_adj
                    elif self.sgc_embed == 2:      #α (1-α) 的话即随机游走策略
                        A[i] = alpha * norm_adj + beta * I
                    elif self.sgc_embed == 3:      #带权重自环
                        A[i] = alpha * norm_adj + beta * (1 - norm_adj) * I
                    elif self.sgc_embed == 4:      #带单位权重自环的注意力机制
                        A[i] = alpha * (norm_adj + (1 - norm_adj) * I) + beta * I
        
        h0 = hidden

        '''
            SGC
        '''
        for i in range(degree):
            sgc_hidden = torch.bmm(A, hidden)
        hidden = sgc_hidden

        if self.gnn_embed == 1:
            #print('Using GNN Embedding only')
            hidden = self.gnn(A_in_out, h0)
        elif self.gnn_embed == 2:
            #print('Using GNN Embedding with random input')
            hidden = self.gnn(A_in_out, h0)
            hidden = self.a * sgc_hidden + self.b * hidden
        elif self.gnn_embed == 3:
            #print('Using GNN Embedding with SGC as input')
            hidden = self.gnn(A_in_out, sgc_hidden)
            hidden = self.a * sgc_hidden + self.b * hidden
        

        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, A_in_out, items, mask, targets = data.get_slice(i)

    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    A_in_out = trans_to_cuda(torch.Tensor(A_in_out).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    hidden = model(items, A, A_in_out)

    del items
    del A

    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    
    del alias_inputs
    
    return targets, model.module.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    '''
    train_data.inputs 这里就是输入的n条session序列，其中session序列的最大长度为Max，整个train_data的大小为n × Max
    '''
    #print(train_data.inputs.shape)
    #print(train_data.inputs)

    print('start training: ', datetime.datetime.now())
    model.module.train()
    total_loss = 0.0

    slices = train_data.generate_batch(model.module.batch_size)
    #slices根据batch_size把n条数据（sample中就是1205）划分成若干个batch_size，最后一个batch大小小于等于batch_size
    #print(slices)

    for i, j in zip(slices, np.arange(len(slices))):
        #print('Slices in this circulation is:')
        #print(i)
        model.module.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.module.loss_function(scores, targets - 1)

        del targets
        del scores

        loss.backward()
        model.module.optimizer.step()
        total_loss += float(loss.item())
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    model.module.scheduler.step()
    
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.module.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
