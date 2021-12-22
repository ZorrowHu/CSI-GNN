import argparse
import pickle
import os
import sys
import time
import datetime
from utils import build_graph, Data, split_validation
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size') # 原始默认为100，但是gpu空间不够就先设置成50
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=20, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=2, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--sgc_embed', type=int, default=1, help='Type of SGC embedding')
parser.add_argument('--gnn_embed', type=int, default=1, help='Use GNN embedding')
parser.add_argument('--alpha', type=float, default=1, help='the hyper parameter to control the effect of ajadency matrix')
parser.add_argument('--beta', type=float, default=1, help='the hyper parameter to control the effect of ajadency matrix')
parser.add_argument('--degree', type=int, default=2, help='the hyper parameter to control the degree of matrix multiplication')
parser.add_argument('--normalize', action='store_true', default=False, help='Should ajadency matrix do this or not: A_hat = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2')
parser.add_argument('--truncate', action='store_true', default=False, help='Remove long session longer than 20')
parser.add_argument('--a', type=float, default=1, help='the hyper parameter to control the effect of GCNs')
parser.add_argument('--b', type=float, default=1, help='the hyper parameter to control the effect of GGNN')
parser.add_argument('--sgc_att', action='store_true', default=False, help='Applying attention to sgc')
parser.add_argument('--gnn_att', action='store_true', default=False, help='Applying attention to gnn')

opt = parser.parse_args()

#opt.dataset = 'demo'
#opt.batchSize = 100
#opt.normalize = True    #到底做不做 A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
#opt.sgc_embed = 4        #邻接矩阵A中要不要做(αA+(1-α)I)X
#opt.gnn_embed = True
#opt.alpha = 0.8            #邻接矩阵A的比例
#opt.beta = 0.8            #额外信息的比例
#opt.degree = 2           #SGC特征处理的度

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def main():
    if opt.truncate:
        train_data = pickle.load(open('../datasets/' + opt.dataset + '/train_shortonly.txt', 'rb'))
    else:
        train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        if opt.truncate:
            test_data = pickle.load(open('../datasets/' + opt.dataset + '/test_shortonly.txt', 'rb'))
        else:
            test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(opt, train_data, shuffle=True)
    test_data = Data(opt, test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310
    
    model = trans_to_cuda(SessionGraph(opt, n_node))
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if (hit - best_result[0]) > 0.0001:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if (mrr - best_result[1]) > 0.0001:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    #if hasattr(torch.cuda, 'empty_cache'):
    #    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1, 0'

    output_time = datetime.datetime.now().strftime('%d%H%M%S')


    sys.stdout = Logger('2021消融实验+Short_' + opt.dataset + '-' + str(opt.a) + '×' + str(opt.b) + '+' + 'a' + str(opt.alpha) + 'b' + str(opt.beta) + '_' + output_time + '.txt')
    print(opt)
    main()
