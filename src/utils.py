import networkx as nx
import numpy as np


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, opt, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph
        self.sgc_att = opt.sgc_att
        self.gnn_att = opt.gnn_att

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, A_in_out, alias_inputs = [], [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            # 这里是每一行原始数据
            #print(u_input)
            
            ###########################################
            # 原始数据中同一序列中的重复点击视为一个点击
            node = np.unique(u_input)
            
            # 这里是对原始数据进行长度统一化的处理结果
            #print(node.tolist() + (max_n_node - len(node)) * [0])
            items.append(node.tolist() + (max_n_node - len(node)) * [0])

            u_A = np.zeros((max_n_node, max_n_node))
            u_A_in_out = np.zeros((max_n_node, max_n_node))
            
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]

                if self.sgc_att == False :
                    u_A[u][v] = 1
                    u_A[v][u] = 1
                else :
                    u_A[u][v] += 1
                    u_A[v][u] += 1
                
                if self.gnn_att == False :
                    u_A_in_out[u][v] = 1
                else :
                    u_A_in_out[u][v] += 1
                    if u_A_in_out[v][u] > 0:
                        u_A_in_out[v][u] += 1
                        u_A_in_out[u][v] = u_A_in_out[v][u]

            #在52.4894 18.2065这个历史最佳结果里，我在下面这一行用的A是A.append(u_A_in_out)，即有向图的特征矩阵
            A.append(u_A)         #在这里我构造一个无向图的特征矩阵输入sgc，看看结果有没有变好

            u_sum_in = np.sum(u_A_in_out, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A_in_out, u_sum_in)

            u_sum_out = np.sum(u_A_in_out, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A_in_out.transpose(), u_sum_out)

            u_A_in_out = np.concatenate([u_A_in, u_A_out]).transpose()
            A_in_out.append(u_A_in_out)
            
            #alias就相当于一个记号，在对原始数据进行排序和合并后仍然记录着原始数据的大小顺序及位置
            alias = [np.where(node == i)[0][0] for i in u_input]
            #print(alias)
            alias_inputs.append(alias)

        return alias_inputs, A, A_in_out, items, mask, targets
