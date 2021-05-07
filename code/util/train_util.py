import random
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.data import Data

class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)


    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            if edge_type.shape[0] > 100000:  # prevent memory overflow, only for testing temporarily
                with torch.no_grad():
                    batch_size = 3000
                    batches = math.ceil(edge_type.shape[0]/batch_size)
                    out = torch.zeros_like(x_j)
                    for batch in range(batches):
                        index1 = batch*batch_size
                        index2 = min((batch+1)*batch_size, edge_type.shape[0])
                        edge_type_batch = edge_type[index1:index2]
                        w_batch = torch.index_select(w, 0, edge_type_batch)
                        x_j_batch = x_j[index1:index2]
                        out[index1:index2] = torch.bmm(x_j_batch.unsqueeze(1), w_batch).squeeze(-2)
            else:
                w = torch.index_select(w, 0, edge_type)
                out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm

def generate_graph(triples, relation_total):
    head, rel, tail = triples.transpose()
    uniq_entity, entity_idx, entity_idx_inv = np.unique((head, tail), return_index=True, return_inverse=True)
    head, tail = np.reshape(entity_idx_inv, (2, -1))
    relabeled_edges = np.stack((head, rel, tail)).transpose()

    # calculate A and D of relation
    relation_entity_table = dict()
    A_rel = torch.eye(relation_total, relation_total).cuda()
    D_rel = np.eye(relation_total, relation_total)
    for i in range(relabeled_edges.shape[0]):
        h, r, t = relabeled_edges[i]
        relation_entity_table.setdefault(r, set()).add(h)
        relation_entity_table.setdefault(r, set()).add(t)
    for relation in range(relation_total):
        if relation in relation_entity_table:
            for index in range(relation + 1, relation_total):
                if index != relation and index in relation_entity_table:
                    if not relation_entity_table[relation].isdisjoint(relation_entity_table[index]):
                        A_rel[relation, index] = 1
                        A_rel[index, relation] = 1
                        D_rel[relation, relation] += 1

    D_rel = np.linalg.inv(D_rel)
    D_rel = torch.Tensor(D_rel).cuda()
    i = list(range(relation_total))
    D_rel[i, i] = torch.sqrt(D_rel[i, i])

    DAD_rel = D_rel.mm(A_rel).mm(D_rel)
    # /calculate A and D of relation

    # Negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_entity), negative_rate=1)
    samples = torch.from_numpy(samples)

    head = torch.tensor(head, dtype=torch.long)
    tail = torch.tensor(tail, dtype=torch.long)
    rel = torch.tensor(rel, dtype=torch.long)
    head, tail = torch.cat((head, tail)), torch.cat((tail, head))
    rel = torch.cat((rel, rel + relation_total))

    edge_index = torch.stack((head, tail))
    edge_type = rel
    edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), relation_total)

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_norm
    data.samples = samples
    data.labels = torch.from_numpy(labels)
    data.DAD_rel = DAD_rel
    data.uniq_entity = uniq_entity
    data.relabeled_edges = relabeled_edges
    return data


def read_file(file_name):
    data = []  # [(h, r, t)]
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            li = line.split()
            if len(li) == 3:
                data.append((int(li[0]), int(li[2]), int(li[1])))
    return data

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def get_total(file_name):
    with open(file_name, 'r', encoding="utf-8") as f:
        return int(f.readline())


def get_basic_info(train_set):
    entity_set = set()
    relation_set = set()
    entity_context_dict = dict()
    relation_context_dict = dict()
    relation_entity_context_dict = dict()
    for (h, r, t) in train_set:
        entity_set.add(h)
        entity_set.add(t)
        relation_set.add(r)
        entity_context_dict.setdefault(h, set()).add((t, r))    # h: (t,r)
        entity_context_dict.setdefault(t, set()).add((h, r))
        relation_entity_context_dict.setdefault(r, set()).add(h)
        relation_entity_context_dict.setdefault(r, set()).add(t)
    for relation1 in relation_set:
        for relation2 in relation_set:
            if relation1 == relation2:
                continue
            if not relation_entity_context_dict[relation1].isdisjoint(relation_entity_context_dict[relation2]):
                relation_context_dict.setdefault(relation1, set()).add(relation2)

    return entity_set, relation_set, entity_context_dict, relation_context_dict


def construct_text2id_dict(file_name):
    result_dict = dict()
    with open(file_name, 'r', encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            text, id = line.rstrip("\n").split("\t")[:]
            result_dict[text] = int(id)
    return result_dict


def construct_id2text_dict(file_name):
    result_dict = dict()
    with open(file_name, 'r', encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            text, id = line.rstrip("\n").split("\t")[:]
            result_dict[int(id)] = text
    return result_dict


def convert_id_to_text(train_list, dataset):
    text_train_list = list()
    entityid2text_dict = construct_id2text_dict('./data/' + dataset + '/entity2id.txt')
    relationid2text_dict = construct_id2text_dict('./data/' + dataset + '/relation2id.txt')
    for (h, r, t) in train_list:
        text_train_list.append((entityid2text_dict[h], relationid2text_dict[r], entityid2text_dict[t]))
    return text_train_list


def get_1or2_path_from_head(head_ent, rel, entity_adj_table_with_rel):
    paths = {}  # actually second-order + first-order, {entity: [[edge]]}
    first_order_entity = set()
    first_order_relation = dict()

    if head_ent not in entity_adj_table_with_rel:
        return paths
    for tail_entity, relation in entity_adj_table_with_rel[head_ent]:
        first_order_entity.add(tail_entity)
        if relation != rel:
            if tail_entity in paths:
                paths[tail_entity].append([relation])
            else:
                paths[tail_entity] = [[relation]]

        if tail_entity in first_order_relation:
            first_order_relation[tail_entity].append(relation)
        else:
            first_order_relation[tail_entity] = [relation]

    for node in first_order_entity:
        if node not in entity_adj_table_with_rel:
            continue
        for tail_entity, relation in entity_adj_table_with_rel[node]:
            if tail_entity in paths:
                for r in first_order_relation[node]:
                    paths[tail_entity].append([r, relation])
            else:
                paths[tail_entity] = []
                for r in first_order_relation[node]:
                    paths[tail_entity].append([r, relation])

    return paths  # {entity: [[edge]]}


def find_relation_context(h, r, t, entity_adj_table_with_rel):
    tail_ent2paths = get_1or2_path_from_head(h, r, entity_adj_table_with_rel)
    return tail_ent2paths.get(t, [])


def construct_adj_table(train_list, entity_total, relation_total, max_context):
    entity_adj_table_with_rel = dict()  # {head_entity: [(tail_entity, relation)]}
    entity_adj_table = dict()  # {head_entity: [tail_entity]}
    relation_adj_table = dict()  # {relation: [[edge]]}

    for train_data in train_list:
        h, r, t = train_data
        entity_adj_table.setdefault(h, set()).add(t)
        entity_adj_table.setdefault(t, set()).add(h)
        entity_adj_table_with_rel.setdefault(h, list()).append((t, r))

    for train_data in train_list:
        h, r, t = train_data
        paths = find_relation_context(h, r, t, entity_adj_table_with_rel)
        relation_adj_table.setdefault(r, []).extend(paths)
    for k, v in relation_adj_table.items():
        relation_adj_table[k] = set([tuple(i) for i in v])

    max_context_num = max_context
    for k, v in entity_adj_table.items():
        if len(v) > max_context_num:
            res = list(v)
            res = res[:max_context_num]
            entity_adj_table[k] = set(res)
    for k, v in relation_adj_table.items():
        if len(v) > max_context_num:
            res = list(v)
            res = res[:max_context_num]
            relation_adj_table[k] = set(res)

    entity_DAD = torch.Tensor(entity_total, max_context_num + 1, max_context_num + 1).cuda()
    relation_DAD = torch.Tensor(relation_total, max_context_num + 1, max_context_num + 1).cuda()

    for entity in range(entity_total):
        A = torch.eye(max_context_num + 1, max_context_num + 1).cuda()
        tmp = torch.ones(max_context_num + 1).cuda()
        A[0, :max_context_num + 1] = tmp
        A[:max_context_num + 1, 0] = tmp

        D = np.eye(max_context_num + 1, max_context_num + 1)
        i = list(range(max_context_num + 1))
        D[i, i] = 2
        D[0][0] = max_context_num + 1

        if entity in entity_adj_table:
            neighbours_list = list(entity_adj_table[entity])
            for index, neighbour in enumerate(neighbours_list):
                if neighbour not in entity_adj_table:
                    continue
                for index2, neighbour2 in enumerate(neighbours_list):
                    if index == index2:
                        continue
                    if neighbour2 in entity_adj_table[neighbour]:
                        A[index+1, index2+1] = 1
                        D[index+1][index+1] += 1

        D = np.linalg.inv(D)
        D = torch.Tensor(D).cuda()
        D[i, i] = torch.sqrt(D[i, i])

        entity_DAD[entity] = D.mm(A).mm(D)

    for relation in range(relation_total):
        A = torch.eye(max_context_num + 1, max_context_num + 1).cuda()
        tmp = torch.ones(max_context_num + 1).cuda()
        A[0, :max_context_num + 1] = tmp
        A[:max_context_num + 1, 0] = tmp

        D = np.eye(max_context_num + 1, max_context_num + 1)
        i = list(range(max_context_num + 1))
        D[i, i] = 2
        D[0][0] = max_context_num + 1

        if relation in relation_adj_table:
            neighbours_set = relation_adj_table[relation]
            for index, neighbour in enumerate(neighbours_set):
                if len(neighbour) != 1:
                    continue
                if neighbour[0] not in relation_adj_table:
                    continue
                adj_set = relation_adj_table[neighbour[0]]
                for index2, neighbour2 in enumerate(neighbours_set):
                    if index == index2:
                        continue
                    if neighbour2 in adj_set:
                        A[index+1, index2+1] = 1
                        D[index+1][index+1] += 1
        D = np.linalg.inv(D)
        D = torch.Tensor(D).cuda()
        i = list(range(max_context_num + 1))
        D[i, i] = torch.sqrt(D[i, i])

        relation_DAD[relation] = D.mm(A).mm(D)

    for k, v in entity_adj_table.items():
        res = list(v)
        entity_adj_table[k] = res + [entity_total] * (max_context_num - len(res))  # 补padding

    for k, v in relation_adj_table.items():
        res = []
        for i in v:
            if len(i) == 1:
                res.extend(list(i))
                res.append(relation_total)
            else:
                res.extend(list(i))

        relation_adj_table[k] = res + [relation_total] * 2 * (max_context_num - len(res) // 2)  # 补padding

    return entity_adj_table, relation_adj_table, max_context_num, entity_DAD, relation_DAD
    # return entity_adj_table, max_context_num, entity_DAD


def bern_sampling_prepare(train_list):
    head2count = dict()
    tail2count = dict()
    for h, r, t in train_list:
        head2count[h] = head2count.get(h, 0) + 1
        tail2count[t] = tail2count.get(t, 0) + 1

    hpt = 0.0  # head per tail
    for t, count in tail2count.items():
        hpt += count
    hpt /= len(tail2count)

    tph = 0.0
    for h, count in head2count.items():
        tph += count
    tph /= len(head2count)

    return tph, hpt


def one_negative_sampling(golden_triple, train_set, entity_total, bern=True, tph=0.0, hpt=0.0):
    h, r, t = golden_triple
    if not bern:  # uniform sampling
        while True:
            e = random.randint(0, entity_total - 1)
            is_head = random.randint(0, 1)
            if is_head:
                if (e, r, t) in train_set:
                    continue
                else:
                    negative_triple = (e, r, t)
                    break
            else:
                if (h, r, e) in train_set:
                    continue
                else:
                    negative_triple = (h, r, e)
                    break
    else:
        sampling_head_prob = tph / (tph + hpt)
        while True:
            e = random.randint(0, entity_total - 1)
            is_head = random.random() > sampling_head_prob
            if is_head:
                if (e, r, t) in train_set:
                    continue
                else:
                    negative_triple = (e, r, t)
                    break
            else:
                if (h, r, e) in train_set:
                    continue
                else:
                    negative_triple = (h, r, e)
                    break

    return negative_triple


def get_batch(batch_size, batch, epoch, phs, prs, pts, nhs, nrs, nts):
    r = min((batch + 1) * batch_size, len(phs))

    return (phs[batch * batch_size: r], prs[batch * batch_size: r], pts[batch * batch_size: r]), \
           (nhs[epoch, batch * batch_size: r], nrs[epoch, batch * batch_size: r], nts[epoch, batch * batch_size: r])


def get_batch_A(triples, entity_A, relation_A):
    h, r, t = triples
    return entity_A[h.cpu().numpy()], relation_A[r.cpu().numpy()], entity_A[t.cpu().numpy()]