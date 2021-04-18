import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.data import Data

import test
from config import config
from util.parameter_util import *
from util.train_util import *


# gpu_ids = [0, 1]


class DynamicKGE(nn.Module):
    def __init__(self, config):
        super(DynamicKGE, self).__init__()

        self.entity_emb = nn.Embedding(config.entity_total, config.dim)
        self.relation_emb = nn.Embedding(config.relation_total, config.dim)

        self.entity_context = nn.Embedding(config.entity_total, config.dim)
        self.relation_context = nn.Embedding(config.relation_total, config.dim)
        # self.entity_context = nn.Embedding(config.entity_total + 1, config.dim, padding_idx=config.entity_total)
        # self.relation_context = nn.Embedding(config.relation_total + 1, config.dim, padding_idx=config.relation_total)

        # self.entity_gcn_weight = nn.Parameter(torch.Tensor(config.dim, config.dim))
        self.relation_gcn_weight = nn.Parameter(torch.Tensor(config.dim, config.dim))

        self.gate_entity = nn.Parameter(torch.Tensor(config.dim))
        self.gate_relation = nn.Parameter(torch.Tensor(config.dim))

        self.v_ent = nn.Parameter(torch.Tensor(config.dim))
        self.v_rel = nn.Parameter(torch.Tensor(config.dim))

        self.conv1 = RGCNConv(config.dim, config.dim, config.relation_total * 2, num_bases=4)
        self.conv2 = RGCNConv(config.dim, config.dim, config.relation_total * 2, num_bases=4)

        self.entity_o = nn.Parameter(torch.rand(config.entity_total, config.dim), requires_grad=False)
        self.relation_o = nn.Parameter(torch.rand(config.entity_total, config.dim), requires_grad=False)
        # self.entity_o = torch.rand(config.entity_total, config.dim)
        # self.relation_o = torch.rand(config.entity_total, config.dim)

        self._init_parameters()

    def _init_parameters(self):
        # nn.init.xavier_uniform_(self.entity_emb.data)
        # nn.init.xavier_uniform_(self.relation_emb.data)
        # nn.init.xavier_uniform_(self.entity_context.data)
        # nn.init.xavier_uniform_(self.relation_context.data)
        nn.init.uniform_(self.gate_entity.data)
        nn.init.uniform_(self.gate_relation.data)
        nn.init.uniform_(self.v_ent.data)
        nn.init.uniform_(self.v_rel.data)

        # stdv = 1. / math.sqrt(self.entity_gcn_weight.size(1))
        # self.entity_gcn_weight.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.relation_gcn_weight.size(1))
        self.relation_gcn_weight.data.uniform_(-stdv, stdv)

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, p=config.norm, dim=1)

    def get_entity_context(self, entities):
        entities_context = []
        for e in entities:
            entities_context.extend(config.entity_adj_table.get(int(e), [config.entity_total] * config.max_context_num))
        # return entities_context
        return torch.LongTensor(entities_context).cuda()

    def get_relation_context(self, relations):
        relations_context = []
        for r in relations:
            relations_context.extend(
                config.relation_adj_table.get(int(r), [config.relation_total] * 2 * config.max_context_num))
        # return relations_context
        return torch.LongTensor(relations_context).cuda()

    def get_adj_entity_vec(self, entity_vec_list, adj_entity_list):
        # adj_entity_vec_list = self.entity_context[adj_entity_list]
        adj_entity_vec_list = self.entity_context(adj_entity_list)
        adj_entity_vec_list = adj_entity_vec_list.view(-1, config.max_context_num, config.dim)

        return torch.cat((entity_vec_list.unsqueeze(1), adj_entity_vec_list), dim=1)

    def get_adj_relation_vec(self, relation_vec_list, adj_relation_list):
        # adj_relation_vec_list = self.relation_context[adj_relation_list]
        adj_relation_vec_list = self.relation_context(adj_relation_list)
        adj_relation_vec_list = adj_relation_vec_list.view(-1, config.max_context_num, 2,
                                                           config.dim).cuda()
        adj_relation_vec_list = torch.sum(adj_relation_vec_list, dim=2)

        return torch.cat((relation_vec_list.unsqueeze(1), adj_relation_vec_list), dim=1)

    def score(self, o, adj_vec_list, target='entity'):
        os = torch.cat(tuple([o] * (config.max_context_num+1)), dim=1).reshape(-1, config.max_context_num+1, config.dim)
        tmp = F.relu(torch.mul(adj_vec_list, os), inplace=False)  # batch x max x 2dim
        if target == 'entity':
            score = torch.matmul(tmp, self.v_ent)  # batch x max
        else:
            score = torch.matmul(tmp, self.v_rel)
        return score

    def calc_subgraph_vec(self, o, adj_vec_list, target="entity"):
        alpha = self.score(o, adj_vec_list, target)
        alpha = F.softmax(alpha)

        sg = torch.sum(torch.mul(torch.unsqueeze(alpha, dim=2), adj_vec_list), dim=1)  # batch x dim
        return sg

    def gcn(self, A, H, target='entity'):
        support = torch.matmul(A, H)
        if target == 'entity':
            output = F.relu(torch.matmul(support, self.entity_gcn_weight))
        elif target == 'relation':
            output = F.relu(torch.matmul(support, self.relation_gcn_weight))
        return output

    def save_parameters(self, parameter_path):
        if not os.path.exists(parameter_path):
            os.makedirs(parameter_path)

        ent_f = open(os.path.join(parameter_path, 'entity_o'), "w")
        ent_f.write(json.dumps(self.pht_o))
        ent_f.close()

        rel_f = open(os.path.join(parameter_path, 'relation_o'), "w")
        rel_f.write(json.dumps(self.pr_o))
        rel_f.close()

        para2vec = {}
        lists = self.state_dict()
        for var_name in lists:
            para2vec[var_name] = lists[var_name].cpu().numpy().tolist()

        f = open(os.path.join(parameter_path, 'all_parameters'), "w")
        f.write(json.dumps(para2vec))
        f.close()

    def save_phrt_o(self, pos_h, pos_r, pos_t, ph_o, pr_o, pt_o):
        for i in range(len(pos_h)):
            h = str(int(pos_h[i]))
            self.pht_o[h] = ph_o[i].detach().cpu().numpy().tolist()

            t = str(int(pos_t[i]))
            self.pht_o[t] = pt_o[i].detach().cpu().numpy().tolist()

            r = str(int(pos_r[i]))
            self.pr_o[r] = pr_o[i].detach().cpu().numpy().tolist()

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.binary_cross_entropy_with_logits(score, target)

    def forward(self, entity, edge_index, edge_type, edge_norm, samples, DAD_rel):
        entity_context = self.entity_context(entity.long())
        relation_idx = torch.arange(config.relation_total).cuda()
        relation_context = self.relation_context(relation_idx)

        entity_context = F.relu(self.conv1(entity_context, edge_index, edge_type, edge_norm))
        # entity_context = F.dropout(entity_context, p=0.2, training=self.training)
        entity_context = F.relu(self.conv2(entity_context, edge_index, edge_type, edge_norm))

        relation_context = torch.matmul(DAD_rel, relation_context)
        relation_context = F.relu(torch.matmul(relation_context, self.relation_gcn_weight))


        head_o = torch.mul(torch.sigmoid(self.gate_entity), self.entity_emb(samples[:, 0])) + torch.mul(1 - torch.sigmoid(self.gate_entity), entity_context[samples[:, 0]])
        rel_o = torch.mul(torch.sigmoid(self.gate_relation), self.relation_emb(samples[:, 1])) + torch.mul(1 - torch.sigmoid(self.gate_relation), relation_context[samples[:, 1]])
        tail_o = torch.mul(torch.sigmoid(self.gate_entity), self.entity_emb(samples[:, 2])) + torch.mul(1 - torch.sigmoid(self.gate_entity), entity_context[samples[:, 2]])

        # save embeddings
        self.entity_o[entity[samples[:config.sample_size, 0]].long()] = head_o[:config.sample_size]
        self.entity_o[entity[samples[:config.sample_size, 2]].long()] = tail_o[:config.sample_size]
        self.relation_o[samples[:config.sample_size, 1]] = rel_o[:config.sample_size]

        return head_o, rel_o, tail_o

    # def forward(self, epoch, golden_triples, negative_triples, ph_A, pr_A, pt_A, nh_A, nr_A, nt_A):
    #     # multi golden and multi negative
    #     pos_h, pos_r, pos_t = golden_triples
    #     neg_h, neg_r, neg_t = negative_triples
    #
    #     p_h = self.entity_emb[pos_h.cpu().numpy()]
    #     p_t = self.entity_emb[pos_t.cpu().numpy()]
    #     p_r = self.relation_emb[pos_r.cpu().numpy()]
    #     n_h = self.entity_emb[neg_h.cpu().numpy()]
    #     n_t = self.entity_emb[neg_t.cpu().numpy()]
    #     n_r = self.relation_emb[neg_r.cpu().numpy()]
    #
    #     ph_adj_entity_list = self.get_entity_context(pos_h)
    #     pt_adj_entity_list = self.get_entity_context(pos_t)
    #     nh_adj_entity_list = self.get_entity_context(neg_h)
    #     nt_adj_entity_list = self.get_entity_context(neg_t)
    #     pr_adj_relation_list = self.get_relation_context(pos_r)
    #     nr_adj_relation_list = self.get_relation_context(neg_r)
    #
    #     ph_adj_entity_vec_list = self.get_adj_entity_vec(p_h, ph_adj_entity_list)
    #     pt_adj_entity_vec_list = self.get_adj_entity_vec(p_t, pt_adj_entity_list)
    #     nh_adj_entity_vec_list = self.get_adj_entity_vec(n_h, nh_adj_entity_list)
    #     nt_adj_entity_vec_list = self.get_adj_entity_vec(n_t, nt_adj_entity_list)
    #     pr_adj_relation_vec_list = self.get_adj_relation_vec(p_r, pr_adj_relation_list)
    #     nr_adj_relation_vec_list = self.get_adj_relation_vec(n_r, nr_adj_relation_list)
    #
    #     # gcn
    #     ph_adj_entity_vec_list = self.gcn(ph_A, ph_adj_entity_vec_list, target='entity')
    #     pt_adj_entity_vec_list = self.gcn(pt_A, pt_adj_entity_vec_list, target='entity')
    #     nh_adj_entity_vec_list = self.gcn(nh_A, nh_adj_entity_vec_list, target='entity')
    #     nt_adj_entity_vec_list = self.gcn(nt_A, nt_adj_entity_vec_list, target='entity')
    #     pr_adj_relation_vec_list = self.gcn(pr_A, pr_adj_relation_vec_list, target='relation')
    #     nr_adj_relation_vec_list = self.gcn(nr_A, nr_adj_relation_vec_list, target='relation')
    #
    #     # ph_sg = ph_adj_entity_vec_list.mean(dim=1)
    #     # pt_sg = pt_adj_entity_vec_list.mean(dim=1)
    #     # nh_sg = nh_adj_entity_vec_list.mean(dim=1)
    #     # nt_sg = nt_adj_entity_vec_list.mean(dim=1)
    #     # pr_sg = pr_adj_relation_vec_list.mean(dim=1)
    #     # nr_sg = nr_adj_relation_vec_list.mean(dim=1)
    #
    #     ph_sg = self.calc_subgraph_vec(p_h, ph_adj_entity_vec_list, target='entity')
    #     pt_sg = self.calc_subgraph_vec(p_t, pt_adj_entity_vec_list, target='entity')
    #     nh_sg = self.calc_subgraph_vec(n_h, nh_adj_entity_vec_list, target='entity')
    #     nt_sg = self.calc_subgraph_vec(n_t, nt_adj_entity_vec_list, target='entity')
    #     pr_sg = self.calc_subgraph_vec(p_r, pr_adj_relation_vec_list, target='relation')
    #     nr_sg = self.calc_subgraph_vec(n_r, nr_adj_relation_vec_list, target='relation')
    #
    #     ph_o = torch.mul(F.sigmoid(self.gate_entity), p_h) + torch.mul(1 - F.sigmoid(self.gate_entity), ph_sg)
    #     pt_o = torch.mul(F.sigmoid(self.gate_entity), p_t) + torch.mul(1 - F.sigmoid(self.gate_entity), pt_sg)
    #     nh_o = torch.mul(F.sigmoid(self.gate_entity), n_h) + torch.mul(1 - F.sigmoid(self.gate_entity), nh_sg)
    #     nt_o = torch.mul(F.sigmoid(self.gate_entity), n_t) + torch.mul(1 - F.sigmoid(self.gate_entity), nt_sg)
    #     pr_o = torch.mul(F.sigmoid(self.gate_relation), p_r) + torch.mul(1 - F.sigmoid(self.gate_relation), pr_sg)
    #     nr_o = torch.mul(F.sigmoid(self.gate_relation), n_r) + torch.mul(1 - F.sigmoid(self.gate_relation), nr_sg)
    #
    #     # score for loss
    #     p_score = self._calc(ph_o, pt_o, pr_o)
    #     n_score = self._calc(nh_o, nt_o, nr_o)
    #
    #     if epoch == config.train_times-1:
    #         self.save_phrt_o(pos_h, pos_r, pos_t, ph_o, pr_o, pt_o)
    #
    #     return p_score, n_score

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

def main():
    # print('preparing data...')
    # phs, prs, pts, nhs, nrs, nts = config.prepare_data()
    # print('preparing data complete')
    train_triples = config.train_triples
    valid_triples = config.valid_triples
    test.triples = config.test_triples
    sample_size = config.sample_size
    negative_rate = 1
    num_rels = config.relation_total
    model_state_file = config.model_state_file



    print('train starting...')
    model = DynamicKGE(config).cuda()
    print("model:")
    print(model)

    if config.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    criterion = nn.MarginRankingLoss(config.margin, reduction='sum').cuda()

    for epoch in range(config.train_times):
        start_time = time.time()
        print('----------training the ' + str(epoch) + ' epoch----------')
        model.train()
        optimizer.zero_grad()

        all_edges = np.arange(len(train_triples))
        edges = np.random.choice(all_edges, sample_size, replace=False)
        edges = train_triples[edges]
        head, rel, tail = edges.transpose()
        uniq_entity, entity_idx, entity_idx_inv = np.unique((head, tail), return_index=True, return_inverse=True)
        head, tail = np.reshape(entity_idx_inv, (2, -1))
        relabeled_edges = np.stack((head, rel, tail)).transpose()

        relation_entity_table = dict()
        A_rel = torch.eye(config.relation_total, config.relation_total).cuda()
        D_rel = np.eye(config.relation_total, config.relation_total)
        for i in range(relabeled_edges.shape[0]):
            h, r, t = relabeled_edges[i]
            relation_entity_table.setdefault(r, set()).add(h)
            relation_entity_table.setdefault(r, set()).add(t)
        for relation in range(config.relation_total):
            if relation in relation_entity_table:
                for index in range(relation+1 ,config.relation_total):
                    if index != relation and index in relation_entity_table:
                        if not relation_entity_table[relation].isdisjoint(relation_entity_table[index]):
                            A_rel[relation, index] = 1
                            A_rel[index, relation] = 1
                            D_rel[relation, relation] += 1

        D_rel = np.linalg.inv(D_rel)
        D_rel = torch.Tensor(D_rel).cuda()
        i = list(range(config.relation_total))
        D_rel[i, i] = torch.sqrt(D_rel[i, i])

        DAD_rel = D_rel.mm(A_rel).mm(D_rel)

        # Negative sampling
        samples, labels = negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)

        head = torch.tensor(head, dtype=torch.long)
        tail = torch.tensor(tail, dtype=torch.long)
        rel = torch.tensor(rel, dtype=torch.long)
        head, tail = torch.cat((head, tail)), torch.cat((tail, head))
        rel = torch.cat((rel, rel + num_rels))

        edge_index = torch.stack((head, tail))
        edge_type = rel
        edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)

        train_data = Data(edge_index=edge_index)
        train_data.entity = torch.from_numpy(uniq_entity)
        train_data.edge_type = edge_type
        train_data.edge_norm = edge_norm
        train_data.samples = torch.from_numpy(samples)
        train_data.labels = labels

        device = torch.device('cuda')
        train_data.to(device)

        head_o, rel_o, tail_o = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm, train_data.samples, DAD_rel)

        # score for loss
        p_scores = model._calc(head_o[0:train_data.samples.size()[0]//2], tail_o[0:train_data.samples.size()[0]//2], rel_o[0:train_data.samples.size()[0]//2])
        n_scores = model._calc(head_o[train_data.samples.size()[0]//2:], tail_o[train_data.samples.size()[0]//2:], rel_o[train_data.samples.size()[0]//2:])

        y = torch.Tensor([-1]*sample_size).cuda()
        loss = criterion(p_scores, n_scores, y)

        loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()

        end_time = time.time()
        print('----------epoch loss: ' + str(loss.item()) + ' ----------')
        print('----------epoch training time: ' + str(end_time-start_time) + ' s --------\n')
        pass
        # epoch_avg_loss = 0.0
        # for batch in range(config.nbatchs):
        #     optimizer.zero_grad()
        #     golden_triples, negative_triples = config.get_batch(config.batch_size, batch, epoch, phs, prs, pts, nhs, nrs, nts)
        #     ph_A, pr_A, pt_A = config.get_batch_A(golden_triples, config.entity_A, config.relation_A)
        #     nh_A, nr_A, nt_A = config.get_batch_A(negative_triples, config.entity_A, config.relation_A)
        #
        #     p_scores, n_scores = model(epoch, golden_triples, negative_triples, ph_A, pr_A, pt_A, nh_A, nr_A, nt_A)
        #     y = torch.Tensor([-1]).cuda()
        #     loss = criterion(p_scores, n_scores, y)
        #
        #     loss.backward()
        #     optimizer.step()
        #
        #     epoch_avg_loss += (float(loss.item()) / config.nbatchs)
        #     torch.cuda.empty_cache()
        # end_time = time.time()
        #
        # print('----------epoch avg loss: ' + str(epoch_avg_loss) + ' ----------')
        # print('----------epoch training time: ' + str(end_time-start_time) + ' s --------\n')

    print('train ending...')

    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

    # model.save_parameters(config.res_dir)

    print('test link prediction starting...')
    checkpoint = torch.load(model_state_file)
    state_dict = checkpoint['state_dict']
    entity_o = state_dict['entity_o']
    relation_o = state_dict['relation_o']
    # entity_embedding = state_dict['entity_emb.weight']
    # entity_context = state_dict['entity_context.weight']
    # relation_embedding = state_dict['relation_emb.weight']
    # relation_context = state_dict['relation_context.weight']
    # gate_entity = state_dict['gate_entity']
    # gate_relation = state_dict['gate_relation']
    # entity_o = torch.mul(torch.sigmoid(gate_entity), entity_embedding) + torch.mul(1 - torch.sigmoid(gate_entity), entity_context)
    # relation_o = torch.mul(torch.sigmoid(gate_relation), relation_embedding) + torch.mul(1 - torch.sigmoid(gate_relation), relation_context)

    # entity_emb, relation_emb = load_o_emb(config.res_dir, config.entity_total, config.relation_total, config.dim)


    test.test_link_prediction(config.test_list, set(config.train_list), entity_o, relation_o, config.norm)
    print('test link prediction ending...')


main()
