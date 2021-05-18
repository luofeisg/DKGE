import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from util.train_util import uniform, scatter

class DynamicKGE(nn.Module):
    def __init__(self, num_entities, num_relations, dim, norm):
        super(DynamicKGE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.norm = norm

        self.entity_emb = nn.Parameter(torch.Tensor(num_entities, dim))
        self.relation_emb = nn.Parameter(torch.Tensor(num_relations, dim))
        nn.init.xavier_uniform_(self.entity_emb)
        nn.init.xavier_uniform_(self.relation_emb)
        # self.entity_emb = nn.Embedding(config.entity_total, config.dim)
        # self.relation_emb = nn.Embedding(config.relation_total, config.dim)

        self.entity_context = nn.Embedding(num_entities, dim)
        self.relation_context = nn.Embedding(num_relations, dim)
        # self.entity_context = nn.Embedding(config.entity_total + 1, config.dim, padding_idx=config.entity_total)
        # self.relation_context = nn.Embedding(config.relation_total + 1, config.dim, padding_idx=config.relation_total)

        # self.entity_gcn_weight = nn.Parameter(torch.Tensor(config.dim, config.dim))
        self.relation_gcn_weight = nn.Parameter(torch.Tensor(dim, dim))

        self.gate_entity = nn.Parameter(torch.Tensor(dim))
        self.gate_relation = nn.Parameter(torch.Tensor(dim))

        self.conv1_entity = RGCNConv(dim, dim, num_relations * 2)
        self.conv2_entity = RGCNConv(dim, dim, num_relations * 2)

        # self.conv1_relation = RGCNConv(dim, dim, 1)
        # self.conv2_relation = RGCNConv(dim, dim, 1)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.uniform_(self.gate_entity.data)
        nn.init.uniform_(self.gate_relation.data)

        stdv = 1. / math.sqrt(self.relation_gcn_weight.size(1))
        self.relation_gcn_weight.data.uniform_(-stdv, stdv)

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, p=self.norm, dim=1)

    def gcn(self, A, H, target='entity'):
        support = torch.matmul(A, H)
        if target == 'entity':
            output = F.relu(torch.matmul(support, self.entity_gcn_weight))
        elif target == 'relation':
            output = F.relu(torch.matmul(support, self.relation_gcn_weight))
        return output

    def distmult(self, entity_o, relation_o, triplets):
        s = entity_o[triplets[:, 0]]
        r = relation_o[triplets[:, 1]]
        o = entity_o[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)

        return score

    def score_loss(self, entity_o, relation_o, triplets, target):
        h = entity_o[triplets[:, 0]]
        r = relation_o[triplets[:, 1]]
        t = entity_o[triplets[:, 2]]

        # score = self.distmult(entity_o, relation_o, triplets)
        # return F.binary_cross_entropy_with_logits(score, target)
        score = torch.norm(h + r - t, p=self.norm, dim=1)
        return score

    def reg_loss(self, entity_o, relation_o):
        return torch.mean(entity_o.pow(2)) + torch.mean(relation_o.pow(2))

    def forward(self, entity, edge_index, edge_type, edge_norm, DAD_rel):
        entity_emb = self.entity_emb[entity.long()]
        relation_emb = self.relation_emb
        entity_context = self.entity_context(entity.long())
        relation_context = self.relation_context.weight

        num_entity = entity.shape[0]

        # rgcn
        entity_context = F.relu(self.conv1_entity(entity_context, edge_index, edge_type, edge_norm, dim=num_entity))
        # entity_context = F.dropout(entity_context, p=0.2, training=self.training)
        # entity_context = self.conv2_entity(entity_context, edge_index, edge_type, edge_norm, dim=num_entity)

        # relation_context = F.relu(relation_context, relation_index, relation_type, relation_norm, )
        # gcn
        # relation_context = torch.matmul(DAD_rel, relation_context)
        # relation_context = F.relu(torch.matmul(relation_context, self.relation_gcn_weight))

        # calculate joint embedding
        entity_o = torch.mul(torch.sigmoid(self.gate_entity), entity_emb) + torch.mul(1 - torch.sigmoid(self.gate_entity), entity_context)
        relation_o = torch.mul(torch.sigmoid(self.gate_relation), relation_emb) + torch.mul(1 - torch.sigmoid(self.gate_entity), relation_context)

        return entity_o, relation_o

        # pure RGCN
        # entity_emb = self.entity_emb(entity.long())
        # relation_emb = self.relation_emb
        #
        # entity_emb = F.relu(self.conv1(entity_emb, edge_index, edge_type, edge_norm))
        # entity_emb = F.dropout(entity_emb, p=0.2, training=self.training)
        # entity_emb = self.conv2(entity_emb, edge_index, edge_type, edge_norm)
        #
        # return entity_emb, relation_emb

class DKGE_Online(nn.Module):
    def __init__(self, num_entities, num_relations, dim, norm):
        super(DKGE_Online, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.norm = norm

        self.entity_emb = nn.Parameter(torch.Tensor(num_entities, dim))
        self.relation_emb = nn.Parameter(torch.Tensor(num_relations, dim))
        nn.init.xavier_uniform_(self.entity_emb)
        nn.init.xavier_uniform_(self.relation_emb)
        # self.entity_emb = nn.Embedding(num_entities, dim)
        # self.relation_emb = nn.Embedding(num_relations, dim)

        self.entity_context = nn.Embedding(num_entities, dim)
        self.relation_context = nn.Embedding(num_relations, dim)

        # self.entity_gcn_weight = nn.Parameter(torch.Tensor(config.dim, config.dim), requires_grad=False)
        self.relation_gcn_weight = nn.Parameter(torch.Tensor(dim, dim))

        self.gate_entity = nn.Parameter(torch.Tensor(dim))
        self.gate_relation = nn.Parameter(torch.Tensor(dim))

        self.conv1_entity = RGCNConv(dim, dim, num_relations * 2)
        self.conv2_entity = RGCNConv(dim, dim, num_relations * 2)

        # self.conv1_relation = RGCNConv(dim, dim, 1)
        # self.conv2_relation = RGCNConv(dim, dim, 1)

        # self._init_parameters()

    def _init_parameters(self, entity_emb, relation_emb, entity_context, relation_context, entity_gcn_weight,
                         relation_gcn_weight, gate_entity, gate_relation, v_entity, v_relation,
                         entity_o_emb, relation_o_emb):
        self.entity_gcn_weight.data = entity_gcn_weight
        self.relation_gcn_weight.data = relation_gcn_weight
        self.gate_entity.data = gate_entity
        self.gate_relation.data = gate_relation
        self.v_ent.data = v_entity
        self.v_rel.data = v_relation

        self.entity_emb.weight.data[:, :] = entity_emb[:, :]
        self.relation_emb.weight.data[:, :] = relation_emb[:, :]

        self.entity_context.weight.data[:self.num_entities, :] = entity_context[:self.num_entities, :]
        self.relation_context.weight.data[:self, :] = relation_context[self.num_relations, :]

        for i in range(len(entity_o_emb)):
            e = str(i)
            self.pht_o[e] = entity_o_emb[i].detach().cpu().numpy().tolist()

        for i in range(len(relation_o_emb)):
            r = str(i)
            self.pr_o[r] = relation_o_emb[i].detach().cpu().numpy().tolist()

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, p=self.norm, dim=1)

    def gcn(self, A, H, target='entity'):
        support = torch.matmul(A, H)
        if target == 'entity':
            output = F.relu(torch.matmul(support, self.entity_gcn_weight))
        elif target == 'relation':
            output = F.relu(torch.matmul(support, self.relation_gcn_weight))
        return output

    def score_loss(self, entity_o, relation_o, triplets, target):
        h = entity_o[triplets[:, 0]]
        r = relation_o[triplets[:, 1]]
        t = entity_o[triplets[:, 2]]

        # score = self.distmult(entity_o, relation_o, triplets)
        # return F.binary_cross_entropy_with_logits(score, target)
        score = torch.norm(h + r - t, p=self.norm, dim=1)
        return score

    def reg_loss(self, entity_o, relation_o):
        return torch.mean(entity_o.pow(2)) + torch.mean(relation_o.pow(2))

    # def forward(self, entity, edge_index, edge_type, edge_norm, DAD_rel, affected_entities, affected_relations):
    def forward(self, entity, edge_index, edge_type, edge_norm, DAD_rel):
        entity_emb = self.entity_emb[entity.long()]
        relation_emb = self.relation_emb
        entity_context = self.entity_context(entity.long())
        relation_context = self.relation_context.weight

        num_entity = entity.shape[0]

        # rgcn
        entity_context = F.relu(self.conv1_entity(entity_context, edge_index, edge_type, edge_norm, dim=num_entity))
        # entity_context = F.dropout(entity_context, p=0.2, training=self.training)
        # entity_context = self.conv2_entity(entity_context, edge_index, edge_type, edge_norm, dim=num_entity)

        # relation_context = F.relu(relation_context, relation_index, relation_type, relation_norm, )
        # gcn
        # relation_context = torch.matmul(DAD_rel, relation_context)
        # relation_context = F.relu(torch.matmul(relation_context, self.relation_gcn_weight))

        # calculate joint embedding
        entity_o = torch.mul(torch.sigmoid(self.gate_entity), entity_emb) + torch.mul(1 - torch.sigmoid(self.gate_entity), entity_context)
        relation_o = torch.mul(torch.sigmoid(self.gate_relation), relation_emb) + torch.mul(1 - torch.sigmoid(self.gate_entity), relation_context)

        return entity_o, relation_o

class RGCNConv(nn.Module):
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
    """

    def __init__(self, in_channels, out_channels, num_relations, root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = 4

        self.basis = nn.Parameter(torch.Tensor(self.num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, self.num_bases))

        self.weight_relation = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        stdv = 1. / math.sqrt(self.weight_relation.size(1))
        self.weight_relation.data.uniform_(-stdv, stdv)

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        # size = self.num_bases * self.in_channels
        size = self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm, dim):
        # message passing
        x_j = x[edge_index[1]]
        w = self.weight_relation
        # w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        # w = w.view(self.num_relations, self.in_channels, self.out_channels)
        num_edges = edge_type.shape[0]
        if num_edges > 100000:  # for testing, do not calculate gradient
            with torch.no_grad():
                batch_size = 3000
                batches = math.ceil(num_edges / batch_size)
                out = torch.zeros_like(x_j)
                for batch in range(batches):
                    index1 = batch * batch_size
                    index2 = min((batch + 1) * batch_size, num_edges)
                    edge_type_batch = edge_type[index1:index2]
                    w_batch = torch.index_select(w, 0, edge_type_batch)
                    x_j_batch = x_j[index1:index2]
                    out[index1:index2] = torch.bmm(x_j_batch.unsqueeze(1), w_batch).squeeze(-2)
        else:
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
        if edge_norm is not None:
            out = out * edge_norm.view(-1, 1)

        # aggregate
        out = scatter(out, edge_index[0], dim=-2, dim_size=dim, reduce='sum')

        # update
        if self.root is not None:   #self loop
            if x is None:
                out = out + self.root
            else:
                out = out + torch.matmul(x, self.root) # self loop
        if self.bias is not None:
            # out = out + self.bias
            pass
        return out
