import random
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from typing import Optional

def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    else:
        raise ValueError


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


class Data(object):
    r"""A plain old python object modeling a single graph with various
    (optional) attributes:

    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        normal (Tensor, optional): Normal vector matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        face (LongTensor, optional): Face adjacency matrix with shape
            :obj:`[3, num_faces]`. (default: :obj:`None`)

    The data object is not restricted to these attributes and can be extented
    by any other additional data.

    Example::

        data = Data(x=x, edge_index=edge_index)
        data.train_idx = torch.tensor([...], dtype=torch.long)
        data.test_mask = torch.tensor([...], dtype=torch.bool)
    """
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, normal=None, face=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        self.normal = normal
        self.face = face
        for key, item in kwargs.items():
            if key == 'num_nodes':
                self.__num_nodes__ = item
            else:
                self[key] = item

        if edge_index is not None and edge_index.dtype != torch.long:
            raise ValueError(
                (f'Argument `edge_index` needs to be of type `torch.long` but '
                 f'found type `{edge_index.dtype}`.'))

        if face is not None and face.dtype != torch.long:
            raise ValueError(
                (f'Argument `face` needs to be of type `torch.long` but found '
                 f'type `{face.dtype}`.'))

    @classmethod
    def from_dict(cls, dictionary):
        r"""Creates a data object from a python dictionary."""
        data = cls()

        for key, item in dictionary.items():
            data[key] = item

        return data

    def to_dict(self):
        return {key: item for key, item in self}

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]
    @property
    def num_edges(self):
        r"""Returns the number of edges in the graph."""
        for key, item in self('edge_index', 'edge_attr'):
            return item.size(self.__cat_dim__(key, item))
        for key, item in self('adj', 'adj_t'):
            return item.nnz()
        return None

    @property
    def num_faces(self):
        r"""Returns the number of faces in the mesh."""
        if self.face is not None:
            return self.face.size(self.__cat_dim__('face', self.face))
        return None

    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the graph."""
        if self.x is None:
            return 0
        return 1 if self.x.dim() == 1 else self.x.size(1)

    @property
    def num_features(self):
        r"""Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self):
        r"""Returns the number of features per edge in the graph."""
        if self.edge_attr is None:
            return 0
        return 1 if self.edge_attr.dim() == 1 else self.edge_attr.size(1)

    def is_directed(self):
        r"""Returns :obj:`True`, if graph edges are directed."""
        return not self.is_undirected()

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys):
        r"""Ensures a contiguous memory layout for all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, all present attributes are ensured to
        have a contiguous memory layout."""
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys, **kwargs):
        r"""Performs tensor dtype and/or device conversion to all attributes
        :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

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

    def __init__(self, in_channels, out_channels, num_relations, root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        # self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        # self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        self.weight = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        # uniform(size, self.basis)
        # uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forwarda(self, x, edge_index, edge_type, edge_norm=None, size=None, node_dim=0):
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def forward(self, x, edge_index, edge_type, edge_norm, dim):
        # message passing
        x_j = x[edge_index[0]]
        w = self.weight
        num_edges = edge_type.shape[0]
        if num_edges > 100000:
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
                out = out + torch.matmul(x, self.root)
        # if self.bias is not None:
        #     out = out + self.bias
        return out


    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        # w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = self.weight

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
        return aggr_out
        # if self.root is not None:
        #     if x is None:
        #         out = aggr_out + self.root
        #     else:
        #         out = aggr_out + torch.matmul(x, self.root)
        #
        # if self.bias is not None:
        #     out = out + self.bias
        # return out


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
    deg = scatter_sum(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    # deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm


def generate_graph(triples, relation_total):
    head, rel, tail = triples.transpose()
    uniq_entity, entity_idx, entity_idx_inv = np.unique((head, tail), return_index=True, return_inverse=True)
    head, tail = np.reshape(entity_idx_inv, (2, -1))
    relabeled_edges = np.stack((head, rel, tail)).transpose()

    # Negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_entity), negative_rate=1)
    samples = torch.from_numpy(samples)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(triples.shape[0] * 0.5)
    graph_split_ids = np.random.choice(np.arange(triples.shape[0]), size=split_size, replace=False)
    head = head[graph_split_ids]
    tail = tail[graph_split_ids]
    rel = rel[graph_split_ids]

    split_relabeled_edges = np.stack((head, rel, tail)).transpose()
    # calculate A and D of relation
    relation_entity_table = dict()
    A_rel = torch.eye(relation_total, relation_total).cuda()
    D_rel = np.eye(relation_total, relation_total)
    for i in range(split_relabeled_edges.shape[0]):
        h, r, t = split_relabeled_edges[i]
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
    data.DAD_rel = DAD_rel
    data.uniq_entity = uniq_entity
    data.relabeled_edges = relabeled_edges
    data.samples = samples
    data.labels = torch.from_numpy(labels)

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