import numpy as np
import torch

def generate_graph(triples, relation_total):
    head, rel, tail = triples.transpose()
    uniq_entity, entity_idx, entity_idx_inv = np.unique((head, tail), return_index=True, return_inverse=True)
    head, tail = np.reshape(entity_idx_inv, (2, -1))
    relabeled_edges = np.stack((head, rel, tail)).transpose()

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

    head = torch.tensor(head, dtype=torch.long)
    tail = torch.tensor(tail, dtype=torch.long)
    rel = torch.tensor(rel, dtype=torch.long)
    head, tail = torch.cat((head, tail)), torch.cat((tail, head))
    rel = torch.cat((rel, rel + relation_total))

    edge_index = torch.stack((head, tail))
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.DAD_rel = DAD_rel
    data.uniq_entity = uniq_entity
    data.relabeled_edges = relabeled_edges
    return data

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
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

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

def get_head_batch(golden_triple, entity_total):
    head_batch = np.zeros((entity_total, 3), dtype=np.int32)
    head_batch[:, 0] = np.array(list(range(entity_total)))
    head_batch[:, 1] = np.array([golden_triple[1]] * entity_total)
    head_batch[:, 2] = np.array([golden_triple[2]] * entity_total)
    return head_batch


def get_tail_batch(golden_triple, entity_total):
    tail_batch = np.zeros((entity_total, 3), dtype=np.int32)
    tail_batch[:, 0] = np.array([golden_triple[0]] * entity_total)
    tail_batch[:, 1] = np.array([golden_triple[1]] * entity_total)
    tail_batch[:, 2] = np.array(list(range(entity_total)))
    return tail_batch