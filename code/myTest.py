import numpy as np
import datetime
import torch
import sys
import torch.nn as nn

# x = torch.randn(1)
#
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # a CUDA device object
#     y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
#     x = x.to(device)                       # or just use strings ``.to("cuda")``
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!


# entity_DAD = torch.tensor(1000,31,31)

# log
a = torch.randn(31,31,100,100)
b = torch.randn(1,1,31,100)
c = torch.matmul(b,a)

print((a[0][0]*b[0][0] - c[0][0]).sum())
print()




        # p_h = self.entity_emb[pos_h.cpu().numpy()]
        # p_t = self.entity_emb[pos_t.cpu().numpy()]
        # p_r = self.relation_emb[pos_r.cpu().numpy()]
        # n_h = self.entity_emb[neg_h.cpu().numpy()]
        # n_t = self.entity_emb[neg_t.cpu().numpy()]
        # n_r = self.relation_emb[neg_r.cpu().numpy()]
        #
        # # RGCN start
        # p_edges = torch.transpose(torch.stack((pos_h, pos_r, pos_t)), 0, 1)
        # n_edges = torch.transpose(torch.stack((neg_h, neg_r, neg_t)), 0, 1)
        # p_edges = p_edges.cpu().numpy()
        # n_edges = n_edges.cpu().numpy()
        # src, rel, dst = p_edges.transpose()
        # uniq_entity, edges = np.unique((src, dst), return_inverse=True)
        # src, dst = np.reshape(edges, (2, -1))
        #
        # src = torch.tensor(src, dtype=torch.long)
        # dst = torch.tensor(dst, dtype=torch.long)
        # rel = torch.tensor(rel, dtype=torch.long)
        # # Create bi-directional graph
        # src, dst = torch.cat((src, dst)), torch.cat((dst, src))
        # rel = torch.cat((rel, rel + config.relation_total))
        # uniq_entity = torch.tensor(uniq_entity)
        #
        # edge_index = torch.stack((src, dst)).long()
        # edge_type = rel.long()
        # # edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), config.relation_total)
        # edge_norm = torch.rand(config.batch_size * 2)
        #
        # device = torch.device('cuda')
        # edge_index = edge_index.to(device)
        # edge_type = edge_type.to(device)
        # edge_norm = edge_norm.to(device)
        # uniq_entity = uniq_entity.to(device)
        #
        # x = self.entity_context_emb(uniq_entity.long())
        # x = self.conv1(x, edge_index, edge_type, edge_norm)
        # x = F.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        # x = F.dropout(x, p=config.dropout_ratio, training=self.training)
        # x = self.conv2(x, edge_index, edge_type, edge_norm)
        # # RGCN end


