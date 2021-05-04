import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from config import online_config as config
import test
from util.train_util import *

# gpu_ids = [0, 1]


class DKGE_Online(nn.Module):
    def __init__(self):
        super(DKGE_Online, self).__init__()

        # self.entity_emb = nn.Parameter(torch.Tensor(config.entity_total, config.dim))
        # self.relation_emb = nn.Parameter(torch.Tensor(config.relation_total, config.dim))
        self.entity_emb = nn.Embedding(config.entity_total, config.dim)
        self.relation_emb = nn.Embedding(config.relation_total, config.dim)

        self.entity_context = nn.Embedding(config.entity_total, config.dim)
        self.relation_context = nn.Embedding(config.relation_total, config.dim)

        # self.entity_gcn_weight = nn.Parameter(torch.Tensor(config.dim, config.dim), requires_grad=False)
        self.relation_gcn_weight = nn.Parameter(torch.Tensor(config.dim, config.dim))

        self.gate_entity = nn.Parameter(torch.Tensor(config.dim))
        self.gate_relation = nn.Parameter(torch.Tensor(config.dim))

        self.v_ent = nn.Parameter(torch.Tensor(config.dim))
        self.v_rel = nn.Parameter(torch.Tensor(config.dim))

        self.conv1 = RGCNConv(config.dim, config.dim, config.relation_total * 2, num_bases=4)
        self.conv2 = RGCNConv(config.dim, config.dim, config.relation_total * 2, num_bases=4)

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

        self.entity_context.weight.data[:config.entity_total, :] = entity_context[:config.entity_total, :]
        self.relation_context.weight.data[:config.relation_total, :] = relation_context[:config.relation_total, :]

        for i in range(len(entity_o_emb)):
            e = str(i)
            self.pht_o[e] = entity_o_emb[i].detach().cpu().numpy().tolist()

        for i in range(len(relation_o_emb)):
            r = str(i)
            self.pr_o[r] = relation_o_emb[i].detach().cpu().numpy().tolist()

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

        ent_f = open(os.path.join(parameter_path, "entity_o"), "w")
        ent_f.write(json.dumps(self.pht_o))
        ent_f.close()

        rel_f = open(os.path.join(parameter_path, "relation_o"), "w")
        rel_f.write(json.dumps(self.pr_o))
        rel_f.close()

        para2vec = {}
        lists = self.state_dict()
        for var_name in lists:
            para2vec[var_name] = lists[var_name].cpu().numpy().tolist()

        f = open(os.path.join(parameter_path, "all_parameters"), "w")
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

    def forward(self, entity, edge_index, edge_type, edge_norm, DAD_rel):
        entity_context = self.entity_context(entity.long())
        relation_idx = torch.arange(config.relation_total).cuda()
        relation_context = self.relation_context(relation_idx)

        entity_context = F.relu(self.conv1(entity_context, edge_index, edge_type, edge_norm))
        # entity_context = F.dropout(entity_context, p=0.2, training=self.training)
        entity_context = F.relu(self.conv2(entity_context, edge_index, edge_type, edge_norm))

        relation_context = torch.matmul(DAD_rel, relation_context)
        relation_context = F.relu(torch.matmul(relation_context, self.relation_gcn_weight))

        return entity_context, relation_context

    # def forward(self, epoch, golden_triples, negative_triples, ph_A, pr_A, pt_A, nh_A, nr_A, nt_A):
    #     # multi golden and multi negative
    #     pos_h, pos_r, pos_t = golden_triples
    #     neg_h, neg_r, neg_t = negative_triples
    #
    #     # p_h = self.entity_emb[pos_h.cpu().numpy()]
    #     # p_t = self.entity_emb[pos_t.cpu().numpy()]
    #     # p_r = self.relation_emb[pos_r.cpu().numpy()]
    #     # n_h = self.entity_emb[neg_h.cpu().numpy()]
    #     # n_t = self.entity_emb[neg_t.cpu().numpy()]
    #     # n_r = self.relation_emb[neg_r.cpu().numpy()]
    #     p_h = self.entity_emb(pos_h)
    #     p_t = self.entity_emb(pos_t)
    #     p_r = self.relation_emb(pos_r)
    #     n_h = self.entity_emb(neg_h)
    #     n_t = self.entity_emb(neg_t)
    #     n_r = self.relation_emb(neg_r)
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
    #     if epoch == config.train_times - 1:
    #         self.save_phrt_o(pos_h, pos_r, pos_t, ph_o, pr_o, pt_o)
    #
    #     return p_score, n_score


def main():
    print('preparing online data...')
    new_model_parameter, affected_triples = config.prepare_online_data(config.dataset_v1_res_dir)
    print('preparing online data complete')

    device = torch.device('cuda')
    sample_size = config.sample_size
    test_triples = config.test_triples
    model_state_file = config.model_state_file

    print('train starting...')
    model = DKGE_Online().cuda()
    model.load_state_dict(new_model_parameter)

    # keep some parameters unchanged
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False
    model.relation_gcn_weight.requires_grad = False
    model.gate_entity.requires_grad = False
    model.gate_relation.requires_grad = False
    model.v_ent.requires_grad = False
    model.v_rel.requires_grad = False

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

    train_start_time = time.time()
    for epoch in range(config.train_times):
        epoch_start_time = time.time()
        print('----------training the ' + str(epoch) + ' epoch----------')
        model.train()
        optimizer.zero_grad()

        # sample from affected triples
        all_edges = np.arange(len(affected_triples))
        sample_index = np.random.choice(all_edges, sample_size, replace=False)
        sample_edges = affected_triples[sample_index]
        train_data = generate_graph(sample_edges, config.relation_total)
        train_data.to(device)

        entity_context, relation_context = model(train_data.entity, train_data.edge_index, train_data.edge_type,
                                                 train_data.edge_norm, train_data.DAD_rel)

        # calculation joint embedding
        head_o = torch.mul(torch.sigmoid(model.gate_entity), model.entity_emb(train_data.entity[train_data.samples[:, 0]].long())) + torch.mul(1 - torch.sigmoid(model.gate_entity), entity_context[train_data.samples[:, 0]])
        rel_o = torch.mul(torch.sigmoid(model.gate_relation), model.relation_emb(train_data.samples[:, 1])) + torch.mul(1 - torch.sigmoid(model.gate_relation), relation_context[train_data.samples[:, 1]])
        tail_o = torch.mul(torch.sigmoid(model.gate_entity), model.entity_emb(train_data.entity[train_data.samples[:, 2]].long())) + torch.mul(1 - torch.sigmoid(model.gate_entity), entity_context[train_data.samples[:, 2]])

        # score for loss
        p_scores = model._calc(head_o[0:train_data.samples.size()[0]//2], tail_o[0:train_data.samples.size()[0]//2], rel_o[0:train_data.samples.size()[0]//2])
        n_scores = model._calc(head_o[train_data.samples.size()[0]//2:], tail_o[train_data.samples.size()[0]//2:], rel_o[train_data.samples.size()[0]//2:])

        y = torch.Tensor([-1]*sample_size).cuda()
        loss = criterion(p_scores, n_scores, y)

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        epoch_end_time = time.time()
        print('----------epoch loss: ' + str(loss.item()) + ' ----------')
        print('----------epoch training time: ' + str(epoch_end_time - epoch_start_time) + ' s --------\n')

    print('train ending...')
    train_end_time = time.time()
    print('\n Total training time: ', train_end_time - train_start_time)

    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

    print('test link prediction starting...')
    checkpoint = torch.load(model_state_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_data = generate_graph(test_triples, config.relation_total)
    test_data.to(device)

    entity_context, relation_context = model(test_data.entity, test_data.edge_index, test_data.edge_type, test_data.edge_norm, test_data.DAD_rel)
    entity_embedding = model.entity_emb(torch.from_numpy(test_data.uniq_entity).long().cuda())
    relation_idx = torch.arange(config.relation_total).cuda()
    relation_embedding = model.relation_emb(relation_idx)
    entity_o = torch.mul(torch.sigmoid(model.gate_entity), entity_embedding) + torch.mul(1 - torch.sigmoid(model.gate_entity), entity_context)
    relation_o = torch.mul(torch.sigmoid(model.gate_relation), relation_embedding) + torch.mul(1 - torch.sigmoid(model.gate_relation), relation_context)
    # entity_emb, relation_emb = load_o_emb(config.res_dir, config.entity_total, config.relation_total, config.dim)

    print('test link prediction starts...')
    test.test_link_prediction(test_data.relabeled_edges, entity_o, relation_o, config.norm, test_data)
    print('test link prediction ends...')

    #     epoch_avg_loss = 0.0
    #     nbatchs = math.ceil(len(phs) / config.batch_size)
    #     for batch in range(nbatchs):
    #         optimizer.zero_grad()
    #         golden_triples, negative_triples = config.get_batch(config.batch_size, batch, epoch, phs, prs, pts, nhs, nrs, nts)
    #         ph_A, pr_A, pt_A = config.get_batch_A(golden_triples, config.entity_A, config.relation_A)
    #         nh_A, nr_A, nt_A = config.get_batch_A(negative_triples, config.entity_A, config.relation_A)
    #
    #         p_scores, n_scores = dynamicKGE(epoch, golden_triples, negative_triples, ph_A, pr_A, pt_A, nh_A, nr_A, nt_A)
    #         y = torch.Tensor([-1]).cuda()
    #         loss = criterion(p_scores, n_scores, y)
    #
    #         loss.backward()
    #         optimizer.step()
    #         torch.cuda.empty_cache()
    #
    #         epoch_avg_loss += (float(loss.item()) / nbatchs)
    #
    #         dynamicKGE.entity_emb.weight.data[ent_emb_nochange_list] = entity_emb[ent_emb_nochange_list]
    #         dynamicKGE.relation_emb.weight.data[rel_emb_nochange_list] = relation_emb[rel_emb_nochange_list]
    #         dynamicKGE.entity_context.weight.data[ent_context_emb_nochange_list] = entity_context[ent_context_emb_nochange_list]
    #         dynamicKGE.relation_context.weight.data[rel_context_emb_nochange_list] = relation_context[rel_context_emb_nochange_list]
    #
    #     epoch_end_time = time.time()
    #     print('----------epoch avg loss: ' + str(epoch_avg_loss) + ' ----------')
    #     print('----------epoch training time: ' + str(epoch_end_time - epoch_start_time) + ' s --------\n')
    # print('train ending...')
    # train_end_time = time.time()
    # print('\n Total training time: ', train_end_time-train_start_time)
    #
    # dynamicKGE.save_parameters(config.res_dir)
    #
    # entity_emb, relation_emb = config.load_o_emb(config.res_dir, config.entity_total, config.relation_total, config.dim)
    # print('test link prediction starting...')
    # test.test_link_prediction(config.test_list, set(config.train_list), entity_emb, relation_emb, config.norm)
    # print('test link prediction ending...')

if __name__ == "__main__":
    main()
