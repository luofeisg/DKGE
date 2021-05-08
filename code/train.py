import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import test
from config import config
from util.parameter_util import *
from util.train_util import *

# gpu_ids = [0, 1]

class DynamicKGE(nn.Module):
    def __init__(self, config):
        super(DynamicKGE, self).__init__()

        self.entity_emb = nn.Embedding(config.entity_total, config.dim)
        self.relation_emb = nn.Parameter(torch.Tensor(config.relation_total, config.dim))
        nn.init.xavier_uniform_(self.relation_emb, gain=nn.init.calculate_gain('relu'))
        # self.relation_emb = nn.Embedding(config.relation_total, config.dim)

        self.entity_context = nn.Embedding(config.entity_total, config.dim)
        self.relation_context = nn.Embedding(config.relation_total, config.dim)
        # self.entity_context = nn.Embedding(config.entity_total + 1, config.dim, padding_idx=config.entity_total)
        # self.relation_context = nn.Embedding(config.relation_total + 1, config.dim, padding_idx=config.relation_total)

        # self.entity_gcn_weight = nn.Parameter(torch.Tensor(config.dim, config.dim))
        self.relation_gcn_weight = nn.Parameter(torch.Tensor(config.dim, config.dim))

        self.gate_entity = nn.Parameter(torch.Tensor(config.dim))
        self.gate_relation = nn.Parameter(torch.Tensor(config.dim))

        self.conv1 = RGCNConv(config.dim, config.dim, config.relation_total * 2, num_bases=4)
        self.conv2 = RGCNConv(config.dim, config.dim, config.relation_total * 2, num_bases=4)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.uniform_(self.gate_entity.data)
        nn.init.uniform_(self.gate_relation.data)

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

    def distmult(self, entity_o, relation_o, triplets):
        s = entity_o[triplets[:, 0]]
        r = relation_o[triplets[:, 1]]
        o = entity_o[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)

        return score

    def score_loss(self, entity_o, relation_o, triplets, target):
        # h = entity_o[triplets[:, 0]]
        # r = relation_o[triplets[:, 1]]
        # t = entity_o[triplets[:, 2]]

        score = self.distmult(entity_o, relation_o, triplets)
        return F.binary_cross_entropy_with_logits(score, target)
        # score = torch.norm(h + r - t, p=config.norm, dim=1)
        # return score

    def reg_loss(self, entity_o, relation_o):
        return torch.mean(entity_o.pow(2)) + torch.mean(relation_o.pow(2))

    def forward(self, entity, edge_index, edge_type, edge_norm, DAD_rel):
        entity_emb = self.entity_emb(entity.long())
        relation_emb = self.relation_emb

        entity_emb = F.relu(self.conv1(entity_emb, edge_index, edge_type, edge_norm))
        entity_emb = F.dropout(entity_emb, p=0.2, training=self.training)
        entity_emb = self.conv2(entity_emb, edge_index, edge_type, edge_norm)

        return entity_emb, relation_emb


        # entity_emb = self.entity_emb(entity.long())
        # relation_emb = self.relation_emb.weight
        # entity_context = self.entity_context(entity.long())
        # relation_context = self.relation_context.weight
        #
        # # rgcn
        # entity_context = F.relu(self.conv1(entity_context, edge_index, edge_type, edge_norm))
        # # entity_context = F.dropout(entity_context, p=0.2, training=self.training)
        # entity_context = self.conv2(entity_context, edge_index, edge_type, edge_norm)
        # # gcn
        # relation_context = torch.matmul(DAD_rel, relation_context)
        # relation_context = torch.matmul(relation_context, self.relation_gcn_weight)
        #
        # # calculate joint embedding
        # entity_o = torch.mul(torch.sigmoid(self.gate_entity), entity_emb) + torch.mul(1 - torch.sigmoid(self.gate_entity), entity_context)
        # relation_o = torch.mul(torch.sigmoid(self.gate_relation), relation_emb) + torch.mul(1 - torch.sigmoid(self.gate_entity), relation_context)
        #
        # return entity_o, relation_o

def main():
    train_triples = config.train_triples
    valid_triples = config.valid_triples
    test_triples = config.test_triples
    sample_size = config.sample_size
    validate_every = config.validate_every
    negative_rate = 1
    num_rels = config.relation_total
    model_state_file = config.model_state_file

    train_graph = generate_graph(train_triples, config.relation_total)
    device_cuda = torch.device('cuda')
    device_cpu = torch.device('cpu')

    best_mrr = 0
    best_mrr_epoch = 0

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

    train_start_time = time.time()
    for epoch in range(1, config.train_times+1):
        epoch_start_time = time.time()
        print('----------training the ' + str(epoch) + ' epoch----------')
        model.train()
        optimizer.zero_grad()

        # sample from whole graph
        sample_index = np.random.choice(len(train_triples), sample_size, replace=False)
        sample_edges = train_triples[sample_index]
        train_data = generate_graph(sample_edges, config.relation_total)

        train_data.to(device_cuda)

        entity_o, relation_o = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm, train_data.DAD_rel)
        loss = model.score_loss(entity_o, relation_o, train_data.samples, train_data.labels) + 0.01 * model.reg_loss(entity_o, relation_o)

        # # score for loss
        # p_scores = model._calc(head_o[0:train_data.samples.size()[0]//2], tail_o[0:train_data.samples.size()[0]//2], rel_o[0:train_data.samples.size()[0]//2])
        # n_scores = model._calc(head_o[train_data.samples.size()[0]//2:], tail_o[train_data.samples.size()[0]//2:], rel_o[train_data.samples.size()[0]//2:])

        # y = torch.Tensor([-1]*sample_size).cuda()
        # loss = criterion(score[:len(score)//2], score[len(score)//2:], y)

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        epoch_end_time = time.time()
        print('----------epoch loss: ' + str(loss.item()) + ' ----------')
        print('----------epoch training time: ' + str(epoch_end_time-epoch_start_time) + ' s --------\n')

        # validation
        if epoch % validate_every == 0:
            model.eval()
            with torch.no_grad():
                # train_graph.to(device_cuda)
                # entity_o, relation_o = model.forward(train_graph.entity, train_graph.edge_index, train_graph.edge_type,
                #                                      train_graph.edge_norm, train_graph.DAD_rel)
                # train_graph.to(device_cpu)

                train_graph = generate_graph(train_triples, config.relation_total)
                train_graph.to(device_cuda)
                entity_o, relation_o = model(train_graph.entity, train_graph.edge_index, train_graph.edge_type,
                                                     train_graph.edge_norm, train_graph.DAD_rel)

                print('validate link prediction on train set starts...')
                index = np.random.choice(train_triples.shape[0], 1000)
                mrr = test.test_link_prediction(train_triples[index], entity_o, relation_o, config.norm)
                print('valid link prediction on train set ends...')

                print('validation on validation set starts...')
                mrr = test.test_link_prediction(valid_triples, entity_o, relation_o, config.norm)
                print('validation on validation set ends...')

                if mrr > best_mrr:
                    best_mrr_epoch = epoch
                    print("better mrr at epoch {}, mrr: {}, best mrr before: {}".format(epoch, mrr, best_mrr))
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

    print('train ending...')
    train_end_time = time.time()
    print('\nTotal training time: ', train_end_time-train_start_time)

    print('prepare test data...')
    checkpoint = torch.load(model_state_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        # train_graph.to(device_cuda)
        # entity_o, relation_o = model.forward(train_graph.entity, train_graph.edge_index, train_graph.edge_type,
        #                                                  train_graph.edge_norm, train_graph.DAD_rel)
        # train_graph.to(device_cpu)

        entity_o = model.entity_emb.weight.data
        relation_o = model.relation_emb.weight.data

        print('test link prediction on train set starts...')
        index = np.random.choice(train_triples.shape[0], 1000)
        test.test_link_prediction(train_triples[index], entity_o, relation_o, config.norm)
        print('test link prediction on train set ends...')

        print('test link prediction on test set starts...')
        test.test_link_prediction(test_triples, entity_o, relation_o, config.norm)
        print('test link prediction on test set ends...')

    # test_data = generate_graph(test_triples, config.relation_total)
    # test_data.to(device)
    #
    # entity_context, relation_context = model(test_data.entity, test_data.edge_index, test_data.edge_type, test_data.edge_norm, test_data.DAD_rel)
    # entity_embedding = model.entity_emb(torch.from_numpy(test_data.uniq_entity).long().cuda())
    # relation_idx = torch.arange(config.relation_total).cuda()
    # relation_embedding = model.relation_emb(relation_idx)
    # entity_o = torch.mul(torch.sigmoid(model.gate_entity), entity_embedding) + torch.mul(1 - torch.sigmoid(model.gate_entity), entity_context)
    # relation_o = torch.mul(torch.sigmoid(model.gate_relation), relation_embedding) + torch.mul(1 - torch.sigmoid(model.gate_relation), relation_context)
    # # entity_emb, relation_emb = load_o_emb(config.res_dir, config.entity_total, config.relation_total, config.dim)
    #
    # print('test link prediction starts...')
    # test.test_link_prediction(test_data.relabeled_edges, entity_o, relation_o, config.norm)
    # print('test link prediction ends...')

if __name__ == "__main__":
    main()
