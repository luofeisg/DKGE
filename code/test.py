import torch
import os
from util.test_util import *
import time


def distmult(entity_o, relation_o, triplets):
    s = entity_o[triplets[:, 0]]
    r = relation_o[triplets[:, 1]]
    o = entity_o[triplets[:, 2]]
    score = torch.sum(s * r * o, dim=1)

    return score

def cal_score(entity_o, relation_o, triplets, norm):
    h = entity_o[triplets[:, 0]]
    r = relation_o[triplets[:, 1]]
    t = entity_o[triplets[:, 2]]
    score = torch.norm(h + r - t, p=norm, dim=1)

    # score = distmult(entity_o, relation_o, triplets)

    return score

def _calc(h, t, r, norm):
    return torch.norm(h + r - t, p=norm, dim=1).cpu().detach().numpy().tolist()


def predict(batch, entity_emb, relation_emb, norm):
    pos_hs = batch[:, 0]
    pos_rs = batch[:, 1]
    pos_ts = batch[:, 2]

    pos_hs = torch.IntTensor(pos_hs).cuda()
    pos_rs = torch.IntTensor(pos_rs).cuda()
    pos_ts = torch.IntTensor(pos_ts).cuda()

    p_score = _calc(entity_emb[pos_hs.type(torch.long)],
                    entity_emb[pos_ts.type(torch.long)],
                    relation_emb[pos_rs.type(torch.long)],
                    norm)

    return p_score

def get_rank(scores, golden_score):
    res0 = torch.count_nonzero(scores < golden_score, axis=0) + 1
    # use numpy due to lower version of pytorch
    scores = scores.cpu().numpy()
    golden_score = golden_score.item()
    res = np.count_nonzero(scores < golden_score, axis=0) + 1
    return res

def test_head(golden_triple, entity_emb, relation_emb, norm):
    head_batch = get_head_batch(golden_triple, len(entity_emb))
    scores = cal_score(entity_emb, relation_emb, head_batch, norm)
    golden_score = scores[golden_triple[0]]
    # value = predict(head_batch, entity_emb, relation_emb, norm)
    # golden_value = value[golden_triple[0]]
    # li = np.argsort(value)
    # res = 1
    # sub = 0
    # for pos, val in enumerate(scores):
    #     if val < golden_score:
    #         res += 1
    #         # if (pos, golden_triple[1], golden_triple[2]) in train_set:
    #         #     sub += 1

    res = get_rank(scores, golden_score)
    return res
    # return res, res - sub


def test_tail(golden_triple, entity_emb, relation_emb, norm):
    tail_batch = get_tail_batch(golden_triple, len(entity_emb))
    scores = cal_score(entity_emb, relation_emb, tail_batch, norm)
    golden_score = scores[golden_triple[2]]
    # value = predict(tail_batch, entity_emb, relation_emb, norm)
    # golden_value = value[golden_triple[2]]
    # li = np.argsort(value)
    # res = 1
    # sub = 0
    # for pos, val in enumerate(scores):
    #     if val < golden_score:
    #         res += 1
    #         # if (golden_triple[0], golden_triple[1], pos) in train_set:
    #         #     sub += 1

    res = get_rank(scores, golden_score)
    return res


def test_link_prediction(test_triples, entity_o, relation_o, norm):
    hits = [1, 3, 10]

    l_rank = []
    r_rank = []

    for i, golden_triple in enumerate(test_triples):
        l_pos = test_head(golden_triple, entity_o, relation_o, norm)
        r_pos = test_tail(golden_triple, entity_o, relation_o, norm)  # position, 1-based

        l_rank.append(l_pos)
        r_rank.append(r_pos)

    l_rank = np.array(l_rank)
    r_rank = np.array(r_rank)

    l_mr = np.mean(l_rank)
    r_mr = np.mean(r_rank)
    l_mrr = np.mean(1. / l_rank)
    r_mrr = np.mean(1. / r_rank)

    l_hit1_ratio = np.mean(l_rank <= 1)
    l_hit3_ratio = np.mean(l_rank <= 3)
    l_hit10_ratio = np.mean(l_rank <= 10)
    r_hit1_ratio = np.mean(r_rank <= 1)
    r_hit3_ratio = np.mean(r_rank <= 3)
    r_hit10_ratio = np.mean(r_rank <= 10)

    print('\t\t\t\t\tMR\t\t\t\tMRR\t\t\t\tHit@1,3,10')
    print('head(raw)\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t%.3f\t%.3f' % (l_mr, l_mrr, l_hit1_ratio, l_hit3_ratio, l_hit10_ratio))
    print('tail(raw)\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t%.3f\t%.3f' % (r_mr, r_mrr, r_hit1_ratio, r_hit3_ratio, r_hit10_ratio))
    print('average(raw)\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t%.3f\t%.3f' % ((l_mr + r_mr) / 2, (l_mrr + r_mrr) / 2, (l_hit1_ratio + r_hit1_ratio) / 2, (l_hit3_ratio + r_hit3_ratio) / 2, (l_hit10_ratio + r_hit10_ratio) / 2))

    return (l_mrr + r_mrr) / 2


if __name__ == "__main__":
    online = False
    device = torch.device('cuda')
    with torch.no_grad():
        if not online:
            # from config import config
            from train import *
            from util.train_util import *

            print('prepare test data...')
            model = DynamicKGE(config).cuda()
            checkpoint = torch.load(config.model_state_file)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()

            train_graph = generate_graph(config.train_triples, config.relation_total)
            train_graph.to(device)
            entity_o, relation_o = model.forward(train_graph.entity, train_graph.edge_index, train_graph.edge_type, train_graph.edge_norm, train_graph.DAD_rel)

            test_link_prediction(config.test_triples, entity_o, relation_o, config.norm)

            # test_data = generate_graph(config.test_triples, config.relation_total)
            # test_data.to(device)

            # entity_context, relation_context = model(test_data.entity, test_data.edge_index, test_data.edge_type,
            #                                          test_data.edge_norm, test_data.DAD_rel)
            # entity_embedding = model.entity_emb(torch.from_numpy(test_data.uniq_entity).long().cuda())
            # relation_embedding = model.relation_emb.weight
            # entity_o = torch.mul(torch.sigmoid(model.gate_entity), entity_embedding) + torch.mul(
            #     1 - torch.sigmoid(model.gate_entity), entity_context)
            # relation_o = torch.mul(torch.sigmoid(model.gate_relation), relation_embedding) + torch.mul(
            #     1 - torch.sigmoid(model.gate_relation), relation_context)
            #
            # test_link_prediction(test_data.relabeled_edges, entity_o, relation_o, config.norm, test_data)
            print('test link prediction ending...')

        if online:
            from onlineTrain import *

            print('prepare test data...')
            model = DKGE_Online().cuda()
            checkpoint = torch.load(config.model_state_file)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()

            test_data = generate_graph(config.test_triples, config.relation_total)
            test_data.to(device)

            entity_context, relation_context = model(test_data.entity, test_data.edge_index, test_data.edge_type,
                                                     test_data.edge_norm, test_data.DAD_rel)
            entity_embedding = model.entity_emb(torch.from_numpy(test_data.uniq_entity).long().cuda())
            relation_idx = torch.arange(config.relation_total).cuda()
            relation_embedding = model.relation_emb(relation_idx)
            entity_o = torch.mul(torch.sigmoid(model.gate_entity), entity_embedding) + torch.mul(
                1 - torch.sigmoid(model.gate_entity), entity_context)
            relation_o = torch.mul(torch.sigmoid(model.gate_relation), relation_embedding) + torch.mul(
                1 - torch.sigmoid(model.gate_relation), relation_context)

            test_link_prediction(test_data.relabeled_edges, entity_o, relation_o, config.norm, test_data)
            print('test link prediction ending...')


