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
    # h = entity_o[triplets[:, 0]]
    # r = relation_o[triplets[:, 1]]
    # t = entity_o[triplets[:, 2]]
    # score = torch.norm(h + r - t, p=norm, dim=1)

    score = distmult(entity_o, relation_o, triplets)

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

    res = torch.count_nonzero(scores > golden_score, axis=0) + 1
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

    res = torch.count_nonzero(scores > golden_score, axis=0) + 1
    return res


def test_link_prediction(test_triples, entity_o, relation_o, norm):
    test_total = len(test_triples)

    l_mr = 0
    r_mr = 0
    l_mrr = 0.0
    r_mrr = 0.0

    l_mr_filter = 0
    r_mr_filter = 0

    l_hit1 = 0
    r_hit1 = 0
    l_hit3 = 0
    r_hit3 = 0
    l_hit10 = 0
    r_hit10 = 0

    for i, golden_triple in enumerate(test_triples):
        # print('test ---' + str(i) + '--- triple')
        # print(i, end="\r")
        l_pos = test_head(golden_triple, entity_o, relation_o, norm)
        r_pos = test_tail(golden_triple, entity_o, relation_o, norm)  # position, 1-based

        # print(golden_triple, end=': ')
        # print('l_pos=' + str(l_pos), end=', ')
        # print('l_filter_pos=' + str(l_filter_pos), end=', ')
        # print('r_pos=' + str(r_pos), end=', ')
        # print('r_filter_pos=' + str(r_filter_pos), end='\n')

        l_mr += l_pos
        r_mr += r_pos
        l_mrr += 1/l_pos
        r_mrr += 1/r_pos

        if l_pos <= 10:
            l_hit10 += 1
            if l_pos <= 3:
                l_hit3 += 1
                if l_pos == 1:
                    l_hit1 += 1

        if r_pos <= 10:
            r_hit10 += 1
            if r_pos <= 3:
                r_hit3 += 1
                if r_pos == 1:
                    r_hit1 += 1

        # l_mr_filter += l_filter_pos
        # r_mr_filter += r_filter_pos

    l_mr = float(l_mr)/test_total
    r_mr = float(r_mr) / test_total
    l_mrr = float(l_mrr) / test_total
    r_mrr = float(r_mrr) / test_total

    l_hit1_ratio = float(l_hit1)/test_total
    l_hit3_ratio = float(l_hit3)/test_total
    l_hit10_ratio = float(l_hit10)/test_total
    r_hit1_ratio = float(r_hit1)/test_total
    r_hit3_ratio = float(r_hit3)/test_total
    r_hit10_ratio = float(r_hit10)/test_total

    l_mr_filter /= test_total
    r_mr_filter /= test_total

    print('\t\t\t\t\tMR\t\t\t\tMRR\t\t\t\tHit@1,3,10')
    print('head(raw)\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t%.3f\t%.3f' % (l_mr, l_mrr, l_hit1_ratio, l_hit3_ratio, l_hit10_ratio))
    print('tail(raw)\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t%.3f\t%.3f' % (r_mr, r_mrr, r_hit1_ratio, r_hit3_ratio, r_hit10_ratio))
    print('average(raw)\t\t%.3f\t\t\t%.3f\t\t\t%.3f\t%.3f\t%.3f' % ((l_mr + r_mr) / 2, (l_mrr + r_mrr) / 2, (l_hit1_ratio + r_hit1_ratio) / 2, (l_hit3_ratio + r_hit3_ratio) / 2, (l_hit10_ratio + r_hit10_ratio) / 2 ))

    # print('head(filter)\t\t%.3f\t\t\t' % l_mr_filter)
    # print('tail(filter)\t\t%.3f\t\t\t' % r_mr_filter)
    # print('average(filter)\t\t%.3f\t\t\t' % ((l_mr_filter + r_mr_filter) / 2))

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


