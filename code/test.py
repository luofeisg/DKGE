import torch
import os
from util.test_util import *


def _calc(h, t, r, norm):
    return torch.norm(h + r - t, p=norm, dim=1).cpu().numpy().tolist()


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


def test_head(golden_triple, train_set, entity_emb, relation_emb, norm):
    head_batch = get_head_batch(golden_triple, len(entity_emb))
    value = predict(head_batch, entity_emb, relation_emb, norm)
    golden_value = value[golden_triple[0]]
    # li = np.argsort(value)
    res = 1
    sub = 0
    for pos, val in enumerate(value):
        if val < golden_value:
            res += 1
            if (pos, golden_triple[1], golden_triple[2]) in train_set:
                sub += 1

    return res, res - sub


def test_tail(golden_triple, train_set, entity_emb, relation_emb, norm):
    tail_batch = get_tail_batch(golden_triple, len(entity_emb))
    value = predict(tail_batch, entity_emb, relation_emb, norm)
    golden_value = value[golden_triple[2]]
    # li = np.argsort(value)
    res = 1
    sub = 0
    for pos, val in enumerate(value):
        if val < golden_value:
            res += 1
            if (golden_triple[0], golden_triple[1], pos) in train_set:
                sub += 1

    return res, res - sub


def test_link_prediction(test_list, train_set, entity_emb, relation_emb, norm):
    test_total = len(test_list)

    l_mr = 0
    r_mr = 0

    l_mr_filter = 0
    r_mr_filter = 0

    l_hit1 = 0
    r_hit1 = 0
    l_hit3 = 0
    r_hit3 = 0
    l_hit10 = 0
    r_hit10 = 0

    for i, golden_triple in enumerate(test_list):
        print('test ---' + str(i) + '--- triple')
        print(i, end="\r")
        l_pos, l_filter_pos = test_head(golden_triple, train_set, entity_emb, relation_emb, norm)
        r_pos, r_filter_pos = test_tail(golden_triple, train_set, entity_emb, relation_emb, norm)  # position, 1-based

        print(golden_triple, end=': ')
        print('l_pos=' + str(l_pos), end=', ')
        print('l_filter_pos=' + str(l_filter_pos), end=', ')
        print('r_pos=' + str(r_pos), end=', ')
        print('r_filter_pos=' + str(r_filter_pos), end='\n')

        l_mr += l_pos
        r_mr += r_pos

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

        l_mr_filter += l_filter_pos
        r_mr_filter += r_filter_pos

    l_mr /= test_total
    r_mr /= test_total

    l_hit1_ratio = float(l_hit1)/test_total
    l_hit3_ratio = float(l_hit3)/test_total
    l_hit10_ratio = float(l_hit10)/test_total
    r_hit1_ratio = float(r_hit1)/test_total
    r_hit3_ratio = float(r_hit3)/test_total
    r_hit10_ratio = float(r_hit10)/test_total

    l_mr_filter /= test_total
    r_mr_filter /= test_total

    print('\t\t\t\t\tmean_rank\t\t\thit@1,3,10')
    print('head(raw)\t\t\t%.3f\t\t\t%.3f\t%.3f\t%.3f' % (l_mr, l_hit1_ratio, l_hit3_ratio, l_hit10_ratio))
    print('tail(raw)\t\t\t%.3f\t\t\t%.3f\t%.3f\t%.3f' % (r_mr, r_hit1_ratio, r_hit3_ratio, r_hit10_ratio))
    print('average(raw)\t\t\t%.3f\t\t\t' % ((l_mr + r_mr) / 2))

    print('head(filter)\t\t\t%.3f\t\t\t' % l_mr_filter)
    print('tail(filter)\t\t\t%.3f\t\t\t' % r_mr_filter)
    print('average(filter)\t\t\t%.3f\t\t\t' % ((l_mr_filter + r_mr_filter) / 2))


if __name__ == "__main__":
    from config import config
    from util.parameter_util import load_o_emb

    # entity_emb, relation_emb = load_o_emb(config.res_dir, config.entity_total, config.relation_total, config.dim)
    print('test link prediction starting...')
    checkpoint = torch.load(config.model_state_file)
    state_dict = checkpoint['state_dict']
    entity_o = state_dict['entity_o']
    relation_o = state_dict['relation_o']
    # entity_emb, relation_emb = load_o_emb(config.res_dir, config.entity_total, config.relation_total, config.dim)

    test_link_prediction(config.test_list, set(config.train_list), entity_o, relation_o, config.norm)
    print('test link prediction ending...')