import math
import argparse

import torch
import torch.nn as nn
from collections import OrderedDict

from util.online_util import *
from util.parameter_util import *

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('-e', '--epochs', type=int, dest='train_epochs', help='total train epochs', required=False, default=21)
parser.add_argument('-b', '--batchsize', type=int, dest='batchsize', help='batch size', required=False, default=500)
parser.add_argument('-m', '--margin', type=float, dest='margin', help='margin', required=False, default=10.0)
parser.add_argument('-l', '--learning_rate', type=float, dest="learning_rate", help="learning rate", required=False, default=0.005)
parser.add_argument('-d', '--dimension', type=int, dest="dimension", help="dimension", required=False, default=100)
parser.add_argument('-n', '--norm', type=int, dest="norm", help="normalization", required=False, default=2)
parser.add_argument('-o', '--optim', type=str, dest="optim", help="optimizer", required=False, default="SGD")
parser.add_argument('-p1', '--path1', type=str, dest="dataset_v1", help="dataset_v1 path", required=False, default="YAGO-3SP/snapshot1")
parser.add_argument('-p2', '--path2', type=str, dest="dataset_v2", help="dataset_v2 path", required=False, default="YAGO-3SP/snapshot2")
parser.add_argument('-g', '--gpu', type=int, dest="gpu_id", help="select gpu", required=False, default=0)
parser.add_argument('-s', '--sample_size', type=int, dest="sample_size", help="sample size", required=False, default=4000)
# parser.add_argument('-t', '--test', type=int, dest="test_epoch", help="test epoch", required=False, default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

dataset_v1 = args.dataset_v1
dataset_v2 = args.dataset_v2

entity_total = get_total(file_name='./data/' + dataset_v2 + '/entity2id.txt')
relation_total = get_total(file_name='./data/' + dataset_v2 + '/relation2id.txt')
entity_set = set(range(entity_total))
relation_set = set(range(relation_total))

train_list = read_file(file_name='./data/' + dataset_v2 + '/train2id.txt')
valid_list = read_file(file_name='./data/' + dataset_v1 + '/valid2id.txt')
test_list = read_file(file_name='./data/' + dataset_v2 + '/test2id.txt')

train_triples = np.array(train_list)
valid_triples = np.array(valid_list)
test_triples = np.array(test_list)

print('entity_total: ' + str(entity_total))
print('relation_total: ' + str(relation_total))
print('train_total: ' + str(len(train_list)))
print('test_total: ' + str(len(test_list)))

max_context = 30
train_times = args.train_epochs
learning_rate = args.learning_rate
batch_size = args.batchsize
nbatchs = math.ceil(len(train_list) / batch_size)
margin = args.margin
dim = args.dimension
norm = args.norm
optimizer = args.optim
sample_size = args.sample_size
bern = True
res_dir = './data/' + dataset_v2 + '/parameters/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
dataset_v1_res_dir = './data/' + dataset_v1 + '/parameters/'
model_state_file = './data/' + dataset_v2 + '/parameters/OL_model_state.pth'

print('train_times: ' + str(train_times))
print('learning_rate: ' + str(learning_rate))
print('batch_size: ' + str(batch_size))
print('nbatchs: ' + str(nbatchs))
print('dim: ' + str(dim))
print('margin: ' + str(margin))
print('bern: ' + str(bern))
print('result directory: ' + str(res_dir))

# print("Constructing adj table...")
# entity_adj_table, relation_adj_table, max_context_num, entity_A, relation_A = construct_adj_table(train_list,
#                                                                                                   entity_total,
#                                                                                                   relation_total,
#                                                                                                   max_context)
# print("Constructing adj table completed.")


def prepare_online_data(parameter_path):
    print("Analysing snapshots...")
    affected_entities, affected_relations, affected_triples, added_entities, added_relations, entity_mapping_dict, relation_mapping_dict \
        = analyse_snapshots(dataset_v1, dataset_v2)
    print("Analyse snapshots completed.")

    # print("Negatvie sampling...")
    # phs = np.zeros(len(affected_triples), dtype=int)
    # prs = np.zeros(len(affected_triples), dtype=int)
    # pts = np.zeros(len(affected_triples), dtype=int)
    # nhs = np.zeros((train_times, len(affected_triples)), dtype=int)
    # nrs = np.zeros((train_times, len(affected_triples)), dtype=int)
    # nts = np.zeros((train_times, len(affected_triples)), dtype=int)
    #
    # tph, hpt = bern_sampling_prepare(train_list)
    # train_set = set(train_list)
    # for i, golden_triple in enumerate(affected_triples):
    #     # print(i, end='\r')
    #     phs[i], prs[i], pts[i] = golden_triple
    #
    #     for j in range(train_times):
    #         negative_triple = one_negative_sampling(golden_triple, train_set, entity_total, True, tph, hpt)
    #         nhs[j][i], nrs[j][i], nts[j][i] = negative_triple
    #
    # phs = torch.LongTensor(phs).cuda()
    # prs = torch.LongTensor(prs).cuda()
    # pts = torch.LongTensor(pts).cuda()
    # nhs = torch.LongTensor(nhs).cuda()
    # nrs = torch.LongTensor(nrs).cuda()
    # nts = torch.LongTensor(nts).cuda()
    # print("Negatvie sampling finished.")

    model_state_file = parameter_path + "LFS_model_state.pth"
    checkpoint = torch.load(model_state_file)
    old_model_parameter = checkpoint['state_dict']
    new_model_parameter = OrderedDict()
    for key in old_model_parameter.keys():
        new_model_parameter[key] = old_model_parameter[key].clone()

    new_model_parameter['entity_emb.weight'] = torch.zeros(entity_total, dim).cuda()
    new_model_parameter['entity_context.weight'] = torch.zeros(entity_total, dim).cuda()
    nn.init.xavier_uniform_(new_model_parameter['entity_emb.weight'])
    nn.init.xavier_uniform_(new_model_parameter['entity_context.weight'])
    for id1, id2 in entity_mapping_dict.items():
        new_model_parameter['entity_emb.weight'][id2] = old_model_parameter['entity_emb.weight'][id1]
        new_model_parameter['entity_context.weight'][id2] = old_model_parameter['entity_context.weight'][id1]

    new_model_parameter['relation_emb.weight'] = torch.zeros(relation_total, dim).cuda()
    new_model_parameter['relation_context.weight'] = torch.zeros(relation_total, dim).cuda()
    new_model_parameter['conv1.att'] = torch.zeros(relation_total*2, 4)
    new_model_parameter['conv2.att'] = torch.zeros(relation_total*2, 4)
    nn.init.xavier_uniform_(new_model_parameter['relation_emb.weight'])
    nn.init.xavier_uniform_(new_model_parameter['relation_context.weight'])
    nn.init.xavier_uniform_(new_model_parameter['conv1.att'])
    nn.init.xavier_uniform_(new_model_parameter['conv2.att'])
    relation_total_old = len(relation_mapping_dict.keys())
    for id1, id2 in relation_mapping_dict.items():
        new_model_parameter['relation_emb.weight'][id2] = old_model_parameter['relation_emb.weight'][id1]
        new_model_parameter['relation_context.weight'][id2] = old_model_parameter['relation_context.weight'][id1]
        new_model_parameter['conv1.att'][id2] = old_model_parameter['conv1.att'][id1]
        new_model_parameter['conv2.att'][id2] = old_model_parameter['conv2.att'][id1]
        new_model_parameter['conv1.att'][id2+relation_total] = old_model_parameter['conv1.att'][id1+relation_total_old]
        new_model_parameter['conv2.att'][id2+relation_total] = old_model_parameter['conv2.att'][id1+relation_total_old]

    return new_model_parameter, np.array(affected_triples)

    # entity_o_emb, relation_o_emb = load_o_emb(parameter_path, entity_total, relation_total, dim, input=True)
    #
    # new_entity_o_emb = torch.zeros(entity_total, dim).cuda()
    # nn.init.xavier_uniform_(new_entity_o_emb)
    # for id1, id2 in entity_mapping_dict.items():
    #     new_entity_o_emb[id2] = entity_o_emb[id1]
    #
    # new_relation_o_emb = torch.zeros(relation_total, dim).cuda()
    # nn.init.xavier_uniform_(new_relation_o_emb)
    # for id1, id2 in relation_mapping_dict.items():
    #     new_relation_o_emb[id2] = relation_o_emb[id1]
    #
    # return new_entity_emb, new_relation_emb, new_entity_context, new_relation_context, \
    #        entity_gcn_weight, relation_gcn_weight, gate_entity, gate_relation, v_entity, v_relation, \
    #        phs, prs, pts, nhs, nrs, nts, affected_entities, affected_relations, added_entities, added_relations, \
    #        new_entity_o_emb, new_relation_o_emb

