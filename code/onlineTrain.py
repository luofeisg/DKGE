import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from config import online_config as config
import test
from util.train_util import generate_graph_and_negative_sampling, generate_test_graph
from model import DKGE_Online

# gpu_ids = [0, 1]

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
        train_data = generate_graph_and_negative_sampling(sample_edges, config.relation_total)
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
    test.test_link_prediction(test_data.relabeled_edges, entity_o, relation_o, config.norm)
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
