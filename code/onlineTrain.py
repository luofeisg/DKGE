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
    new_model_parameter, affected_triples, affected_entities, affected_relations = config.prepare_online_data(config.dataset_v1_res_dir)
    print('preparing online data complete')

    num_entities = config.entity_total
    num_relations = config.relation_total
    sample_size = config.sample_size
    train_triples = config.train_triples
    valid_triples = config.valid_triples
    test_triples = config.test_triples
    validate_every = config.validate_every
    model_state_file = config.model_state_file

    device_cuda = torch.device('cuda')
    device_cpu = torch.device('cpu')
    best_mrr = 0
    best_mrr_epoch = 0

    print('train starting...')
    model = DKGE_Online(num_entities, num_relations, config.dim, config.norm).cuda()
    model.load_state_dict(new_model_parameter)

    # keep some parameters unchanged
    for param in model.conv1_entity.parameters():
        param.requires_grad = False
    for param in model.conv2_entity.parameters():
        param.requires_grad = False
    model.relation_gcn_weight.requires_grad = False
    model.gate_entity.requires_grad = False
    model.gate_relation.requires_grad = False

    if config.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    criterion = nn.MarginRankingLoss(config.margin, reduction='sum').cuda()

    train_start_time = time.time()
    for epoch in range(1, config.train_times+1):
        epoch_start_time = time.time()
        print('----------training the ' + str(epoch) + ' epoch----------')
        model.train()
        optimizer.zero_grad()

        # sample from affected triples
        sample_index = np.random.choice(len(affected_triples), sample_size, replace=False)
        sample_edges = affected_triples[sample_index]
        train_data = generate_graph_and_negative_sampling(sample_edges, num_relations)
        train_data.to(device_cuda)

        # entity_o, relation_o = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm, train_data.DAD_rel, affected_entities, affected_relations)
        entity_o, relation_o = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm, train_data.DAD_rel)
        loss = model.score_loss(entity_o, relation_o, train_data.samples, train_data.labels) + 0.01 * model.reg_loss(entity_o, relation_o)

        # entity_context, relation_context = model(train_data.entity, train_data.edge_index, train_data.edge_type,
        #                                          train_data.edge_norm, train_data.DAD_rel)


        # # score for loss
        # p_scores = model._calc(head_o[0:train_data.samples.size()[0]//2], tail_o[0:train_data.samples.size()[0]//2], rel_o[0:train_data.samples.size()[0]//2])
        # n_scores = model._calc(head_o[train_data.samples.size()[0]//2:], tail_o[train_data.samples.size()[0]//2:], rel_o[train_data.samples.size()[0]//2:])

        y = torch.Tensor([-1]*sample_size).cuda()
        loss = criterion(loss[:len(loss)//2], loss[len(loss)//2:], y)

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        epoch_end_time = time.time()
        print('----------epoch loss: ' + str(loss.item()) + ' ----------')
        print('----------epoch training time: ' + str(epoch_end_time - epoch_start_time) + ' s --------\n')

        # validation
        if epoch % validate_every == 0:
            model.eval()
            with torch.no_grad():
                test_graph = generate_graph_and_negative_sampling(train_triples, config.relation_total)
                test_graph.to(device_cuda)
                entity_o, relation_o = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type,
                                             test_graph.edge_norm, test_graph.DAD_rel)

                print('validate link prediction on train set starts...')
                index = np.random.choice(train_triples.shape[0], 100)
                test.test_link_prediction(train_triples[index], entity_o, relation_o, config.norm)
                print('valid link prediction on train set ends...')

                print('validation on validation set starts...')
                mrr = test.test_link_prediction(valid_triples[0:100], entity_o, relation_o, config.norm)
                print('validation on validation set ends...')

                if mrr > best_mrr:
                    print("better mrr at epoch {}, mrr: {}, best mrr before: {}".format(epoch, mrr, best_mrr))
                    best_mrr = mrr
                    best_mrr_epoch = epoch
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                    print("model at epoch {} saved".format(epoch))
    if best_mrr_epoch <= 0:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        print("model at epoch {} saved".format(epoch))

    print('train ending...')
    train_end_time = time.time()
    print('\nTotal training time: ', train_end_time-train_start_time)
    print("best parameter at epoch {}, mrr: {}".format(best_mrr_epoch, best_mrr))

    print('prepare test data...')
    checkpoint = torch.load(model_state_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        test_graph = generate_graph_and_negative_sampling(train_triples, config.relation_total)
        test_graph.to(device_cuda)
        entity_o, relation_o = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type,
                                     test_graph.edge_norm, test_graph.DAD_rel)

        print('test link prediction on train set starts...')
        index = np.random.choice(train_triples.shape[0], 100)
        test.test_link_prediction(train_triples[index], entity_o, relation_o, config.norm)
        print('test link prediction on train set ends...')

        print('test link prediction on test set starts...')
        test.test_link_prediction(test_triples[0:100], entity_o, relation_o, config.norm)
        print('test link prediction on test set ends...')

if __name__ == "__main__":
    main()
