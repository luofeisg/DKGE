from util.train_util import *


def get_affected_entities(entity_set1, entity_set2, entity_context_dict1, entity_context_dict2):
    affected_entities_set = set()
    for entity in entity_set2:
        if entity not in entity_set1:
            affected_entities_set.add(entity)
            continue
        if entity_context_dict1[entity] != entity_context_dict2[entity]:
            affected_entities_set.add(entity)

    return affected_entities_set


def get_affected_relations(relation_set1, relation_set2, relation_context_dict1, relation_context_dict2):
    affected_relations_set = set()
    for relation in relation_set2:
        if relation not in relation_set1:
            affected_relations_set.add(relation)
            continue
        if relation_context_dict1[relation] != relation_context_dict2[relation]:
            affected_relations_set.add(relation)

    return affected_relations_set

def analyse_affected_data(dataset_v1, dataset_v2):
    snapshot1_train_list = read_file(file_name='./data/' + dataset_v1 + '/train2id.txt')
    snapshot2_train_list = read_file(file_name='./data/' + dataset_v2 + '/train2id.txt')

    snapshot1_train_set = set(convert_id_to_text(snapshot1_train_list, dataset_v1))
    snapshot2_train_set = set(convert_id_to_text(snapshot2_train_list, dataset_v2))

    entity_set1, relation_set1, entity_context_dict1, relation_context_dict1 = get_basic_info(snapshot1_train_set)
    entity_set2, relation_set2, entity_context_dict2, relation_context_dict2 = get_basic_info(snapshot2_train_set)

    affected_entities = get_affected_entities(entity_set1, entity_set2, entity_context_dict1, entity_context_dict2)
    affected_relations = get_affected_relations(relation_set1, relation_set2, relation_context_dict1, relation_context_dict2)

    new_entities = entity_set2 - entity_set1
    new_relations = relation_set2 - relation_set1

    affected_triples = list()
    for (h, r, t) in snapshot2_train_set:
        if h in affected_entities or t in affected_entities or r in new_relations:
            affected_triples.append((h, r, t))

    return affected_entities, affected_relations, affected_triples


def construct_snapshots_mapping_dict(dataset_v1, dataset_v2):
    snapshot1_entity2id_dict = construct_text2id_dict(file_name='./data/' + dataset_v1 + '/entity2id.txt')
    snapshot1_relation2id_dict = construct_text2id_dict(file_name='./data/' + dataset_v1 + '/relation2id.txt')

    snapshot2_entity2id_dict = construct_text2id_dict(file_name='./data/' + dataset_v2 + '/entity2id.txt')
    snapshot2_relation2id_dict = construct_text2id_dict(file_name='./data/' + dataset_v2 + '/relation2id.txt')

    entity_mapping_dict = dict()
    for entity, id in snapshot1_entity2id_dict.items():
        if entity in snapshot2_entity2id_dict:
            entity_mapping_dict[id] = snapshot2_entity2id_dict[entity]

    relation_mapping_dict = dict()
    for relation, id in snapshot1_relation2id_dict.items():
        if relation in snapshot2_relation2id_dict:
            relation_mapping_dict[id] = snapshot2_relation2id_dict[relation]

    return snapshot1_entity2id_dict, snapshot1_relation2id_dict, snapshot2_entity2id_dict, snapshot2_relation2id_dict, \
        entity_mapping_dict, relation_mapping_dict


def analyse_snapshots(dataset_v1, dataset_v2):
    snapshot1_entity2id_dict, snapshot1_relation2id_dict, snapshot2_entity2id_dict, snapshot2_relation2id_dict, \
     entity_mapping_dict, relation_mapping_dict = construct_snapshots_mapping_dict(dataset_v1, dataset_v2)

    added_entities = set()
    for entity, id in snapshot2_entity2id_dict.items():
        if entity not in snapshot1_entity2id_dict:
            added_entities.add(id)

    added_relations = set()
    for relation, id in snapshot2_relation2id_dict.items():
        if relation not in snapshot1_relation2id_dict:
            added_relations.add(id)

    affected_entities, affected_relations, affected_triples = analyse_affected_data(dataset_v1, dataset_v2)

    affected_entities = set([snapshot2_entity2id_dict[e] for e in affected_entities])
    affected_relations = set([snapshot2_relation2id_dict[r] for r in affected_relations])
    affected_triples = [(snapshot2_entity2id_dict[h], snapshot2_relation2id_dict[r], snapshot2_entity2id_dict[t]) for (h, r, t) in affected_triples]

    print("affected entities: %d" % len(affected_entities))
    print("affected relations: %d" % len(affected_relations))
    print("affected triples: %d" % len(affected_triples))
    print("added entities: %d" % len(added_entities))
    print("addded relations: %d" % len(added_relations))

    return affected_entities, affected_relations, affected_triples, added_entities, added_relations, entity_mapping_dict, relation_mapping_dict
