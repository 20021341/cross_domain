import codecs
import copy
import os
import random


def read_dataset(file_name):
    data = {}
    with open(file_name,"r",encoding="utf-8") as fr:
        for line in fr:
            user, item, score = line.strip().split("\t")
            if user not in data.keys():
                data[user] = [item]
            else :
                data[user].append(item)
    return data

def create_user_dict(user, data):
    user = copy.deepcopy(user)
    for u in data.keys():
        if u not in user.keys():
            user[u] = len(user)
    return user

def create_item_dict(item, data):
    for u in data.keys():
        for i in data[u]:
            if i not in item.keys():
                item[i] = len(item)
    return item


def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

def generate_train_valid_test(file, total, common, user, item, choose):
    train_file = file + "train.txt"
    valid_file = file + "valid.txt"
    test_file = file + "test.txt"

    #source_total_data, source_common_data, source_user, source_item, 0
    ## total: dict user total: list[item]
    ## common: dict user common: list[item]
    ## user: user mapping dict from index in file reindex.txt to index of that domain only
    ## item: same
    ## chose: 0 is source data, 1 is target data

    start_user_id = int(len(common) * 0.8)
    
    with codecs.open(train_file,"w",encoding="utf-8") as fw:
        with codecs.open(test_file, "w", encoding="utf-8") as fw2:
            with codecs.open(valid_file, "w", encoding="utf-8") as fw3:
                for user_old_id in total: #da is the old user index from file reindex.txt
                    if user[user_old_id] in range(start_user_id, len(common) + 1): 
                        item_old_ids = total[user_old_id]
                        item_old_ids = random.shuffle(item_old_ids)

                        split_item_id = len(item_old_ids) // 2
                        for id in item_old_ids[:split_item_id]:
                            fw2.write(str(user[user_old_id]) + "\t" + str(item[id]) + "\n")
                        for id in item_old_ids[split_item_id:]:
                            fw3.write(str(user[user_old_id]) + "\t" + str(item[id]) + "\n")
                        
                    else:
                        for id in total[user_old_id]:
                            fw.write(str(user[user_old_id])+"\t"+str(item[id])+"\n")


if __name__ == '__main__':
    random.seed(42)
    # cloth sport
    # cell electronic
    # game video
    # cd movie
    # music instrument
    source = "cd"
    target = "music"
    f1 = os.path.join(os.path.dirname(__file__), 'generated_data') + '/' + source + "_" + target + "/"
    f2 = os.path.join(os.path.dirname(__file__), 'generated_data') + '/' + target + "_" + source + "/"

    source_common_data = read_dataset(f1 + "common_new_reindex.txt") # same users data
    target_common_data = read_dataset(f2 + "common_new_reindex.txt")

    user_dict = {} # re-index
    source_item = {}
    target_item = {}
    if len(source_common_data) == len(target_common_data):
        user_dict = create_user_dict(user_dict, source_common_data)
    else:
        print("error!!!!!!")
        exit(0)

    source_total_data = read_dataset(f1 + "new_reindex.txt")
    target_total_data = read_dataset(f2 + "new_reindex.txt")

    source_user = create_user_dict(user_dict, source_total_data) # re-index
    target_user = create_user_dict(user_dict, target_total_data)

    source_item = create_item_dict(source_item, source_total_data)
    target_item = create_item_dict(target_item, target_total_data)

    print(len(source_user))
    print(len(target_user))
    print(len(source_item))
    print(len(target_item))

    generate_train_valid_test(f1, source_total_data, source_common_data, source_user, source_item, 0)
    generate_train_valid_test(f2, target_total_data, target_common_data, target_user, target_item, 1)

