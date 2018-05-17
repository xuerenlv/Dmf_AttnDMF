# coding:utf8
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from collections import OrderedDict
from models import inference_model
import cPickle as pickle
import math
import heapq
from processing_www_data import read_data, get_user_item_interaction_map

epochs = 100
first_layer_size = 300
# last_layer_size = 128


# *****************************************************************************************************
# dataset_name, file_name, batch_size, num_negs , learn_rate, show_peroid, last_layer_size \
#     = "Amusic", './data_www/Amazon_ratings_Digital_Music_pruned.txt', 32, 7, 1e-5, 2, 128
# graph_hyper_params = OrderedDict({'item_attn_attn_beta':1.0,
#                                   'user_attn_attn_beta':1.0,
#                                   'l2_reg_alpha':0.0,
#                                   'opt':'adam',
#                                   # 'opt':'adgrad',
#                                   'combine_norm_score_and_attn_type': 3, # 1:only use norm-score, 2:only use attn , 3:use both
#                                   'score_norm_type': 'exp', # 'exp' or 'sum_norm' or different number
#                                   })

dataset_name, file_name, batch_size, num_negs , learn_rate, show_peroid, last_layer_size \
    = "Amusic", './data_www/Amazon_ratings_Digital_Music_pruned.txt', 32, 4, 1e-5, 2, 128
graph_hyper_params = OrderedDict({'item_attn_attn_beta_score': 1.0,
                                  'user_attn_attn_beta_score': 1.0,
                                  'item_attn_attn_beta_attn': 1.0,
                                  'user_attn_attn_beta_attn': 1.0,
                                  'l2_reg_alpha': 0.0,
                                  'opt':'adam',
                                  # 'opt':'adgrad',
                                  'combine_norm_score_and_attn_type': 1, # 1:only use norm-score, 2:only use attn , 3:use both
                                  'score_norm_type': 'exp', # 'exp' or 'sum_norm' or different number
                                  'attn_norm_type': 'exp', # 'exp' or 'sum_norm' or different number
                                  })

# dataset_name, file_name, batch_size, num_negs, learn_rate, show_peroid , last_layer_size\
#     = "AmovieTv", './data_www/Amazon_ratings_Movies_and_TV_pruned.txt', 32, 7, 1e-5, 10, 128
# graph_hyper_params = OrderedDict({'item_attn_attn_beta_score': 1.0,
#                                   'user_attn_attn_beta_score': 1.0,
#                                   'item_attn_attn_beta_attn': 1.0,
#                                   'user_attn_attn_beta_attn': 1.0,
#                                   'l2_reg_alpha': 0.0,
#                                   'opt':'adam',
#                                   # 'opt':'adgrad',
#                                   'combine_norm_score_and_attn_type': 1, # 1:only use norm-score, 2:only use attn , 3:use both
#                                   'score_norm_type': 'exp', # 'exp' or 'sum_norm' or different number
#                                   'attn_norm_type': 'exp', # 'exp' or 'sum_norm' or different number
#                                   })

# dataset_name, file_name, batch_size, num_negs, learn_rate, show_peroid, last_layer_size\
#     = "ml100k", './data_www/ml100k.txt', 32, 7, 1e-4, 4, 128
# graph_hyper_params = OrderedDict({'item_attn_attn_beta_score': 1.0,
#                                   'user_attn_attn_beta_score': 1.0,
#                                   'item_attn_attn_beta_attn': 1.0,
#                                   'user_attn_attn_beta_attn': 1.0,
#                                   'l2_reg_alpha': 0.0,
#                                   'opt':'adam',
#                                   # 'opt':'adgrad',
#                                   'combine_norm_score_and_attn_type': 1, # 1:only use norm-score, 2:only use attn , 3:use both
#                                   'score_norm_type': 'exp', # 'exp' or 'sum_norm' or different number
#                                   'attn_norm_type': 'exp', # 'exp' or 'sum_norm' or different number
#                                   })

# dataset_name, file_name, batch_size, num_negs, learn_rate, show_peroid, last_layer_size\
#     = "ml1m", './data_www/ml1m_ratings.txt', 32, 4, 5e-6, 10, 128
# graph_hyper_params = OrderedDict({'item_attn_attn_beta_score': 1.0,
#                                   'user_attn_attn_beta_score': 1.0,
#                                   'item_attn_attn_beta_attn': 1.0,
#                                   'user_attn_attn_beta_attn': 1.0,
#                                   'l2_reg_alpha': 0.0,
#                                   'opt':'adam',
#                                   # 'opt':'adgrad',
#                                   'combine_norm_score_and_attn_type': 3, # 1:only use norm-score, 2:only use attn , 3:use both
#                                   'score_norm_type': 'exp', # 'exp' or 'sum_norm' or different number
#                                   'attn_norm_type': 'exp', # 'exp' or 'sum_norm' or different number
#                                   })
# *****************************************************************************************************
# model_name, learn_rate = 'DMF', 1e-4
model_name = 'DmfAttn'
neg_sample_size = batch_size * num_negs

ratings, u_max_num, v_max_num = read_data(file_name)
user_map_item, latest_item_interaction, pruned_all_ratings = get_user_item_interaction_map(ratings)


pruned_user_map_item = {}
pruned_item_map_user = {}
for u, v, r, t in pruned_all_ratings:
    if u not in pruned_user_map_item:
        pruned_user_map_item[u] = {}
    if v not in pruned_item_map_user:
        pruned_item_map_user[v] = {}
    pruned_user_map_item[u][v] = r
    pruned_item_map_user[v][u] = r

user_max_interact = max([len(pruned_user_map_item[u]) for u in pruned_user_map_item])
item_max_interact = max([len(pruned_item_map_user[v]) for v in pruned_item_map_user])
model_user_in = {}
model_item_in = {}
for u in pruned_user_map_item:
    u_minus = user_max_interact - len(pruned_user_map_item[u])
    model_user_in[u] = {"k":np.array(pruned_user_map_item[u].keys()+[0]*u_minus).astype('int32'),
                        "v":np.array([pruned_user_map_item[u].values()+[0.0]*u_minus]).astype('float32'),
                        "l":len(pruned_user_map_item[u])}
for v in pruned_item_map_user:
    v_minus = item_max_interact - len(pruned_item_map_user[v])
    model_item_in[v] = {"k":np.array(pruned_item_map_user[v].keys()+[0]*v_minus).astype('int32'),
                        "v":np.array([pruned_item_map_user[v].values()+[0.0]*v_minus]).astype('float32'),
                        "l":len(pruned_item_map_user[v])}

print model_name
print "2 layers"
print "batch       size: ", batch_size
print "neg sample  size: ", num_negs, neg_sample_size
print "learn       rate: ", learn_rate
print "file        name: ", file_name
print "first layer size: ", first_layer_size
print "last  layer size: ", last_layer_size
print "u               : ", u_max_num, item_max_interact
print "v               : ", v_max_num, user_max_interact
for key in graph_hyper_params:
    print key+":\t"+str(graph_hyper_params[key])


class MySampler():
    def __init__(self, all_ratings, u_max_num, v_max_num):
        self.sample_con = {}
        self.sample_con_size = 0

        self.all_ratings_map_u = {}
        for u, v, r, t in all_ratings:
            if u not in self.all_ratings_map_u:
                self.all_ratings_map_u[u] = {}
            self.all_ratings_map_u[u][v] = 1

        self.u_max_num = u_max_num
        self.v_max_num = v_max_num

    def smple_one(self):
        u_rand_num = int(np.random.rand() * self.u_max_num)
        v_rand_num = int(np.random.rand() * self.v_max_num)
        # if u_rand_num == 0:
        #     u_rand_num += 1
        # if v_rand_num == 0:
        #     v_rand_num += 1

        if u_rand_num in self.all_ratings_map_u and v_rand_num not in self.all_ratings_map_u[u_rand_num]:
            return u_rand_num, v_rand_num
        else:
            return self.smple_one()


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1.0
    return 0.0


def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0.0


time_now = str(datetime.now()).replace(' ','_')
checkpoint_dir = os.path.abspath("./checkpoints/"+model_name+"_"+dataset_name+"/"+time_now)
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# training
def train_matrix_factorization_With_Feed_Neural():
    top_k = 10
    best_hr = best_ndcg = 0.0
    my_sample = MySampler(pruned_all_ratings, u_max_num, v_max_num)

    user_id = tf.placeholder(tf.int32, [None, 1], name="user_id")
    u_index = tf.placeholder(tf.int32, [None, user_max_interact], name= "u_index")
    u_val = tf.placeholder(tf.float32, [None, 1, user_max_interact], name= "u_val")
    u_interact_length = tf.placeholder(tf.int32, [None, 1], name= "u_interact_length")

    item_id = tf.placeholder(tf.int32, [None, 1], name="item_id")
    v_index = tf.placeholder(tf.int32, [None, item_max_interact], name= "v_index")
    v_val = tf.placeholder(tf.float32, [None, 1, item_max_interact], name= "v_val")
    v_interact_length = tf.placeholder(tf.int32, [None, 1], name="u_interact_length")

    true_u_v = tf.placeholder(tf.float32, [None, 1], name= "true_u_v")

    global_step = tf.Variable(0, name="global_step", trainable=False)
    pred_val, model_loss, network_params = inference_model(model_name, user_id, u_index, u_val, u_interact_length, item_id, v_index, v_val, v_interact_length, v_max_num, u_max_num, first_layer_size, last_layer_size, user_max_interact, item_max_interact, true_u_v, graph_hyper_params)

    train_step = None
    if graph_hyper_params['opt'] == 'adam':
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(model_loss, global_step=global_step)
    elif graph_hyper_params['opt'] == 'adgrad':
        train_step = tf.train.AdagradOptimizer(learn_rate).minimize(model_loss, global_step=global_step)
    elif graph_hyper_params['opt'] == 'adadelta':
        train_step = tf.train.AdadeltaOptimizer(learn_rate).minimize(model_loss, global_step=global_step)
    else:
        print 'No optimizer !'

    batch_u_id = np.zeros((batch_size + neg_sample_size, 1)).astype('int32')
    batch_u_interact_length = np.zeros((batch_size + neg_sample_size, 1)).astype('int32')
    batch_u = np.zeros((batch_size + neg_sample_size, user_max_interact)).astype('int32'); tmp_u = np.array([0]*user_max_interact).astype('int32')
    batch_u_val = np.zeros((batch_size + neg_sample_size, 1, user_max_interact)).astype('float32'); tmp_u_val = np.array([[0.0]*user_max_interact]).astype('float32')

    batch_v_id = np.zeros((batch_size + neg_sample_size, 1)).astype('int32')
    batch_v_interact_length = np.zeros((batch_size + neg_sample_size, 1)).astype('int32')
    batch_v = np.zeros((batch_size + neg_sample_size, item_max_interact)).astype('int32'); tmp_v = np.array([0]*item_max_interact).astype('int32')
    batch_v_val = np.zeros((batch_size + neg_sample_size, 1, item_max_interact)).astype('float32'); tmp_v_val = np.array([[0.0]*item_max_interact]).astype('float32')
    batch_true_u_v = np.zeros((batch_size + neg_sample_size, 1)).astype('float32')

    batch_u_test_id = np.zeros((100, 1)).astype('int32')
    batch_u_test_interact_length = np.zeros((100, 1)).astype('int32')
    batch_u_test = np.zeros((100, user_max_interact)).astype('int32')
    batch_u_test_val = np.zeros((100, 1, user_max_interact)).astype('float32')

    batch_v_test_id = np.zeros((100, 1)).astype('int32')
    batch_v_test_interact_length = np.zeros((100, 1)).astype('int32')
    batch_v_test = np.zeros((100, item_max_interact)).astype('int32')
    batch_v_test_val = np.zeros((100, 1, item_max_interact)).astype('float32')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # sess = tf.Session(config=config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    for epoch in range(epochs):
        np.random.shuffle(pruned_all_ratings)
        one_epoch_loss = one_epoch_batchnum = 0.0
        for index in range(len(pruned_all_ratings) / batch_size):
            train_sample_index = 0
            for u_i, v_i, r_i, t_i in pruned_all_ratings[index * batch_size : (index + 1) * batch_size]:
                batch_u_id[train_sample_index], batch_v_id[train_sample_index]  = u_i, v_i
                batch_u_interact_length[train_sample_index][0] = model_user_in[u_i]["l"]
                batch_v_interact_length[train_sample_index][0] = model_item_in[v_i]["l"]

                batch_u[train_sample_index] = model_user_in[u_i]["k"]
                batch_u_val[train_sample_index][0] = model_user_in[u_i]["v"]

                batch_v[train_sample_index] = model_item_in[v_i]["k"]
                batch_v_val[train_sample_index][0] = model_item_in[v_i]["v"]
                batch_true_u_v[train_sample_index][0] = 1.0

                if model_user_in[u_i]["l"] > 1:
                    batch_u_interact_length[train_sample_index][0] -= 1
                    li = batch_u[train_sample_index].tolist()
                    ii_ind = li.index(v_i)
                    batch_u[train_sample_index][ii_ind] = li[batch_u_interact_length[train_sample_index][0]]
                    batch_u_val[train_sample_index][0][ii_ind] = batch_u_val[train_sample_index][0][batch_u_interact_length[train_sample_index][0]]
                    batch_u[train_sample_index][batch_u_interact_length[train_sample_index][0]] = 0
                    batch_u_val[train_sample_index][0][batch_u_interact_length[train_sample_index][0]] = 0.0

                if model_item_in[v_i]["l"] > 1:
                    batch_v_interact_length[train_sample_index][0] -= 1
                    li = batch_v[train_sample_index].tolist()
                    ii_ind = li.index(u_i)
                    batch_v[train_sample_index][ii_ind] = li[batch_v_interact_length[train_sample_index][0]]
                    batch_v_val[train_sample_index][0][ii_ind] = batch_v_val[train_sample_index][0][batch_v_interact_length[train_sample_index][0]]
                    batch_v[train_sample_index][batch_v_interact_length[train_sample_index][0]] = 0
                    batch_v_val[train_sample_index][0][batch_v_interact_length[train_sample_index][0]] = 0.0

                train_sample_index += 1

            for sam in range(neg_sample_size):
                u_i, v_i = my_sample.smple_one()
                batch_u_id[train_sample_index], batch_v_id[train_sample_index]  = u_i, v_i

                if u_i in model_user_in:
                    batch_u[train_sample_index] = model_user_in[u_i]["k"]
                    batch_u_val[train_sample_index][0] = model_user_in[u_i]["v"]
                    batch_u_interact_length[train_sample_index][0] = model_user_in[u_i]["l"]
                else:
                    batch_u[train_sample_index] = tmp_u
                    batch_u_val[train_sample_index][0] = tmp_u_val
                    batch_u_interact_length[train_sample_index][0] = 0

                if v_i in model_item_in:
                    batch_v[train_sample_index] = model_item_in[v_i]["k"]
                    batch_v_val[train_sample_index][0] = model_item_in[v_i]["v"]
                    batch_v_interact_length[train_sample_index][0] = model_item_in[v_i]["l"]
                else:
                    batch_v[train_sample_index] = tmp_v
                    batch_v_val[train_sample_index][0] = tmp_v_val
                    batch_v_interact_length[train_sample_index][0] =0
                batch_true_u_v[train_sample_index][0] = 0.0
                train_sample_index += 1

            feed_train = {user_id:batch_u_id, item_id:batch_v_id, u_index: batch_u, u_interact_length:batch_u_interact_length,
                          u_val:batch_u_val, v_index:batch_v, v_val:batch_v_val, v_interact_length:batch_v_interact_length,
                          true_u_v: batch_true_u_v}
            _, loss_val, pred_value = sess.run([train_step, model_loss, pred_val], feed_dict=feed_train)
            one_epoch_loss += loss_val
            one_epoch_batchnum += 1.0

            if index != 0 and index % ((len(pruned_all_ratings) / batch_size -1)/show_peroid) == 0:
                # print "epoch: ", epoch, " end"
                format_str = '%s epoch=%d in_epoch=%.2f avg_loss=%.4f'
                print (format_str % (datetime.now(), epoch, 1.0 * index/(len(pruned_all_ratings) / batch_size), one_epoch_loss / one_epoch_batchnum))
                one_epoch_loss = one_epoch_batchnum = 0.0

                # 计算 NDCG@10 与 HR@10
                # evaluate_1
                # evaluate_2
                test_hr_list, test_ndcg_list = [], []
                for u_i in latest_item_interaction:
                    v_latest = latest_item_interaction[u_i]

                    # print u_i, v_latest
                    v_random = [v_latest]
                    i = 1
                    while i < 100:
                        rand_num = int(np.random.rand() * (v_max_num - 1) + 1)
                        if rand_num not in user_map_item[u_i] and rand_num not in v_random and rand_num in pruned_item_map_user:
                            v_random.append(rand_num)
                            i += 1

                    for train_sample_index in range(100):
                        if u_i in model_user_in:
                            batch_u_test[train_sample_index] = model_user_in[u_i]["k"]
                            batch_u_test_val[train_sample_index][0] = model_user_in[u_i]["v"]
                            batch_u_test_interact_length[train_sample_index][0] = model_user_in[u_i]["l"]
                        else:
                            batch_u_test[train_sample_index] = tmp_u
                            batch_u_test_val[train_sample_index][0] = tmp_u_val
                            batch_u_test_interact_length[train_sample_index][0] = 0

                        v_i = v_random[train_sample_index]
                        if v_i in model_item_in:
                            batch_v_test[train_sample_index] = model_item_in[v_i]["k"]
                            batch_v_test_val[train_sample_index][0] = model_item_in[v_i]["v"]
                            batch_v_test_interact_length[train_sample_index][0] = model_item_in[v_i]["l"]
                        else:
                            batch_v_test[train_sample_index] = tmp_v
                            batch_v_test_val[train_sample_index][0] = tmp_v_val
                            batch_v_test_interact_length[train_sample_index][0] = 0

                        batch_u_test_id[train_sample_index], batch_v_test_id[train_sample_index] = u_i, v_i

                    feed_test = {user_id:batch_u_test_id, u_index:batch_u_test, u_val:batch_u_test_val, u_interact_length:batch_u_test_interact_length,
                                 item_id:batch_v_test_id, v_index:batch_v_test, v_val:batch_v_test_val, v_interact_length:batch_v_test_interact_length}
                    pred_value = sess.run([pred_val], feed_dict=feed_test)
                    pre_real_val = np.array(pred_value).reshape((-1))

                    items = v_random
                    gtItem = items[0]
                    # Get prediction scores
                    map_item_score = {}
                    for i in xrange(len(items)):
                        item = items[i]
                        map_item_score[item] = pre_real_val[i]

                    # Evaluate top rank list
                    # print map_item_score
                    ranklist = heapq.nlargest(top_k, map_item_score, key=map_item_score.get)
                    test_hr_list.append(getHitRatio(ranklist, gtItem))
                    test_ndcg_list.append(getNDCG(ranklist, gtItem))

                hr_val, ndcg_val = np.array(test_hr_list).mean(), np.array(test_ndcg_list).mean()
                if hr_val > best_hr or (hr_val == best_hr and ndcg_val > best_ndcg):
                    best_hr, best_ndcg = hr_val, ndcg_val
                    if epoch > 10: # 10轮之后再保存模型
                        current_step = tf.train.global_step(sess, global_step)
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("saved model to: %s" % path)

                print("result: hr=%.4f ndcg=%.4f best_hr=%.4f best_ndcg=%.4f" % (hr_val, ndcg_val, best_hr, best_ndcg))






if __name__ == "__main__":
    train_matrix_factorization_With_Feed_Neural()
