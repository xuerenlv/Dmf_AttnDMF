import tensorflow as tf


# base_model
def inference_model(model_name, user_id, u_index, u_val, u_interact_length, item_id, v_index, v_val, v_interact_length, v_max_num, u_max_num, first_layer_size, last_layer_size, user_max_interact, item_max_interact, true_u_v, graph_hyper_params):
    regularizer = tf.contrib.layers.l2_regularizer(graph_hyper_params['l2_reg_alpha'])
    if model_name == 'DMF':
        return inference_neural_DSSM_onehot(u_index, u_val, v_index, v_val, v_max_num, u_max_num, first_layer_size, last_layer_size, true_u_v, graph_hyper_params, regularizer)
    elif model_name == 'DmfAttn':
        return dmfAttn(user_id, u_index, u_val, u_interact_length, item_id, v_index, v_val, v_interact_length, v_max_num, u_max_num, first_layer_size, last_layer_size, user_max_interact, item_max_interact, true_u_v, graph_hyper_params, regularizer)
    else:
        print "NO MODEL !"


# dmf
def inference_neural_DSSM_onehot(u_index, u_val, v_index, v_val, v_max_num, u_max_num, first_layer_size, last_layer_size, true_u_v, graph_hyper_params, regularizer):
    batch_size = tf.shape(u_index)[0]
    with tf.variable_scope("OutAll"):
        u_w1 = tf.get_variable("u_w1", shape=(v_max_num, first_layer_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        u_w2 = tf.get_variable("u_w2", shape=(first_layer_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        u_b1 = tf.get_variable("u_b1", shape=[first_layer_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        u_b2 = tf.get_variable("u_b2", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        v_w1 = tf.get_variable("v_w1", shape=(u_max_num, first_layer_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        v_w2 = tf.get_variable("v_w2", shape=(first_layer_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        v_b1 = tf.get_variable("v_b1", shape=[first_layer_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        v_b2 = tf.get_variable("v_b2", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        net_u_1 = tf.nn.relu(tf.reshape(tf.matmul(u_val, tf.nn.embedding_lookup(u_w1, u_index)), [batch_size, first_layer_size]) + u_b1)
        net_u_2 = tf.matmul(net_u_1, u_w2) + u_b2

        net_v_1 = tf.nn.relu(tf.reshape(tf.matmul(v_val, tf.nn.embedding_lookup(v_w1, v_index)), [batch_size, first_layer_size]) + v_b1)
        net_v_2 = tf.matmul(net_v_1, v_w2) + v_b2

        fen_zhi = tf.reduce_sum(net_u_2 * net_v_2, 1, keep_dims=True)

        norm_u = tf.sqrt(tf.reduce_sum(tf.square(net_u_2), 1, keep_dims=True))
        norm_v = tf.sqrt(tf.reduce_sum(tf.square(net_v_2), 1, keep_dims=True))
        fen_mu = norm_u * norm_v

        pred_val = tf.nn.relu(fen_zhi / fen_mu)

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        gmf_loss = tf.reduce_mean(-true_u_v * tf.log(pred_val + 1e-10) - (1.0 - true_u_v) * tf.log(1.0 - pred_val + 1e-10))

        return pred_val, gmf_loss + regularization_loss, []


def gernal_softmax_with_length(attn_alpha, length, max_length, attn_beta, norm_type): # (batch_size, 1, max_length)
    if norm_type == 'exp':
        attn_alpha = tf.exp(attn_alpha)
    elif norm_type == 'sum_norm':
        return sum_normalize(attn_alpha, length, max_length)
    elif norm_type == 'sigmoid_norm':
        return sigmoid_att_weight(attn_alpha, length, max_length)
    else:
        attn_alpha = tf.pow(float(norm_type), attn_alpha)
    # elif norm_type == 'one_norm': # current, norm_type is a float number
    #     attn_alpha = tf.pow(float(1.0), attn_alpha)
    # else:
    #     print "no this norm_type:", norm_type
    attn_alpha *= tf.reshape(tf.cast(tf.sequence_mask(tf.reshape(length, [-1]), max_length), tf.float32), [-1, 1, max_length])
    _sum = tf.pow(tf.reduce_sum(attn_alpha, reduction_indices=2, keep_dims=True) + 1e-9, attn_beta)
    return attn_alpha / _sum


def sigmoid_att_weight(attn_alpha, length, max_length):
    attn_alpha *= tf.reshape(tf.cast(tf.sequence_mask(tf.reshape(length, [-1]), max_length), tf.float32), [-1, 1, max_length])
    return tf.nn.sigmoid(attn_alpha)

# xi = xi / sum(xi1, xi2 ,,,, xin) , because: all scores big than 0.0
def sum_normalize(attn_alpha, length, max_length):
    _sum = tf.reduce_sum(attn_alpha, reduction_indices=2, keep_dims=True) + 1e-9
    return attn_alpha/_sum


# ori_history_emb_list:(batch_size, max_length, hiddern_size)   current_emb:(batch, 1, hiddern_size)
# Attention default use exp-softmax
def sim_emb_2_emb_list(ori_history_emb_list, hiddern_size, current_emb, length, max_length, attn_beta, op_scope_name, graph_hyper_params, regularizer):
    with tf.variable_scope(op_scope_name):

        # inner-product attn
        h_2_1 = tf.get_variable("h_2_1", shape=(hiddern_size, 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        h_2_h = tf.get_variable("h_2_h", shape=(hiddern_size, hiddern_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        h_2_h_b = tf.get_variable("h_2_h_b", shape=(hiddern_size,), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        inner_product_res = tf.reshape(ori_history_emb_list * current_emb, [-1, hiddern_size])
        # print "attn func is tanh !"
        f_act = tf.nn.relu(tf.matmul(inner_product_res, h_2_h) + h_2_h_b) # now is tanh
        attn_alpha = tf.reshape(tf.matmul(f_act, h_2_1), [-1, 1, max_length])

        if 'attn_norm_type' in graph_hyper_params:
            norm = graph_hyper_params['attn_norm_type']
        else:
            norm = 'exp'

        return gernal_softmax_with_length(attn_alpha, length, max_length, attn_beta, norm_type=norm)


def dmfAttn(user_id, u_index, u_val, u_interact_length, item_id, v_index, v_val, v_interact_length, v_max_num, u_max_num, first_layer_size, last_layer_size, user_max_interact, item_max_interact, true_u_v, graph_hyper_params, regularizer):
    batch_size = tf.shape(u_index)[0]
    with tf.variable_scope("OutAll"):
        item_history_emb_w = tf.get_variable("item_history_item_emb_w", shape=(v_max_num, first_layer_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        current_item_emb_w = tf.get_variable("current_item_emb_w", shape=(v_max_num, first_layer_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        user_history_emb_w = tf.get_variable("user_history_item_emb_w", shape=(u_max_num, first_layer_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        current_user_emb_w = tf.get_variable("current_user_emb_w", shape=(u_max_num, first_layer_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        u_b1 = tf.get_variable("u_b1", shape=[first_layer_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        u_w2 = tf.get_variable("u_w2", shape=(first_layer_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        u_b2 = tf.get_variable("u_b2", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)


        v_b1 = tf.get_variable("v_b1", shape=[first_layer_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        v_w2 = tf.get_variable("v_w2", shape=(first_layer_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        v_b2 = tf.get_variable("v_b2", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)


        ori_user_history_emb_list = tf.nn.embedding_lookup(item_history_emb_w, u_index)
        current_item_emb = tf.nn.embedding_lookup(current_item_emb_w, item_id)
        u_val_normalize = gernal_softmax_with_length(u_val, u_interact_length, user_max_interact, graph_hyper_params['item_attn_attn_beta_score'], graph_hyper_params['score_norm_type'])
        item_attn_result = sim_emb_2_emb_list(ori_user_history_emb_list, first_layer_size, current_item_emb, u_interact_length, user_max_interact, graph_hyper_params['item_attn_attn_beta_attn'], "item_based_attn", graph_hyper_params, regularizer)


        ori_item_history_emb_list = tf.nn.embedding_lookup(user_history_emb_w, v_index)
        current_user_emb = tf.nn.embedding_lookup(current_user_emb_w, user_id)
        v_val_normalize = gernal_softmax_with_length(v_val, v_interact_length, item_max_interact, graph_hyper_params['user_attn_attn_beta_score'], graph_hyper_params['score_norm_type'])
        user_attn_result = sim_emb_2_emb_list(ori_item_history_emb_list, first_layer_size, current_user_emb, v_interact_length, item_max_interact, graph_hyper_params['user_attn_attn_beta_attn'], "user_based_attn", graph_hyper_params, regularizer)

        u_val_conbined = None
        v_val_conbined = None
        # only three different type
        if graph_hyper_params['combine_norm_score_and_attn_type'] == 0:
            u_val_conbined = u_val
            v_val_conbined = v_val
        elif graph_hyper_params['combine_norm_score_and_attn_type'] == 1:
            u_val_conbined = u_val_normalize
            v_val_conbined = v_val_normalize
        elif graph_hyper_params['combine_norm_score_and_attn_type'] == 2:
            u_val_conbined = item_attn_result
            v_val_conbined = user_attn_result
        elif graph_hyper_params['combine_norm_score_and_attn_type'] == 3:
            u_val_conbined = u_val_normalize * item_attn_result
            v_val_conbined = v_val_normalize * user_attn_result
        elif graph_hyper_params['combine_norm_score_and_attn_type'] == 4:
            u_val_conbined = u_val * item_attn_result
            v_val_conbined = v_val * user_attn_result
        elif graph_hyper_params['combine_norm_score_and_attn_type'] == 5:
            u_val_conbined = u_val + item_attn_result*5.0
            v_val_conbined = v_val + user_attn_result*5.0
        elif graph_hyper_params['combine_norm_score_and_attn_type'] == 6:
            u_val_conbined = u_val_normalize + item_attn_result
            v_val_conbined = v_val_normalize + user_attn_result

        elif graph_hyper_params['combine_norm_score_and_attn_type'] == 7:
            u_val_conbined = u_val * item_attn_result * 5.0
            v_val_conbined = v_val * user_attn_result * 5.0

        elif graph_hyper_params['combine_norm_score_and_attn_type'] == 8: ## sigmoid norm
            u_val_conbined = u_val * item_attn_result * 5.0
            v_val_conbined = v_val * user_attn_result * 5.0

        else:
            print "combine_norm_score_and_attn_type, wrong !"

        net_u_1 = tf.nn.relu(tf.reshape(tf.matmul(u_val_conbined, ori_user_history_emb_list), [batch_size, first_layer_size]) + u_b1)
        net_u_2 = tf.matmul(net_u_1, u_w2) + u_b2

        net_v_1 = tf.nn.relu(tf.reshape(tf.matmul(v_val_conbined, ori_item_history_emb_list), [batch_size, first_layer_size]) + v_b1)
        net_v_2 = tf.matmul(net_v_1, v_w2) + v_b2

        fen_zhi = tf.reduce_sum(net_u_2 * net_v_2, 1, keep_dims=True)

        norm_u = tf.sqrt(tf.reduce_sum(tf.square(net_u_2), 1, keep_dims=True))
        norm_v = tf.sqrt(tf.reduce_sum(tf.square(net_v_2), 1, keep_dims=True))
        fen_mu = norm_u * norm_v

        pred_val = tf.nn.relu(fen_zhi / fen_mu)

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        gmf_loss = tf.reduce_mean(-true_u_v * tf.log(pred_val + 1e-10) - (1.0 - true_u_v) * tf.log(1.0 - pred_val + 1e-10))

        return pred_val, gmf_loss + regularization_loss, []






















