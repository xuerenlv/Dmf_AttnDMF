# coding:utf8

import math
import numpy as np
import sys


# 读取源文件，返回：三元组（user,item,rating）
def read_data(file_name):
    ratings = []
    u_max_num = v_max_num = 0
    u_min_index = v_min_index = 0
    with open(file_name, 'r') as fr:
        for one_line in fr.readlines():
            if '100k' in file_name:
                one_line_list = one_line[:-1].split('\t')
                ratings.append((int(one_line_list[0]), int(one_line_list[1]), int(one_line_list[2]), int(one_line_list[3])))
                u_max_num, v_max_num = max(u_max_num, int(one_line_list[0])), max(v_max_num, int(one_line_list[1]))
                u_min_index, v_min_index = min(u_min_index, int(one_line_list[0])), min(v_min_index, int(one_line_list[1]))
            elif 'Amazon' in file_name:
                one_line_list = one_line[:-1].split(' ')
                ratings.append((int(one_line_list[0]), int(one_line_list[1]), float(one_line_list[2]), int(one_line_list[3])))
                u_max_num, v_max_num = max(u_max_num, int(one_line_list[0])), max(v_max_num, int(one_line_list[1]))
                u_min_index, v_min_index = min(u_min_index, int(one_line_list[0])), min(v_min_index, int(one_line_list[1]))
            else:
                one_line_list = one_line[:-1].split(':')
                ratings.append((int(one_line_list[0]), int(one_line_list[2]), int(one_line_list[4]), int(one_line_list[6])))
                u_max_num, v_max_num = max(u_max_num, int(one_line_list[0])), max(v_max_num, int(one_line_list[2]))
                u_min_index, v_min_index = min(u_min_index, int(one_line_list[0])), min(v_min_index, int(one_line_list[2]))
    if u_min_index == 0 and v_min_index == 0:
        return ratings, u_max_num + 1, v_max_num + 1
    elif u_min_index == 1 and v_min_index == 1:
        new_ratings = []
        for u, v, r, t in all_ratings:
            new_ratings.append((u-1, v-1, r, t))
        return ratings, u_max_num, v_max_num
    else:
        print "Dataset Wrong !"



def get_min_index_value(dd):
    re = []
    for k in dd:
        if dd[k] <= 0:
            re.append((k, dd[k]))
    return re

# 返回每一个用户交互过的 商品
def get_user_item_interaction_map(all_ratings):
    user_map_item = {}

    # user_count = {}
    # item_count = {}

    for u, v, r, t in all_ratings:
        if u not in user_map_item:
            user_map_item[u] = {}
        user_map_item[u][v] = t

        # user_count[u] = 1 + (1 if u not in user_count else 0)
        # item_count[v] = 1 + (1 if u not in item_count else 0)

    latest_item_interaction = {}
    for u in user_map_item:
        item = -1
        time = -1
        for v in user_map_item[u]:
            if time < user_map_item[u][v]:
                time = user_map_item[u][v]
                item = v
        latest_item_interaction[u] = item

    # 过滤掉每个用户的最后一次打分
    pruned_all_ratings = []
    for u, v, r, t in all_ratings:
        if v != latest_item_interaction[u]:
            pruned_all_ratings.append((u, v, r, t))
        # else:
        #     user_count[u] -= 1
        #     item_count[v] -= 1

    # print "min occur time:\t", get_min_index_value(user_count), "\tneg means only occur in last interaction"
    # print "min occur time:\t", get_min_index_value(item_count), "\tneg means only occur in last interaction"
    #
    # print "min occur time:\t", min(user_count.values()), "\tneg means only occur in last interaction"
    # print "min occur time:\t", min(item_count.values()), "\tneg means only occur in last interaction"

    return user_map_item, latest_item_interaction, pruned_all_ratings


if __name__ == '__main__':
    # ratings, u_max_num, v_max_num = read_data('../data_www/ml100k.txt')
    ratings, u_max_num, v_max_num = read_data('../data_www/Amazon_ratings_Movies_and_TV_pruned.txt')

    print u_max_num, v_max_num, len(ratings)

    # ratings, u_max_num, v_max_num = read_data('../data_www/ratings.dat')
    # user_map_item, latest_item_interaction, pruned_all_ratings = get_user_item_interaction_map(ratings)
    #
    # pruned_user_map_item = {}
    # pruned_item_map_user = {}
    # for u, v, r, t in pruned_all_ratings:
    #     if u not in pruned_user_map_item:
    #         pruned_user_map_item[u] = {}
    #     if v not in pruned_item_map_user:
    #         pruned_item_map_user[v] = {}
    #     pruned_user_map_item[u][v] = r
    #     pruned_item_map_user[v][u] = r
    #
    # for u in latest_item_interaction:
    #     if latest_item_interaction[u] in pruned_user_map_item[u]:
    #         print u, latest_item_interaction[u]
    #
    # print "user num: ", len(pruned_user_map_item), " item num:", len(pruned_item_map_user)

    pass
