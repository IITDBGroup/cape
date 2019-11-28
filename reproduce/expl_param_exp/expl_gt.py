#!/usr/bin/python
# -*- coding:utf-8 -*- 

import psycopg2
import sys, getopt
import pandas
import csv
from itertools import combinations
import statsmodels.formula.api as sm
from sklearn import preprocessing
from scipy.stats import chisquare,mode
from numpy import percentile,mean,arctanh
import math
import time
from heapq import *
import re
import itertools
import bisect
import sqlalchemy as sa
import datetime
import random
# import matplotlib.pyplot as plt

from similarity.category_similarity_matrix import *
from similarity.category_network_embedding import *
from similarity.category_similarity_naive import *
from explanation_model.explanation_model import *
from utils import *
import utils
from explain.pattern_retrieval import find_patterns_relevant, find_patterns_refinement, load_patterns
from explain.tuple_retrieval import get_tuples_by_F_V




# DEFAULT_QUERY_RESULT_TABLE = 'crime_subset'
# DEFAULT_PATTERN_TABLE = 'crime.crime_subset'
# DEFAULT_QUERY_RESULT_TABLE = 'crime_clean_100000_2'
# DEFAULT_PATTERN_TABLE = 'dev.crime_clean_100000'

# DEFAULT_QUERY_RESULT_TABLE = 'crime_partial'
# DEFAULT_PATTERN_TABLE = 'dev.crime_partial'
# DEFAULT_PATTERN_TABLE = 'crime_2017_2'

TEST_ID = '_7'
DEFAULT_QUESTION_PATH = './exp_parameter/user_question_expl_gt_7.txt'

DEFAULT_QUERY_RESULT_TABLE = 'synthetic.crime_exp' + TEST_ID
DEFAULT_PATTERN_TABLE = 'dev.crime_exp'+ TEST_ID

# DEFAULT_QUERY_RESULT_TABLE = 'crime_exp'
# DEFAULT_PATTERN_TABLE = 'dev.crime_exp'
# # DEFAULT_PATTERN_TABLE = 'crime_2017_2'
# DEFAULT_QUESTION_PATH = '../input/user_question_crime_partial_1.csv'
DEFAULT_LOCAL_SUPPORT = 5
DEFAULT_EPSILON = 0.1
EXAMPLE_NETWORK_EMBEDDING_PATH = '../input/NETWORK_EMBEDDING'
EXAMPLE_SIMILARITY_MATRIX_PATH = '../input/SIMILARITY_DEFINITION'
DEFAULT_AGGREGATE_COLUMN = '*'
TOP_K = 10


def predict(local_pattern, t):
    # print('In predict ', local_pattern)
    if local_pattern[4] == 'const':
        # predictY = float(local_pattern[-1][1:-1])
        # predictY = float(local_pattern[-2][1:-1].split(',')[0])
        predictY = float(local_pattern[-4][1:-1].split(',')[0])
    elif local_pattern[4] == 'linear':
        # print(local_pattern, t)
        v = get_V_value(local_pattern[2], t)
        # params = list(map(float, local_pattern[-1][1:-1].split(',')))
        # print(local_pattern[-1])
        # if isinstance(local_pattern[-1], str):
            # params_str = local_pattern[-1].split('\n')
        if isinstance(local_pattern[-3], str):
            params_str = local_pattern[-3].split('\n')
            # params = list(map(float, ))
            # print(params_str, v)
            params_dict = {}
            for i in range(0, len(params_str)-1):
                # print(params_str[i])
                # p_cate = re.compile(r'(.*)\[T\.\s+(.*)\]\s+(-?\d+\.\d+)')
                p_cate = re.compile(r'(.*)\[T\.\s*(.*)\]\s+(-?\d+\.\d+)')
                cate_res = p_cate.findall(params_str[i])
                if len(cate_res) != 0:
                    cate_res = cate_res[0]
                    v_attr = cate_res[0]
                    v_val = cate_res[1]
                    param = float(cate_res[2])
                    if v_attr not in params_dict:
                        params_dict[v_attr] = {}
                    params_dict[v_attr][v_val] = param
                else:
                    p_nume = re.compile(r'([^\s]+)\s+(-?\d+\.\d+)')
                    nume_res = p_nume.findall(params_str[i])
                    if len(nume_res) == 0:
                        continue
                    nume_res = nume_res[0]
                    v_attr = nume_res[0]
                    param = float(nume_res[1])
                    params_dict[v_attr] = param
        else:
            # params_dict = local_pattern[-1]
            params_dict = local_pattern[-3]

        predictY = params_dict['Intercept']
        # print(t, params_dict)
        for v_attr in t:
            v_key = '{}[T.{}]'.format(v_attr, t[v_attr])
            if v_key in params_dict:
                predictY += params_dict[v_key]
            else:
                if v_attr in params_dict:
                    predictY += params_dict[v_attr] * float(t[v_attr])
            
        # for v_attr, v_dict in params_dict.items():
        #     print(v_attr, v_dict, t)
        #     if v_attr == 'Intercept':
        #         predictY += v_dict
        #     else:
        #         if isinstance(v_dict, dict):
        #             v_key = t[v_attr].replace('\'', '').replace(' ', '')
        #             # print(v_attr, v_key)
        #             # print(v_dict.keys())
        #             if v_key in v_dict:
        #                 predictY += v_dict[v_key]
        #         else:
        #             print(v_attr)
        #             if v_attr in t:
        #                 predictY += v_dict * t[v_attr]

        # predictY = sum(map(lambda x: x[0]*x[1], zip(params[:-1], v))) + params[-1]

    return predictY

def tuple_distance(t1, t2, var_attr, cat_sim, num_dis_norm, agg_col):
    """Compute the similarity between two tuples t1 and t2 on their attributes var_attr

    Args:
        t1, t2: two tuples
        var_attr: variable attributes
        cat_sim: the similarity measure for categorical attributes
        num_dis_norm: normalization terms for numerical attributes
        agg_col: the column of aggregated value
    Returns:
        the Gower similarity between t1 and t2
    """
    dis = 0.0
    cnt = 0
    if var_attr is None:
        var_attr = t1.keys()
    max_dis = 0.0

    for v_col in var_attr:
        col = v_col.replace(' ', '')

        if col not in t1 and col not in t2:
            # if col == 'name':
            #     dis += 10000
            # else:
            #     dis += 100
            # cnt += 1
            continue
        if col not in t1 or col not in t2:
            # if col == 'name':
            #     dis += 10000
            # else:
            #     dis += 100
            # cnt += 1
            continue

        if col == 'name':
            if t1[col] != t2[col]:
                dis += 10000
            cnt += 1
            continue

        if col == 'venue' or col == 'pubkey':
            if t1[col] != t2[col]:
                dis += 0.25
            cnt += 1
            continue

        if cat_sim.is_categorical(col):

            t1_key = str(t1[col]).replace("'", '').replace(' ', '')
            t2_key = str(t2[col]).replace("'", '').replace(' ', '')
            s = 0
            if t1[col] == t2[col]:
                s = 1
            else:
                s = cat_sim.compute_similarity(col, t1_key, t2_key, agg_col)

            if s == 0:
                dis += 1
                max_dis = 1
            else:
                dis += (((1.0 / s)) * ((1.0 / s))) / 100
                # dis += (1-s) * (1-s)
                if math.sqrt((((1.0 / s)) * ((1.0 / s)) - 1) / 100) > max_dis:
                    max_dis = math.sqrt((((1.0 / s)) * ((1.0 / s)) - 1) / 100)

            cnt += 1
        else:
            if col != 'year' and (col not in num_dis_norm or num_dis_norm[col]['range'] is None):
                if t1[col] == t2[col]:
                    dis += 0
                else:
                    dis += 1
            else:
                if col != agg_col and col != 'index':
                    if isinstance(t1[col], datetime.date):
                        diff = datetime.datetime(t1[col].year, t1[col].month, t1[col].day) - datetime.datetime.strptime(
                            t2[col], "%Y-%m-%d")
                        temp = diff.days
                    else:
                        temp = abs(float(t1[col]) - float(t2[col]))

                    dis += 0.5 * math.pow(temp, temp+5)
                    if temp > max_dis:
                        max_dis = temp
                cnt += 1

    return math.pow(dis, 0.5)

def get_local_patterns(F, Fv, V, agg_col, model_type, t, conn, cur, pat_table_name, res_table_name, cat_sim, theta_lb=0.1):
    def validate_local_support(F_list, V_list, f_value, cur, table_name, cat_sim):
        def tuple_column_to_str_in_where_clause_2(col_value):
            # print(col_value, cat_sim.is_categorical(col_value[0]))
            if cat_sim.is_categorical(col_value[0]) or col_value == 'year':
                # return "like '%" + str(col_value[1]) + "%'"
                return "= '" + str(col_value[1]) + "'"
            else:
                if is_float(col_value[1]):
                    return '=' + str(col_value[1])
                else:
                    # return "like '%" + str(col_value[1]) + "%'"
                    return "= '" + str(col_value[1]) + "'"

        F = str(F_list).replace("\'", '')[1:-1]
        V = str(V_list).replace("\'", '')[1:-1]
        
        where_clause = ' AND '.join(
            list(map(lambda x: "{} {}".format(x[0], x[1]), 
                zip(local_pattern[0], 
                    map(tuple_column_to_str_in_where_clause_2, 
                        zip(F_list, f_value))))))

        tuples_query = '''SELECT {},COUNT(DISTINCT {}) FROM {} WHERE {} GROUP BY {};'''.format(
                F, V, table_name, where_clause, F
            )
        
    tF = get_F_value(F, t)
        
    if model_type is not None:
        mt_predicate = " AND model='{}'".format(model_type)
    else:
        mt_predicate = ''

    if Fv is not None:
        if DEFAULT_LOCAL_SUPPORT != 5:
            if not validate_local_support(F, V, Fv, cur, res_table_name, cat_sim):
                return []
        local_pattern_query = '''SELECT * FROM {} WHERE array_to_string(fixed, ', ')='{}' AND 
            REPLACE(array_to_string(fixed_value, ', '), '"', '') LIKE '%{}%' AND 
            array_to_string(variable, ', ') = '{}' AND
            agg='{}'{} AND theta > {};'''.format(
                pat_table_name + '_local', str(F).replace("\'", '').replace('[', '').replace(']', ''), 
                str(Fv).replace("\'", '').replace('[', '').replace(']', ''),
                str(V).replace("\'", '').replace('[', '').replace(']', ''), 
                agg_col, mt_predicate, str(theta_lb)
            )
        # print(406, Fv, local_pattern_query)
    else:
        if DEFAULT_LOCAL_SUPPORT != 5:
            if not validate_local_support(F, V, list(map(str, tF)), cur, res_table_name, cat_sim):
                return []
    #     local_pattern_query = '''SELECT * FROM {} WHERE array_to_string(fixed, ', ')='{}' AND 
    #         REPLACE(array_to_string(fixed_value, ', '), '"', '') = REPLACE('{}', '"', '') AND array_to_string(variable, ', ')='{}' AND
    #         agg='{}'{};'''.format(
    #             pat_table_name + '_local'+TEST_ID, str(F).replace("\'", '').replace('[', '').replace(']', ''), 
    #             str(tF).replace("\'", '"').replace('[', '').replace(']', ''),
    #             str(V).replace("\'", '').replace('[', '').replace(']', ''), 
    #             agg_col, mt_predicate
    #         )
        local_pattern_query = '''SELECT * FROM {} WHERE array_to_string(fixed, ', ')='{}' AND 
            REPLACE(array_to_string(fixed_value, ', '), '"', '') LIKE '%{}%' AND array_to_string(variable, ', ')='{}' AND
            agg='{}'{};'''.format(
                pat_table_name + '_local', str(F).replace("\'", '').replace('[', '').replace(']', ''), 
                '%'.join(list(map(str, tF))),
                str(V).replace("\'", '').replace('[', '').replace(']', ''), 
                agg_col, mt_predicate
            )
        # print(local_pattern_query)
        
    cur.execute(local_pattern_query)
    local_patterns = cur.fetchall()
        
    return local_patterns


def score_of_explanation(t1, t2, cat_sim, num_dis_norm, dir, denominator=1, lp1=None, lp2=None):
    if lp1 is None:
        return 1.0
    else:
        #print(lp1, lp2, t1, t2)
        agg_col = lp1[3]

        # t1fv = dict(zip(lp1[0] + lp1[2], map(lambda x:x, get_F_value(lp1[0] + lp1[2], t1))))
        # t2fv = dict(zip(lp2[0] + lp2[2], map(lambda x:x, get_F_value(lp2[0] + lp2[2], t2))))
        t1fv = dict()
        t2fv = dict()
        for a in lp2[0] + lp2[2]:
            t1fv[a] = t1[a]
            if a in t2:
                t2fv[a] = t2[a]
        # if lp1 == lp2:
        #     t_sim = tuple_similarity(t1fv, t2fv, lp1[2], cat_sim, num_dis_norm, agg_col)
        # else:
        #     t_sim = tuple_similarity(t1fv, t2fv, None, cat_sim, num_dis_norm, agg_col)
        # t_sim = tuple_similarity(t1fv, t2fv, None, cat_sim, num_dis_norm, agg_col)
        # print(745, t1, t1fv)
        # print(746, t2, t2fv)
        # print(lp1[0], lp1[2])
        # print(lp2[0], lp2[2])
        t_dis = tuple_distance(t1fv, t2fv, None, cat_sim, num_dis_norm, agg_col)
        cnt1 = 0
        cnt2 = 0
        for a1 in t1:
            if a1 != 'lambda' and a1 != agg_col:
                cnt1 += 1
        for a2 in t2:
            if a2 != 'lambda' and a2 != agg_col:
                cnt2 += 1
        
        diff = 0
        if len(t1.keys()) + 1 != len(t2.keys()):
            # diff = len(t2.keys()) - len(t1.keys()) - 1
            # if 'lambda' not in t2:
            #     diff += 1
            # w = 1
            # if 'name' in t2 and 'name' not in t1:
            #     w = 10000
            for col in t1:
                if col not in t2:
                    diff += 1
            for col in t2:
                if col not in t1:
                    diff += 1
        else:
            diff = 0
            for col in t1:
                if col != 'lambda' and col not in t2:
                    diff += 1
            for col in t2:
                if col != 'lambda' and col not in t1:
                    diff += 1
        w = 1
        t_dis = math.sqrt(t_dis * t_dis + w * diff * diff)
            # print(488, t1, t2)
        # print local_cons[i].var_attr, row
        t1v = dict(zip(lp1[2], map(lambda x:x, get_V_value(lp1[2], t1))))        
        predicted_agg1 = predict(lp1, t1v)
        t2v = dict(zip(lp2[2], map(lambda x:x, get_V_value(lp2[2], t2))))        
        predicted_agg2 = predict(lp2, t2v)
        # deviation - counterbalance_needed 
        deviation = float(t1[agg_col]) - predicted_agg1
        # counterbalance = float(t2[agg_col]) - predicted_agg2
        # deviation_normalized = (t1[agg_col] - predicted_agg1) / predicted_agg1
        # influence = -deviation / counterbalance
        #score = (deviation + counterbalance) * t_sim
        #score = deviation * (math.exp(t_sim) - 1)
        

        # if t_sim == 1:
        #     score = deviation * -dir
        # else:
        #     score = deviation * math.exp(t_sim) * -dir
        if t_dis == 0:
            score = deviation * -dir
        else:
            score = deviation / t_dis * -dir
        
        # score = deviation * t_sim * -dir * 1 / (1 + math.exp(-(influence - 0.5)))
        # score *= math.log(deviation_normalized + 3)
        # score /= counterbalance
        # print(414, t1fv, t2fv)
        # print(t_dis, deviation, score)
        # return score / float(denominator) * cnt1 / cnt2
        return [100 * score / float(denominator), t_dis, deviation, float(denominator), t_dis_raw]

# def compare_tuple(t1, t2):
#     flag1 = True
#     for a in t1:
#         # if (a != 'lambda' and a.find('_') == -1):
#         if a != 'lambda' and not a.startswith('count_') and not a.startswith('sum_'):
#             if a not in t2:
#                 flag1 = False
#             elif t1[a] != t2[a]:
#                 return 0
#     flag2 = True
#     for a in t2:
#         # if (a != 'lambda' and a.find('_') == -1):
#         if a != 'lambda' and not a.startswith('count_') and not a.startswith('sum_'):
#             if a not in t1:
#                 flag2 = False
#             elif t1[a] != t2[a]:
#                 return 0

#     if flag1 and flag2:
#         return -1
#     elif flag1:
#         return -1
#     else:
#         return 0

def compare_tuple(t1, t2):
    flag1 = True
    # if 'year' in t1 and t1['year'] == '2014' and 'primary_type' in t1 and t1['primary_type'] == 'CRIMINAL DAMAGE':
    #     print(788, t1, t2)
    for a in t1:
        # if (a != 'lambda' and a.find('_') == -1):
        if (a != 'lambda' and a != 'count'):
            if a not in t2:
                flag1 = False
            elif t1[a] != t2[a]:
                return 0
    flag2 = True
    for a in t2:
        # if (a != 'lambda' and a.find('_') == -1):
        if (a != 'lambda' and a != 'count'):
            if a not in t1:
                flag2 = False
            elif t1[a] != t2[a]:
                return 0
    # print(t1, t2, flag1, flag2)
    if flag1 and flag2:
        return -1
    elif flag1:
        return -1
    else:
        return 0

def DrillDown(global_patterns_dict, local_pattern, F_set, U_set, V_set, t_prime_coarser, t_coarser, t_prime, target_tuple,
        conn, cur, pat_table_name, res_table_name, cat_sim, num_dis_norm, 
        epsilon, dir, query_result, theta_lb=0.1):
    reslist = []
    F_prime_set = F_set.union(U_set)
    agg_col = local_pattern[3]
    # gp2_list = find_global_patterns_exact_match(global_patterns_dict, F_prime_set, V_set, local_pattern[3], local_pattern[4])
    # gp2_list = find_patterns_refinement(global_patterns_dict, F_prime_set, V_set, local_pattern[3], local_pattern[4])
    gp2_list = find_patterns_refinement(global_patterns_dict, F_set, V_set, local_pattern[3], local_pattern[4])
    # print(792, gp2_list)
    if len(gp2_list) == 0:
        return []
    for gp2 in gp2_list:

    
        # lp2_list = get_local_patterns(gp2[0], None, gp2[1], gp2[2], None, t_prime, conn, cur, pat_table_name, res_table_name)
        lp2_list = get_local_patterns(gp2[0], None, gp2[1], gp2[2], gp2[3], t_prime, conn, cur, pat_table_name, res_table_name, cat_sim, theta_lb)
        
        if len(lp2_list) == 0:
            continue
        lp2 = lp2_list[0]
        
        # if 'community_area' not in gp2[0] or len(gp2[0]) != 2:
        #     continue
        # print(lp2[0])
        # print(800, gp2, lp2_list)
        f_value = get_F_value(local_pattern[0], t_prime)
        tuples_same_F, agg_range, tuples_same_F_dict = get_tuples_by_F_V(local_pattern, lp2, f_value, 
            # [get_V_value(local_pattern[2], t_prime), [[-3, 3]]],
            None,
            conn, cur, res_table_name, cat_sim)
        lp3_list = get_local_patterns(lp2[0], f_value, lp2[2], lp2[3], lp2[4], t_prime, conn, cur, pat_table_name, res_table_name, cat_sim, theta_lb)
        # tuples_same_F, agg_range = get_tuples_by_F(local_pattern, lp2, f_value, 
        #     conn, cur, res_table_name, cat_sim)
        # tuples_same_F, agg_range, tuples_same_F_dict = get_tuples_by_F_V(local_pattern, lp2, f_value, None,
        #     conn, cur, res_table_name, cat_sim)
        # print(725, local_pattern[0], local_pattern[1], local_pattern[2], lp2[0], lp2[1], lp2[2])
        # print(725, len(tuples_same_F), len(lp3_list))
        # print(tuples_same_F_dict.keys())
        for lp3 in lp3_list:
            f_key = str(lp3[1]).replace('\'', '')[1:-1]
            # print(836, f_key)
            f_key = f_key.replace('.0', '')
            if f_key in tuples_same_F_dict:
                for idx, row in enumerate(tuples_same_F_dict[f_key]):
                    # print(821, row)
                    for idx2, row2 in enumerate(t_coarser):
                        if get_V_value(local_pattern[2], row2) == get_V_value(local_pattern[2], row):
                            s = score_of_explanation(row, target_tuple, cat_sim, num_dis_norm, dir, float(row2[agg_col]), lp3, lp2)
                            break
                
                    # e.g. U = {Venue}, u = {ICDE}, do not need to check whether {Author, Venue} {Year} holds on (JH, ICDE)
                    # expected values are replaced with the average across year for all (JH, ICDE, year) tuples
                    #s = score_of_explanation(row, t_prime, cat_sim)
                    cmp_res = compare_tuple(row, target_tuple)
                    if cmp_res == 0: # row is not subset of target_tuple, target_tuple is not subset of row
                        # print(551, row, target_tuple, s)
                        # reslist.append([s[0], s[1:], dict(row), local_pattern, lp3, 1])  
                        reslist.append(
                            Explanation(1, s[0], s[1], s[2], s[3], dir, dict(row), TOP_K, local_pattern, lp3))

                    # else:
                    #     reslist.append([s[0], s[1:], dict(row), local_pattern, lp3, 1])  
            # for f_key in tuples_same_F_dict:
            # # commented finer pattern holds only
            # if gp2 does not hold on (t[F], u):
            #     continue
            # e.g.{Author, venue} {Year} holds on (JH, ICDE)
            # lp3 = (F', (t[F], u), V, agg, a, m3)
            # s = score_of_explanation((t[F],u,t[V]), t, lp3, lp2)
            # f_prime_value = f_key.split(', ')
            # v_prime_value = get_V_value(lp2[0], row)
            # if v_prime_value[0]
            # f_prime_value = get_F_value(lp2[0], row)
            # lp3 = get_local_patterns(lp2[0], f_prime_value, lp2[2], lp2[3], lp2[4], row, conn, cur, pat_table_name, res_table_name)
            # lp3 = get_local_patterns(lp2[0], f_prime_value, lp2[2], lp2[3], lp2[4], tuples_same_F_dict[f_key][0], conn, cur, pat_table_name, res_table_name)

            # print(791, f_key, f_prime_value, len(lp3))
            # if len(lp3) == 0:
            #     continue
            # for row in tuples_same_F_dict[f_key]:
            #     s = score_of_explanation(row, target_tuple, cat_sim, num_dis_norm, dir, float(t_coarser[agg_col]), lp3[0], lp2)
            
            #     # e.g. U = {Venue}, u = {ICDE}, do not need to check whether {Author, Venue} {Year} holds on (JH, ICDE)
            #     # expected values are replaced with the average across year for all (JH, ICDE, year) tuples
            #     #s = score_of_explanation(row, t_prime, cat_sim)
            #     if not equal_tuple(row, target_tuple):
            #         # print(551, row, target_tuple, s)
            #         reslist.append([s[0], s[1:], dict(row), local_pattern, lp3[0], 1])

    
    return reslist

def find_explanation_regression_based(user_question_list, global_patterns, global_patterns_dict, 
    cat_sim, num_dis_norm, epsilon, agg_col, conn, cur, pat_table_name, res_table_name,
    lam, the, gs, local_patterns_count):

    """Find explanations for user questions

    Args:
        data: data['df'] is the data frame storing Q(R)
            data['le'] is the label encoder, data['ohe'] is the one-hot encoder
        user_question_list: list of user questions (t, dir), all questions have the same Q(R)
        cons: list of fixed attributes and variable attributes of global constraints
        cat_sim: the similarity measure for categorical attributes
        num_dis_norm: normalization terms for numerical attributes
        cons_epsilon: threshold for local regression constraints
        agg_col: the column of aggregated value
        regression_package: which package is used to compute regression 
    Returns:
        the top-k list of explanations for each user question
    """
    answer = [[] for i in range(len(user_question_list))]
    local_pattern_loading_time = 0
    question_validating_time = 0
    score_computing_time = 0
    score_computing_time_list = []
    result_merging_time = 0
    local_patterns_list = []
    # print(492, global_patterns)

    for j, uq in enumerate(user_question_list):
        dir = uq['dir']
        topK_heap = TopkHeap(TOP_K)
        marked = {}

        t = uq['target_tuple']
        # print(505, t)
        uq['global_patterns'] = find_patterns_relevant(
                global_patterns_dict, uq['target_tuple'], conn, cur, res_table_name, cat_sim)
        score_computing_time_start = time.time()
        start = time.time()
        # local_patterns = get_local_patterns(
        #     # list(map(lambda x:global_patterns[x], uq['global_patterns'])), 
        #     uq['global_patterns'],
        #     t, conn, cur, pat_table_name, res_table_name
        # )
        local_patterns = []
        # print(uq['global_patterns'])
        # print(709, uq['global_patterns'])
        # print(710, uq['local_patterns'])
        # for l_pat in uq['local_patterns']:
        #     if len(l_pat) > 1:
        #         local_patterns.append(l_pat)

        
        end = time.time()
        local_pattern_loading_time += end - start

        candidate_list = [[] for i in range(len(uq['global_patterns']))]
        top_k_lists = [[] for i in range(len(uq['global_patterns']))]
        validate_res_list = []
        # local_patterns_list.append(local_patterns)
        local_patterns = []
        
        psi = []
        
        utils.VISITED_DICT = dict()
        score_computing_time_cur_uq = 0
        score_computing_start = time.time()
        explanation_type = 0
        for i in range(0, len(uq['global_patterns'])):
            top_k_lists[i] = [4, uq['global_patterns'][i], t, []]
            local_patterns.append(None)
            F_key = str(sorted(uq['global_patterns'][i][0]))
            V_key = str(sorted(uq['global_patterns'][i][1]))
            # print(980, F_key, V_key)
            pat_key = F_key + '|,|' + V_key + '|,|' + uq['global_patterns'][i][2] + '|,|' + uq['global_patterns'][i][3]
            # if F_key.find('community_area') == -1:
            #     continue
            # print(955, uq['global_patterns'][i])
            if pat_key in utils.VISITED_DICT:
                continue
            utils.VISITED_DICT[pat_key] = True

            local_pattern_query_count = '''SELECT count(*) FROM {} 
                    WHERE array_to_string(fixed, ', ')='{}' AND 
                    array_to_string(variable, ', ')='{}' AND 
                    agg='{}' AND model='{}' AND theta>{};
            '''.format(
                pat_table_name + '_local',
                str(uq['global_patterns'][i][0]).replace("\'", '').replace('[', '').replace(']', ''),
                str(uq['global_patterns'][i][1]).replace("\'", '').replace('[', '').replace(']', ''), 
                uq['global_patterns'][i][2], uq['global_patterns'][i][3],
                the
            )
            cur.execute(local_pattern_query_count)
            res_count = cur.fetchall()
            print(1070, res_count[0][0])
            if res_count[0][0] < lam * local_patterns_count[i] or res_count[0][0] < gs:
                continue
            
            
            tF = get_F_value(uq['global_patterns'][i][0], t)
            local_pattern_query_fixed = '''SELECT * FROM {} 
                    WHERE array_to_string(fixed, ', ')='{}' AND 
                    array_to_string(fixed_value, ', ')='{}' AND
                    array_to_string(variable, ', ')='{}' AND 
                    agg='{}' AND model='{}' AND theta>{}
                ORDER BY theta DESC;
            '''.format(
                pat_table_name + '_local', 
                str(uq['global_patterns'][i][0]).replace("\'", '').replace('[', '').replace(']', ''),
                str(tF)[1:-1].replace("\'", ''),
                str(uq['global_patterns'][i][1]).replace("\'", '').replace('[', '').replace(']', ''), 
                uq['global_patterns'][i][2], uq['global_patterns'][i][3],
                the
            )
            cur.execute(local_pattern_query_fixed)
            res_fixed = cur.fetchall()
            print(1087, res_fixed)
            if len(res_fixed) == 0:
                continue
            local_patterns[i] = res_fixed[0]
            # if len(local_patterns[i][2]) > 1 or local_patterns[i][2][0] != 'year':
            #     continue
            T_set = set(t.keys()).difference(set(['lambda', uq['global_patterns'][i][2]]))
            psi.append(0)
            agg_col = local_patterns[i][3]
            start = time.time()
            # print('PAT', i, local_patterns[i])
            # t_coarser_list, agg_range, t_coarser_dict = get_tuples_by_F_V(local_patterns[i], local_patterns[i], 
            #     get_F_value(local_patterns[i][0], t), [get_V_value(local_patterns[i][2], t), [[0, 0]]],
            #     conn, cur, res_table_name, cat_sim)
            t_t_list, agg_range, t_t_dict = get_tuples_by_F_V(local_patterns[i], local_patterns[i], 
                get_F_value(local_patterns[i][0], t), 
                #[get_V_value(local_patterns[i][2], t), [[-3, 3]]],
                None,
                conn, cur, res_table_name, cat_sim)
            # print(t_t_list)
            dist_lb = 1e10
            dev_ub = 0
            for t_t in t_t_list:
                if compare_tuple(t_t, t) == 0:
                    s = score_of_explanation(t_t, t, cat_sim, num_dis_norm, dir, t_t[agg_col], local_patterns[i], local_patterns[i])
                    expl_temp = Explanation(0, s[0], s[1], s[2], s[3], uq['dir'],
                                                   # list(map(lambda y: y[1], sorted(t_t.items(), key=lambda x: x[0]))),
                                                   dict(t_t),
                                                   TOP_K, local_patterns[i], None)

                    expl_temp_str = expl_temp.ordered_tuple_string()
                    if expl_temp_str not in marked:
                        marked[expl_temp_str] = True
                        topK_heap.Push(expl_temp)
                        top_k_lists[i][-1].append(expl_temp)
                        # print(t_t, t, compare_tuple(t_t, t))
                    if s[-1] < dist_lb:
                        dist_lb = s[-1]
                        # use raw distance (without penalty on missing attributes) as the lower bound
                    if abs(s[2]) > dev_ub:
                        dev_ub = abs(s[2])

            
            end = time.time()
            question_validating_time += end - start

            
            F_set = set(local_patterns[i][0])
            V_set = set(local_patterns[i][2])
            
            # if 'venue' in local_patterns[i][0] and 'name' not in local_patterns[i][0]:
            #     continue
            # if 'pubkey' in local_patterns[i][0] and 'name' not in local_patterns[i][0]:
            #     continue
            # F union V \subsetneq G
            
            start = time.time()
            
            
            # print(t_coarser)
            t_coarser_copy = list(t_t_list)
            # t_coarser_copy[agg_col] = 1
            # if len(local_patterns[i][0]) + len(local_patterns[i][2]) < len(t.keys()) -2:
                # if 'pubkey' in local_patterns[i][0] and 'name' not in local_patterns[i][0]:
                #     continue
            # if 'venue' in local_patterns[i][0] and 'name' not in local_patterns[i][0]:
            #     continue

            # dev_ub_list = get_dev_upper_bound(local_patterns[i])
            # dev_ub = agg_range[1] - agg_range[0]
            norm_lb = min(list(map(lambda x: x[agg_col], t_coarser_copy)))
            # cur_top_k_res = topK_heap.TopK()
            # k_score = min(map(lambda x:x[0], cur_top_k_res))
            k_score = topK_heap.MinValue()
            # k_score = topK_heap.MaxValue()
            print(993, k_score, 100 * float(dev_ub) / (dist_lb * float(norm_lb)))
            if topK_heap.HeapSize() == TOP_K and 100 * float(dev_ub) / (dist_lb * float(norm_lb)) <= k_score:
                print(998, dev_ub, dist_lb, norm_lb, local_patterns[i][0], local_patterns[i][1])
                continue
            top_k_lists[i][-1] += DrillDown(global_patterns_dict, local_patterns[i], 
                F_set, T_set.difference(F_set.union(V_set)), V_set, t_coarser_copy, t_coarser_copy, t, t,
                conn, cur, pat_table_name, res_table_name, cat_sim, num_dis_norm, epsilon,
                dir, uq['query_result'], the)

            for tk in top_k_lists[i][-1]:
                # if str(tk.tuple_value) not in marked:
                #    marked[str(tk.tuple_value)] = True
                tk_str = tk.ordered_tuple_string()
                if tk_str not in marked:
                    marked[tk_str] = True
                    topK_heap.Push(tk)

            # for tk in top_k_lists[i][-1]:
            #     mark_key = str(list(map(lambda y: y[1], sorted(tk[2].items(), key=lambda x: x[0]))))
            #     if mark_key not in marked:
            #         marked[mark_key] = True
            #         topK_heap.Push([tk[0], tk[1], 
            #             # list(map(lambda y: str(y[1]), sorted(tk[2].items(), key=lambda x: x[0]))), 
            #             tk[2].items(),
            #             tk[3], tk[4], tk[5]])
            

            end = time.time()
            score_computing_time_cur_uq += end - start
            score_computing_time += end - start
            
                
        score_computing_time_end = time.time()    
        # uses heapq to manipulate merge of explanations from multiple constraints
        start = time.time()
        # complementary_explanation_list = []
        # heapify(complementary_explanation_list)
        # refute_explanation_list = []
        # for i in range(len(local_patterns)):
        #     if len(top_k_lists[i]) > 3 and len(top_k_lists[i][3]) > 0:
        #         for idx, tk in enumerate(top_k_lists[i][3]):
        #             top_k_lists[i][3][idx][0] = -tk[0]
        #             top_k_lists[i][3][idx][2] = list(map(lambda y: y[1], sorted(tk[2].items(), key=lambda x: x[0])))
        #         print(top_k_lists[i])
        #         heapify(top_k_lists[i][-1])
        #         # print(top_k_lists[i])
        #         poped_tuple = list(heappop(top_k_lists[i][-1]))
        #         poped_tuple.append(i)
        #         heappush(complementary_explanation_list, poped_tuple)
        #     else:
        #         refute_explanation_list.append(top_k_lists[i])
        answer[j] = [{} for i in range(TOP_K)]
        # score_computing_time_list.append([t, score_computing_time_cur_uq])
        score_computing_time_list.append([t, score_computing_time_end - score_computing_time_start])
        # cnt = 0
        # while cnt < TOP_K and len(complementary_explanation_list) > 0:
        #     poped_tuple = heappop(complementary_explanation_list)
        #     if str(poped_tuple[2]) not in marked:
        #         marked[str(poped_tuple[2])] = True
        #         answer[j][cnt] = (poped_tuple[0], poped_tuple[1], poped_tuple[2], poped_tuple[3], poped_tuple[4], poped_tuple[5])
        #         cnt += 1
        #     if len(top_k_lists[poped_tuple[-1]][3]) == 0:
        #         for i in range(len(local_patterns)):
        #             if len(top_k_lists[i]) > 3 and len(top_k_lists[i][-1]) > 0:
        #                 poped_tuple2 = list(heappop(top_k_lists[i][-1]))
        #                 poped_tuple2.append(i)
        #                 heappush(complementary_explanation_list, poped_tuple2)
                        
        #         heapify(complementary_explanation_list)
        #     else:
        #         # for tk in top_k_lists[poped_tuple[3]][3]:
        #         #     print(tk)

        #         poped_tuple2 = list(heappop(top_k_lists[poped_tuple[-1]][-1]))
        #         poped_tuple2.append(poped_tuple[-1])
        #         heappush(complementary_explanation_list, poped_tuple2)
        answer[j] = topK_heap.TopK()
        end = time.time()
        result_merging_time += end - start
        # print(749, answer[j])
        # print(topK_heap)

        # for i in range(len(local_patterns)):
        #     # print("TOP K: ", top_k_lists[i])
        #     if len(top_k_lists[i]) > 3:
        #         print(top_k_lists[i])
        #         answer[j].append(sorted(top_k_lists[i][-1], key=lambda x: x[1], reverse=True)[0:TOP_K])
        #     else:
        #         answer[j].append(top_k_lists[i])
        #     # print(len(answer[j][-1]))



    print('Local pattern loading time: ' + str(local_pattern_loading_time) + 'seconds')
    print('Question validating time: ' + str(question_validating_time) + 'seconds')
    print('Score computing time: ' + str(score_computing_time) + 'seconds')
    print('Result merging time: ' + str(result_merging_time) + 'seconds')
    return answer, local_patterns_list, score_computing_time_list



def find_user_question(cur):
    # find_query = '''
    # SELECT A.primary_type,A.district,A.year,A.cnt_new,A.cnt_new-B.cnt_old FROM
    # (SELECT primary_type,district,year,count(*) as cnt_new FROM synthetic.crime_exp_{} GROUP BY primary_type,district,year) A,
    # (SELECT primary_type,district,year,count(*) as cnt_old FROM crime_exp GROUP BY primary_type,district,year) B
    # WHERE A.primary_type=B.primary_type AND A.district = B.district AND A.year=B.year AND A.cnt_new <> B.cnt_old;
    # '''
    find_query = '''
    SELECT A.primary_type,A.beat,A.year,A.cnt_new,A.cnt_new-B.cnt_old FROM
    (SELECT primary_type,beat,year,count(*) as cnt_new FROM synthetic.crime_exp_{} GROUP BY primary_type,beat,year) A,
    (SELECT primary_type,beat,year,count(*) as cnt_old FROM crime_exp GROUP BY primary_type,beat,year) B
    WHERE A.primary_type=B.primary_type AND A.beat = B.beat AND A.year=B.year AND A.cnt_new <> B.cnt_old;
    '''
    uq_list = []
    if TEST_ID == '_7':
        cur.execute(find_query.format('7'))
        res = cur.fetchall()
        for row in res:
            uq_list.append({'primary_type':row[0],'beat':row[1],
                # 'year':row[2],'count_star':row[3],'direction':'high', 'lambda':0.0})
                'year':row[2],'count':row[3],'direction':'high', 'lambda':0.0})
            uq_list.append({'primary_type':row[0],'beat':row[1],
                # 'year':row[2],'count_star':row[3],'direction':'low', 'lambda':0.0})
                'year':row[2],'count':row[3],'direction':'low', 'lambda':0.0})
    if TEST_ID == '_9':
        cur.execute(find_query.format('9'))
        res = cur.fetchall()
        for row in res:
            uq_list.append({'primary_type':row[0],'beat':row[1],
                # 'year':row[2],'count_star':row[3],'direction':'high', 'lambda':0.0})
                'year':row[2],'count':row[3],'direction':'high', 'lambda':0.0})
            uq_list.append({'primary_type':row[0],'beat':row[1],
                # 'year':row[2],'count_star':row[3],'direction':'low', 'lambda':0.0})
                'year':row[2],'count':row[3],'direction':'low', 'lambda':0.0})
    # find_query = '''
    # SELECT A.primary_type,A.description,A.community_area,A.year,A.cnt_new,A.cnt_new-B.cnt_old FROM
    # (SELECT primary_type,description,community_area,year,count(*) as cnt_new FROM synthetic.crime_exp_{} 
    # GROUP BY primary_type,description,community_area,year) A,
    # (SELECT primary_type,description,community_area,year,count(*) as cnt_old FROM crime_exp 
    # GROUP BY primary_type,description,community_area,year) B
    # WHERE A.primary_type=B.primary_type AND A.description = B.description AND 
    # A.community_area = B.community_area AND A.year=B.year AND A.cnt_new <> B.cnt_new;
    # '''
    find_query = '''
    SELECT A.primary_type,A.community_area,A.year,A.cnt_new,A.cnt_new-B.cnt_old FROM
    (SELECT primary_type,community_area,year,count(*) as cnt_new FROM synthetic.crime_exp_{} 
    GROUP BY primary_type,community_area,year) A,
    (SELECT primary_type,community_area,year,count(*) as cnt_old FROM crime_exp 
    GROUP BY primary_type,community_area,year) B
    WHERE A.primary_type=B.primary_type AND A.community_area = B.community_area AND A.year=B.year AND A.cnt_new <> B.cnt_old;
    '''
    if 'TEST_ID' == '_10':
        cur.execute(find_query.format('10'))
        res = cur.fetchall()
        for row in res:
            uq_list.append({'primary_type':row[0],'community_area':row[1],
                'year':row[2],'count_star':row[3],'direction':'high', 'lambda':0.0})
            uq_list.append({'primary_type':row[0],'community_area':row[1],
                'year':row[2],'count_star':row[3],'direction':'low', 'lambda':0.0})

    return uq_list
    # return random.sample(uq_list, 10)

def load_user_question(global_patterns, global_patterns_dict, uq_path=DEFAULT_QUESTION_PATH, schema=None, cur=None, pattern_table='', query_result_table=''):
    '''
        load user questions
    '''
    print(global_patterns_dict)
    uq = []
    uq_cnt = 0

    # uqs = [1,9,25,33,41,49,65,73,89,97,105,113,121,137,145,153,161,169,177,193]
    uqs = [1,9,25,81,89,97,121,137,153,93]
    with open(uq_path, 'rt') as uqfile:
        reader = csv.DictReader(uqfile, quotechar='\'')
        headers = reader.fieldnames
        #temp_data = csv.reader(uqfile, delimiter=',', quotechar='\'')
        #for row in temp_data:
        for row in reader:
            row_data = {}
            raw_row_data = {}
            agg_col = None
            uq_cnt += 1
            #  0, 1, 2, 3, 4,  5, 6,  7, 8, 9, 10, 11, 12, 13,14, 15, 16,17,18, 19,20,21,22, 23,24
            # [9, 9, 3, 10, 8, 10, 4, 8, 9, 3, 8,  10, 9, 10, 10, 8,  1, 8,  9, 9, 10, 5, 7, 0, 10]
            # if (uq_cnt % 8 != 1) or uq_cnt > 199:
            #     continue
            if uq_cnt not in uqs:
                continue
            # for k, v in enumerate(headers):
            for v in row:
                # print(k, v)
                if schema is None or v not in schema:
                    if v != 'direction':
                        # if is_float(row[v]):
                        #     row_data[v] = float(row[v])
                        # elif is_integer(row[v]):
                        #     row_data[v] = float(long(row[v]))
                        # else:
                        row_data[v] = row[v]
                    if v not in schema and v != 'lambda' and v != 'direction':
                        agg_col = v
                else:
                    if row[v] != '*':
                        if v.startswith('count_') or v.startswith('sum_'):
                            agg_col = v 
                        if schema[v] == 'integer':
                            row_data[v] = int(row[v])
                            raw_row_data[v] = int(row[v])
                        elif schema[v].startswith('double') or schema[v].startswith('float'):
                            row_data[v] = float(row[v])
                            raw_row_data[v] = float(row[v])
                        else:    
                            row_data[v] = row[v]
                            raw_row_data[v] = row[v]
            # print(row, row_data)
            if row['direction'][0] == 'h':
                dir = 1
            else:
                dir = -1
        # print(615, schema, raw_row_data, row_data)
            uq.append({'target_tuple': row_data, 'dir':dir})

            uq[-1]['query_result'] = []
            # break

    return uq, global_patterns, global_patterns_dict

def load_patterns(cur, pat_table_name, query_table_name):
    '''
        load pre-defined constraints(currently only fixed attributes and variable attributes)
    '''
    global_pattern_table = pat_table_name + '_global'
    load_query = "SELECT * FROM {};".format(global_pattern_table)
    # load_query = "SELECT * FROM {} WHERE fixed = '[primary_type]' AND variable = '[community_area, arrest, week]';".format(global_pattern_table)
    
    cur.execute(load_query)
    res = cur.fetchall()
    print(res)
    patterns = []
    pattern_dict = [{}, {}]
    for pat in res:
        if 'date' in pat[0] or 'date' in pat[1]:
            continue
        if 'id' in pat[0] or 'id' in pat[1]:
            continue
        if 'primary_type' in pat[1] or 'description' in pat[1] or 'location_description' in pat[1] or 'community_area' in pat[1] or 'beat' in pat[1]:
            continue
        patterns.append(list(pat))
        # print(pat)
        # print(patterns[-1])
        # patterns[-1][0] = patterns[-1][0][1:-1].replace(' ', '').split(',')
        # patterns[-1][1] = patterns[-1][1][1:-1].replace(' ', '').split(',')
        
        f_key = str(sorted(patterns[-1][0]))
        # f_key = str(patterns[-1][0])
        # print(694, f_key)
        v_key = str(sorted(patterns[-1][1]))
        # v_key = str(patterns[-1][1])
        if v_key not in pattern_dict[0]:
            pattern_dict[0][v_key] = {}
        if f_key not in pattern_dict[0][v_key]:
            pattern_dict[0][v_key][f_key] = []
        pattern_dict[0][v_key][f_key].append(patterns[-1])
        if f_key not in pattern_dict[1]:
            pattern_dict[1][f_key] = {}
        if v_key not in pattern_dict[1][f_key]:
            pattern_dict[1][f_key][v_key] = []
        pattern_dict[1][f_key][v_key].append(patterns[-1])
    schema_query = '''select column_name, data_type, character_maximum_length
        from INFORMATION_SCHEMA.COLUMNS where table_name=\'{}\''''.format(query_table_name);
    cur.execute(schema_query)
    res = cur.fetchall()
    schema = {}
    for s in res:
        schema[s[0]] = s[1]

    return patterns, schema, pattern_dict




def plot_heatmap(hm):
    
    cmap = cm.get_cmap('Spectral')  # Colour map (there are many others)


    x = [i*0.05 for i in range(0, 21)]
    y = [i*0.05 for i in range(0, 21)]
    # setup the 2D grid with Numpy
    x, y = np.meshgrid(x, y)

    # convert intensity (list of lists) to a numpy array for plotting
    intensity = np.array(hm)

    # now just plug the data into pcolormesh, it's that easy!
    plt.pcolormesh(x, y, intensity)
    # plt.xticks(x, list(map(lambda t: str(t*0.05)[:4], x)))
    # plt.yticks(y, list(map(lambda t: str(t*0.05)[:4], y)))
    plt.colorbar()  # need a colorbar to show the intensity scale

    # plt.show()  # boom
    # plt.show()
    # plt.savefig(df_key + '.png')
    l1 = plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, fontsize='x-small')

    pp.savefig(bbox_inches='tight')
    plt.close()


def compute_expl_quality_gt(cur, expl_list):
    if TEST_ID == '_7':
        fixed_list = ['primary_type','beat','district']
        var_list = ['year']
    gt_query = '''
    SELECT * FROM synthetic.crime_exp_changes 
    WHERE num={} AND {} AND {}
    LIMIT 1;
    '''
    gt_score_list = [0 for i in range(len(expl_list))]
    for idx, expl in enumerate(expl_list):
        for expl_k in expl:
            # print(topk[2])
            where_clause_fixed = []
            where_clause_var = []
            year_val = ''
            for col in expl_k.tuple_value.items():
                # print(col)
                if col[0] in fixed_list:
                    where_clause_fixed.append("'{}' = ANY(fval)".format(col[1]))
                else:
                    if col[0] in var_list:
                        where_clause_var.append("('{}'=ANY(vold) OR '{}'=ANY(vnew))".format(col[1], col[1]))
            cur.execute(gt_query.format(
                TEST_ID[1], 
                ' AND '.join(where_clause_fixed), 
                ' AND '.join(where_clause_var)
            ))
            res = cur.fetchall()
            if len(res) > 0:
                gt_score_list[idx] += 1

    return gt_score_list

def main(argv=[]):
    query_result_table = DEFAULT_QUERY_RESULT_TABLE
    pattern_table = DEFAULT_PATTERN_TABLE
    user_question_file = DEFAULT_QUESTION_PATH
    outputfile = ''
    epsilon = DEFAULT_EPSILON
    aggregate_column = DEFAULT_AGGREGATE_COLUMN
    try:
        conn = psycopg2.connect("host=localhost port=5436 dbname=antiprov user=antiprov")
        cur = conn.cursor()
    except psycopg2.OperationalError:
        print('Connection to database failed！')

    
    try:
        opts, args = getopt.getopt(argv,"hq:p:u:o:e:a",["qtable=", "ptable=", "ufile=","ofile=","epsilon=","aggregate_column="])
    except getopt.GetoptError:
        print('explanation.py -q <query_result_table> -p <pattern_table> -u <user_question_file> -o <outputfile> -e <epsilon> -a <aggregate_column>')
        sys.exit(2)

    try:
        opts, args = getopt.getopt(argv,"hq:p:u:o:e:a",["qtable=", "ptable=", "ufile=","ofile=","epsilon=","aggregate_column="])
    except getopt.GetoptError:
        print('explanation.py -q <query_result_table> -p <pattern_table> -u <user_question_file> -o <outputfile> -e <epsilon> -a <aggregate_column>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('explanation.py -q <query_result_table> -p <pattern_table> -u <user_question_file> -o <outputfile> -e <epsilon> -a <aggregate_column>')
            sys.exit(2)    
        elif opt in ("-q", "--qtable"):
            query_result_table = arg
        elif opt in ("-p", "--ptable"):
            pattern_table = arg
        elif opt in ("-u", "--ufile"):
            user_question_file = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-e", "--epsilon"):
            epsilon = float(arg)
        elif opt in ("-a", "--aggcolumn"):
            aggregate_column = arg

    # standard_expl_dict = load_explanations(outputfile)
    # print(standard_expl_dict)
    load_start = time.time()
    global_patterns, schema, global_patterns_dict = load_patterns(cur, pattern_table, query_result_table)
    # category_similarity = CategorySimilarityMatrix(EXAMPLE_SIMILARITY_MATRIX_PATH, schema)
    category_similarity = CategorySimilarityNaive(cur=cur, table_name=query_result_table, embedding_table_list=[('community_area', 'community_area_loc')])
    num_dis_norm = normalize_numerical_distance(cur=cur, table_name=query_result_table)

    Q, global_patterns, global_patterns_dict = load_user_question(
        global_patterns, global_patterns_dict, user_question_file, 
        schema, cur, pattern_table, query_result_table)

    local_patterns_count = [0 for i in range(len(global_patterns))]
    '''load local patterns for each global pattern'''
    for g_id, pat in enumerate(global_patterns):
        local_pattern_query = '''
           SELECT * 
           FROM {} 
           WHERE array_to_string(fixed, ', ')='{}' 
           AND array_to_string(variable, ', ')='{}'
           AND agg='{}' AND model='{}';'''.format(
            pattern_table + '_local', 
            str(pat[0]).replace("\'", '').replace('[', '').replace(']', ''),
            str(pat[1]).replace("\'", '').replace('[', '').replace(']', ''),
            pat[2], pat[3]
        )
        cur.execute(local_pattern_query)
        res = cur.fetchall()
        local_patterns_count[g_id] = len(res)


    quality_lam_the = [[0 for the_i in range(0, 21)] for lam_i in range(0, 21)]
    expl_time_lam_the = [[0 for the_i in range(0, 21)] for lam_i in range(0, 21)]
    quality_dict = dict()
    expl_time_dict = dict()
    ofile = open('expl_gt{}.txt'.format(TEST_ID), 'w')
    lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # lambda_list = [0.1, 0.25, 0.4, 0.55, 0.7]
    theta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # theta_list = [0.1, 0.25, 0.4, 0.55]
    global_supp_list = [1, 5, 15, 25]
    # global_supp_list = [5]
    for lam in lambda_list:
        print('Lambda: ', lam)
        for the in theta_list:
            for gs in global_supp_list:
                exp_key = str(lam)[:4] + ',' + str(the)[:4] + ',' + str(gs)
                ofile.write('Lambda, Theta, Global Support: {}, {}, {}\n'.format(str(lam), str(the), str(gs)))
                expl_start = time.time()
                regression_package = 'statsmodels'
                explanations_list, local_patterns_list, score_computing_time_list = find_explanation_regression_based(
                    Q, global_patterns, global_patterns_dict, category_similarity, 
                    num_dis_norm, epsilon, 
                    aggregate_column, 
                    conn, cur, 
                    pattern_table, query_result_table, lam, the, gs, local_patterns_count)

                expl_end = time.time()
                query_time = expl_end-expl_start

                for i, top_k_list in enumerate(explanations_list):
                    ofile.write('User question {} in direction {}: {}\n'.format(
                        str(i + 1), 'high' if Q[i]['dir'] > 0 else 'low', str(Q[i]['target_tuple']))
                    )

                    for j, e in enumerate(top_k_list):
                        ofile.write('------------------------\n')
                        ofile.write('Top ' + str(j + 1) + ' explanation:\n')
                        ofile.write(e.to_string())
                        ofile.write('------------------------\n')
        

                # quality_dict[exp_key] = compute_expl_quality(standard_expl_dict, explanations_list)
                quality_dict[exp_key] = compute_expl_quality_gt(cur, explanations_list)
                expl_time_dict[exp_key] = sum(map(lambda x: x[1], score_computing_time_list))
                print(quality_dict[exp_key], expl_time_dict[exp_key])
                ofile.write('Running time: {}\n'.format(
                    # str(quality_dict[exp_key]),
                    exp_key,
                    str(expl_time_dict[exp_key])
                ))
                ofile.write('-------------------------------------------\n')
                break
            break
    
    
    for g_key in utils.MATERIALIZED_DICT:
        for fv_key in utils.MATERIALIZED_DICT[g_key]:
            dv_query = '''DROP VIEW IF EXISTS MV_{};'''.format(str(utils.MATERIALIZED_DICT[g_key][fv_key]))
            cur.execute(dv_query)
            conn.commit()

    gt_ofile = open('./gt_params_{}_7_delta_{}.txt'.format(str(TOP_K), str(DEFAULT_LOCAL_SUPPORT)), 'w')
    for ek in quality_dict:
        gt_ofile.write(ek + '\n')
        gt_ofile.write(str(quality_dict[ek]) + '\n')
        gt_ofile.write(str(expl_time_dict[ek]) + '\n')
    gt_ofile.close()
    # plot_heatmap(quality_lam_the)
    # plot_heatmap(expl_time_lam_the)
    # pp.close()




if __name__ == "__main__":
    main(sys.argv[1:])

