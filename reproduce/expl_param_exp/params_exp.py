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


class global_vars:

    MATERIALIZED_DICT = dict()
    MATERIALIZED_CNT = 0
    VISITED_DICT = dict()


TEST_ID = '_7'
DEFAULT_QUESTION_PATH = './exp_parameter/user_question_expl_gt_7.txt'

DEFAULT_QUERY_RESULT_TABLE = 'synthetic.crime_exp' + TEST_ID
DEFAULT_PATTERN_TABLE = 'dev.crime_exp'+ TEST_ID

# DEFAULT_QUERY_RESULT_TABLE = 'crime_exp'
# DEFAULT_PATTERN_TABLE = 'dev.crime_exp'
# # DEFAULT_PATTERN_TABLE = 'crime_2017_2'
# DEFAULT_QUESTION_PATH = '../input/user_question_crime_partial_1.csv'
DEFAULT_LOCAL_SUPPORT = 5
EXAMPLE_NETWORK_EMBEDDING_PATH = '../input/NETWORK_EMBEDDING'
EXAMPLE_SIMILARITY_MATRIX_PATH = '../input/SIMILARITY_DEFINITION'
DEFAULT_AGGREGATE_COLUMN = '*'
DEFAULT_PORT = 5436
TOP_K = 10

def get_tuples_by_F_V(lp1, lp2, f_value, v_value, conn, cur, table_name, cat_sim):
    def tuple_column_to_str_in_where_clause_2(col_value):
        # logger.debug(col_value)
        # logger.debug(cat_sim.is_categorical(col_value[0]))
        if cat_sim.is_categorical(col_value[0]) or col_value[0] == 'year':
            # return "like '%" + (
            #     str(col_value[1]).replace('.0', '') if col_value[1][-2:] == '.0' else str(col_value[1])) + "%'"
            return "= '" + str(col_value[1]) + "'"
        else:
            if is_float(col_value[1]):
                return '=' + str(col_value[1])
            else:
                # return "like '%" + str(col_value[1]) + "%'"
                return "= '" + str(col_value[1]) + "'"

    def tuple_column_to_str_in_where_clause_3(col_value):
        # logger.debug(col_value)
        # logger.debug(cat_sim.is_categorical(col_value[0]))
        if cat_sim.is_categorical(col_value[0]) or col_value[0] == 'year':
            # return "like '%" + (
            #     str(col_value[1]).replace('.0', '') if col_value[1][-2:] == '.0' else str(col_value[1])) + "%'"
            return "= '" + str(col_value[1]) + "'"
        else:
            if is_float(col_value[1]):
                return '>=' + str(col_value[1])
            else:
                return "= '" + str(col_value[1]) + "'"

    def tuple_column_to_str_in_where_clause_4(col_value):
        # logger.debug(col_value)
        # logger.debug(cat_sim.is_categorical(col_value[0]))
        if cat_sim.is_categorical(col_value[0]) or col_value[0] == 'year':
            # return "like '%" + (
            #     str(col_value[1]).replace('.0', '') if col_value[1][-2:] == '.0' else str(col_value[1])) + "%'"
            return "= '" + str(col_value[1]) + "'"
        else:
            if is_float(col_value[1]):
                return '<=' + str(col_value[1])
            else:
                return "= '" + str(col_value[1]) + "'"

    V1 = str(lp1[2]).replace("\'", '')[1:-1]
    F1 = str(lp1[0]).replace("\'", '')[1:-1]
    F1_list = F1.split(', ')
    V1_list = V1.split(', ')

    F2 = str(lp2[0]).replace("\'", '')[1:-1]
    V2 = str(lp2[2]).replace("\'", '')[1:-1]
    F2_list = F2.split(', ')
    V2_list = V2.split(', ')
    G_list = sorted(F2_list + V2_list)
    G_key = str(G_list).replace("\'", '')[1:-1]
    f_value_key = str(f_value).replace("\'", '')[1:-1]

    if lp2[3] == 'count':
        agg_fun = 'count(*)'
    else:
        agg_fun = lp2[3].replace('_', '(') + ')'

    if G_key not in global_vars.MATERIALIZED_DICT:
        global_vars.MATERIALIZED_DICT[G_key] = dict()

    if f_value_key not in global_vars.MATERIALIZED_DICT[G_key]:
        global_vars.MATERIALIZED_DICT[G_key][f_value_key] = global_vars.MATERIALIZED_CNT
        dv_query = '''DROP VIEW IF EXISTS MV_{};'''.format(str(global_vars.MATERIALIZED_CNT))
        cur.execute(dv_query)
        cmv_query = '''
            CREATE VIEW MV_{} AS SELECT {}, {} as {} FROM {} WHERE {} GROUP BY {};
        '''.format(
            str(global_vars.MATERIALIZED_CNT), G_key, agg_fun, lp2[3], table_name,
            ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]),
                                  zip(lp1[0], map(tuple_column_to_str_in_where_clause_2, zip(F1_list, f_value)))))),
            G_key
        )

        cur.execute(cmv_query)
        conn.commit()
        global_vars.MATERIALIZED_CNT += 1

    where_clause = ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]), zip(lp1[0], map(
        tuple_column_to_str_in_where_clause_2, zip(F1_list, f_value))))))

    if v_value is not None:
        where_clause += ' AND '
        v_range_l = map(lambda x: v_value[0][x] + v_value[1][x][0], range(len(v_value[0])))
        v_range_r = map(lambda x: v_value[0][x] + v_value[1][x][1], range(len(v_value[0])))
        where_clause += ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]),
                                              zip(lp1[2],
                                                  map(tuple_column_to_str_in_where_clause_3,
                                                      zip(V1_list, v_range_l))))))
        where_clause += ' AND '
        where_clause += ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]),
                                              zip(lp1[2],
                                                  map(tuple_column_to_str_in_where_clause_4,
                                                      zip(V1_list, v_range_r))))))

    tuples_query = '''SELECT {},{},{} FROM MV_{} WHERE {};'''.format(
        F2, V2, lp2[3], str(global_vars.MATERIALIZED_DICT[G_key][f_value_key]), where_clause
    )

    column_name = F2_list + V2_list + [lp2[3]]
    cur.execute(tuples_query)
    # logger.debug(tuples_query)
    tuples = []
    tuples_dict = dict()
    res = cur.fetchall()
    min_agg = 1e10
    max_agg = -1e10
    for row in res:
        min_agg = min(min_agg, row[-1])
        max_agg = max(max_agg, row[-1])
        tuples.append(dict(zip(map(lambda x: x, column_name), row)))
        fv = get_F_value(lp2[0], tuples[-1])
        f_key = str(fv).replace('\'', '')[1:-1]
        if f_key not in tuples_dict:
            tuples_dict[f_key] = []
        tuples_dict[f_key].append(tuples[-1])
    # logger.debug(tuples_dict)
    return tuples, (min_agg, max_agg), tuples_dict


def get_tuples_by_gp_uq(gp, f_value, v_value, conn, cur, table_name, cat_sim):
    def tuple_column_to_str_in_where_clause_2(col_value):
        # logger.debug(col_value)
        # logger.debug(cat_sim.is_categorical(col_value[0]))
        if cat_sim.is_categorical(col_value[0]) or col_value[0] == 'year':
            # return "like '%" + (
            #     str(col_value[1]).replace('.0', '') if col_value[1][-2:] == '.0' else str(col_value[1])) + "%'"
            return "= '" + str(col_value[1]) + "'"
        else:
            if is_float(col_value[1]):
                return '=' + str(col_value[1])
            else:
                # return "like '%" + str(col_value[1]) + "%'"
                return "= '" + str(col_value[1]) + "'"

    F1 = str(gp[0]).replace("\'", '')[1:-1]
    V1 = str(gp[1]).replace("\'", '')[1:-1]
    F1_list = F1.split(', ')
    V1_list = V1.split(', ')
    G_list = sorted(F1_list + V1_list)
    G_key = str(G_list).replace("\'", '')[1:-1]
    f_value_key = str(f_value).replace("\'", '')[1:-1]

    if gp[2] == 'count':
        agg_fun = 'count(*)'
    else:
        agg_fun = gp[2].replace('_', '(') + ')'

    if G_key not in global_vars.MATERIALIZED_DICT:
        global_vars.MATERIALIZED_DICT[G_key] = dict()

    if f_value_key not in global_vars.MATERIALIZED_DICT[G_key]:
        global_vars.MATERIALIZED_DICT[G_key][f_value_key] = global_vars.MATERIALIZED_CNT
        dv_query = '''DROP VIEW IF EXISTS MV_{};'''.format(str(global_vars.MATERIALIZED_CNT))
        cur.execute(dv_query)

        cmv_query = '''
            CREATE VIEW MV_{} AS SELECT {}, {} as {} FROM {} WHERE {} GROUP BY {};
        '''.format(
            str(global_vars.MATERIALIZED_CNT), G_key, agg_fun, gp[2], table_name,
            ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]),
                                  zip(gp[0], map(tuple_column_to_str_in_where_clause_2, zip(F1_list, f_value)))))),
            G_key
        )
        # logger.debug(cmv_query)
        cur.execute(cmv_query)
        conn.commit()
        global_vars.MATERIALIZED_CNT += 1

    where_clause = ' AND '.join(
        list(map(lambda x: "{} {}".format(x[0], x[1]), zip(gp[0], map(
            tuple_column_to_str_in_where_clause_2, zip(F1_list, f_value)))))) + ' AND ' + \
                   ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]),
                                         zip(gp[1],
                                             map(tuple_column_to_str_in_where_clause_2, zip(V1_list, v_value))))))

    tuples_query = '''SELECT {} FROM MV_{} WHERE {};'''.format(
        gp[2], str(global_vars.MATERIALIZED_DICT[G_key][f_value_key]), where_clause
    )
    # logger.debug(tuples_query)
    cur.execute(tuples_query)
    res = cur.fetchall()
    if len(res) == 0:
        return []
    return res[0]

def get_local_patterns(F, Fv, V, agg_col, model_type, t, conn, cur, local_pat_table_name):

    if model_type is not None:
        mt_predicate = " AND model='{}'".format(model_type)
    else:
        mt_predicate = ''
    if Fv is not None:
        local_pattern_query = '''SELECT * FROM {} WHERE array_to_string(fixed, ', ')='{}' AND 
            REPLACE(array_to_string(fixed_value, ', '), '"', '') LIKE '%{}%' AND 
            array_to_string(variable, ', ') = '{}' AND
            agg='{}'{};
        '''.format(
            local_pat_table_name, str(F).replace("\'", '').replace('[', '').replace(']', ''),
            str(Fv).replace("\'", '').replace('[', '').replace(']', ''),
            str(V).replace("\'", '').replace('[', '').replace(']', ''),
            agg_col,
            mt_predicate
        )
    else:
        tF = get_F_value(F, t)
        local_pattern_query = '''SELECT * FROM {} WHERE array_to_string(fixed, ', ')='{}' AND
            REPLACE(array_to_string(fixed_value, ', '), '"', '') LIKE '%{}%' AND array_to_string(variable, ', ')='{}' AND
            agg='{}'{};
        '''.format(
            local_pat_table_name, str(F).replace("\'", '').replace('[', '').replace(']', ''),
            '%'.join(list(map(str, tF))),
            str(V).replace("\'", '').replace('[', '').replace(']', ''),
            agg_col,
            mt_predicate
        )

    cur.execute(local_pattern_query)
    local_patterns = cur.fetchall()

    return local_patterns


def find_patterns_refinement(global_patterns_dict, F_prime_set, V_set, agg_col, reg_type):
    # pattern refinement can have different model types
    # e.g., JH’s #pub increases linearly, but JH’s #pub on VLDB remains a constant
    gp_list = []
    v_key = str(sorted(list(V_set)))
    if v_key not in global_patterns_dict[0]:
        return []
    for f_key in global_patterns_dict[0][v_key]:
        if f_key.find('arrest') != -1 or f_key.find('domestic') != -1:
            continue
        F_key_set = set(f_key[1:-1].replace("'", '').split(', '))
        if F_prime_set.issubset(F_key_set):
            for pat in global_patterns_dict[0][v_key][f_key]:
                if pat[2] == agg_col:
                    pat_key = f_key + '|,|' + v_key + '|,|' + pat[2] + '|,|' + pat[3]

                    gp_list.append(pat)
                    if pat_key not in global_vars.VISITED_DICT:
                        global_vars.VISITED_DICT[pat_key] = True
    return gp_list


def find_patterns_relevant(global_patterns_dict, t, conn, cur, query_table_name, cat_sim):
    res_list = []
    t_set = set(t.keys())
    for v_key in global_patterns_dict[0]:
        V_set = set(v_key[1:-1].replace("'", '').split(', '))
        if not V_set.issubset(t_set):
            continue

        for f_key in global_patterns_dict[0][v_key]:
            for pat in global_patterns_dict[0][v_key][f_key]:
                F_set = set(f_key[1:-1].replace("'", '').split(', '))
                if not F_set.issubset(t_set):
                    continue
                # if pat[2] not in t and pat[2] + '_star' not in t:
                #     continue
                if pat[2] not in t:
                    continue

                agg_value = get_tuples_by_gp_uq(pat, get_F_value(pat[0], t), get_V_value(pat[1], t),
                                                conn, cur, query_table_name, cat_sim)
                if len(agg_value) > 0:
                    res_list.append([pat, agg_value[0]])

    res_list = sorted(res_list, key=lambda x: (len(x[0][0]) + len(x[0][1]), x[1]))
    g_pat_list = list(map(lambda x: x[0], res_list))
    return g_pat_list


def load_patterns(cur, pat_table_name, query_table_name, theta_thres=0.1, lambda_thres=0.1):
    '''
        load pre-defined constraints(currently only fixed attributes and variable attributes)
    '''
    global_pattern_table = pat_table_name + '_global'
    load_query = "SELECT * FROM {};".format(global_pattern_table)

    cur.execute(load_query)
    res = cur.fetchall()
    patterns = []
    pattern_dict = [{}, {}]
    for pat in res:
        if 'date' in pat[0] or 'date' in pat[1]:
            continue
        if 'id' in pat[0] or 'id' in pat[1]:
            continue
        if 'year' in pat[0]:
            continue
        if 'name' in pat[1] or 'venue' in pat[1]:
            continue
        if 'primary_type' in pat[1] or 'description' in pat[1] or 'location_description' in pat[1] or 'community_area' in pat[1] or 'beat' in pat[1]:
            continue

        patterns.append(list(pat))

        f_key = str(sorted(patterns[-1][0]))
        v_key = str(sorted(patterns[-1][1]))
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



def predict(local_pattern, t):
    # print('In predict ', local_pattern)
    if local_pattern[4] == 'const':
        predictY = float(local_pattern[-4][1:-1].split(',')[0])
    elif local_pattern[4] == 'linear':
        # print(local_pattern, t)
        v = get_V_value(local_pattern[2], t)
        if isinstance(local_pattern[-3], str):
            params_str = local_pattern[-3].split('\n')
            params_dict = {}
            for i in range(0, len(params_str)-1):
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
            params_dict = local_pattern[-3]

        predictY = params_dict['Intercept']
        for v_attr in t:
            v_key = '{}[T.{}]'.format(v_attr, t[v_attr])
            if v_key in params_dict:
                predictY += params_dict[v_key]
            else:
                if v_attr in params_dict:
                    predictY += params_dict[v_attr] * float(t[v_attr])

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
            if col == 'name':
                dis += 10000
                cnt += 1
            continue
        if col not in t1 or col not in t2:
            if col == 'name':
                dis += 10000
                cnt += 1
            continue
    
        if cat_sim.is_categorical(col) and col != 'year':
            
            t1_key = str(t1[col]).replace("'", '').replace(' ', '')
            t2_key = str(t2[col]).replace("'", '').replace(' ', '')
            s = 0
            if t1[col] == t2[col]:
                s = 1
            else:
                s = cat_sim.compute_similarity(col, t1_key, t2_key, agg_col)
            if s == 0:
                dis += 1
            else:
                dis += (((1.0/s)) * ((1.0/s))) / 100
                
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
                        diff = datetime.datetime(t1[col].year, t1[col].month, t1[col].day) - datetime.datetime.strptime(t2[col], "%Y-%m-%d")
                        temp = diff.days
                    else:
                        temp = abs(float(t1[col]) - float(t2[col]))
                    
                    dis += 0.5 * math.pow(temp, temp+5)
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
    else:
        if DEFAULT_LOCAL_SUPPORT != 5:
            if not validate_local_support(F, V, list(map(str, tF)), cur, res_table_name, cat_sim):
                return []
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
        agg_col = lp1[3]
        t1fv = dict()
        t2fv = dict()
        for a in lp2[0] + lp2[2]:
            t1fv[a] = t1[a]
            if a in t2:
                t2fv[a] = t2[a]
        
        t_dis_raw = tuple_distance(t1fv, t2fv, None, cat_sim, num_dis_norm, agg_col)
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
        t_dis = math.sqrt(t_dis_raw * t_dis_raw + w * diff * diff)
            
        t1v = dict(zip(lp1[2], map(lambda x:x, get_V_value(lp1[2], t1))))        
        predicted_agg1 = predict(lp1, t1v)
        t2v = dict(zip(lp2[2], map(lambda x:x, get_V_value(lp2[2], t2))))        
        predicted_agg2 = predict(lp2, t2v)
        # deviation - counterbalance_needed 
        deviation = float(t1[agg_col]) - predicted_agg1
        
        if t_dis == 0:
            score = deviation * -dir
        else:
            score = deviation / t_dis * -dir
        
        return [100 * score / float(denominator), t_dis, deviation, float(denominator), t_dis_raw]

def compare_tuple(t1, t2):
    flag1 = True
    for a in t1:
        # if (a != 'lambda' and a.find('_') == -1):
        if a != 'lambda' and not a.startswith('count') and not a.startswith('sum'):
            if a not in t2:
                flag1 = False
            elif t1[a] != t2[a]:
                return 0
    flag2 = True
    for a in t2:
        # if (a != 'lambda' and a.find('_') == -1):
        if a != 'lambda' and not a.startswith('count') and not a.startswith('sum'):
            if a not in t1:
                flag2 = False
            elif t1[a] != t2[a]:
                return 0

    if flag1 and flag2:
        return -1
    elif flag1:
        return -1
    else:
        return 0

# def compare_tuple(t1, t2):
#     flag1 = True
#     for a in t1:
#         # if (a != 'lambda' and a.find('_') == -1):
#         if (a != 'lambda' and a != 'count'):
#             if a not in t2:
#                 flag1 = False
#             elif t1[a] != t2[a]:
#                 return 0
#     flag2 = True
#     for a in t2:
#         # if (a != 'lambda' and a.find('_') == -1):
#         if (a != 'lambda' and a != 'count'):
#             if a not in t1:
#                 flag2 = False
#             elif t1[a] != t2[a]:
#                 return 0
#     # print(t1, t2, flag1, flag2)
#     if flag1 and flag2:
#         return -1
#     elif flag1:
#         return -1
#     else:
#         return 0

def DrillDown(global_patterns_dict, local_pattern, F_set, U_set, V_set, t_prime_coarser, t_coarser, t_prime, target_tuple,
        conn, cur, pat_table_name, res_table_name, cat_sim, num_dis_norm, dir, query_result, theta_lb=0.1):
    reslist = []
    F_prime_set = F_set.union(U_set)
    agg_col = local_pattern[3]
    gp2_list = find_patterns_refinement(global_patterns_dict, F_set, V_set, local_pattern[3], local_pattern[4])
    
    if len(gp2_list) == 0:
        return []
    for gp2 in gp2_list:

        # lp2_list = get_local_patterns(gp2[0], None, gp2[1], gp2[2], None, t_prime, conn, cur, pat_table_name, res_table_name)
        lp2_list = get_local_patterns(gp2[0], None, gp2[1], gp2[2], gp2[3], t_prime, conn, cur, pat_table_name, res_table_name, cat_sim, theta_lb)
        
        if len(lp2_list) == 0:
            continue
        lp2 = lp2_list[0]
        
        
        f_value = get_F_value(local_pattern[0], t_prime)
        tuples_same_F, agg_range, tuples_same_F_dict = get_tuples_by_F_V(local_pattern, lp2, f_value, 
            None,
            conn, cur, res_table_name, cat_sim)
        lp3_list = get_local_patterns(lp2[0], f_value, lp2[2], lp2[3], lp2[4], t_prime, conn, cur, pat_table_name, res_table_name, cat_sim, theta_lb)
        
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

    
    return reslist

def find_explanation_regression_based(user_question_list, global_patterns, global_patterns_dict, 
    cat_sim, num_dis_norm, agg_col, conn, cur, pat_table_name, res_table_name,
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
        
        local_patterns = []
        
        
        end = time.time()
        local_pattern_loading_time += end - start

        candidate_list = [[] for i in range(len(uq['global_patterns']))]
        top_k_lists = [[] for i in range(len(uq['global_patterns']))]
        validate_res_list = []
        # local_patterns_list.append(local_patterns)
        local_patterns = []
        
        psi = []
        
        global_vars.VISITED_DICT = dict()
        score_computing_time_cur_uq = 0
        score_computing_start = time.time()
        explanation_type = 0
        for i in range(0, len(uq['global_patterns'])):
            top_k_lists[i] = [4, uq['global_patterns'][i], t, []]
            local_patterns.append(None)
            F_key = str(sorted(uq['global_patterns'][i][0]))
            V_key = str(sorted(uq['global_patterns'][i][1]))
            pat_key = F_key + '|,|' + V_key + '|,|' + uq['global_patterns'][i][2] + '|,|' + uq['global_patterns'][i][3]
            
            if pat_key in global_vars.VISITED_DICT:
                continue
            global_vars.VISITED_DICT[pat_key] = True

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
            # print(476, res_count[0][0])
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
            if len(res_fixed) == 0:
                continue
            local_patterns[i] = res_fixed[0]
            # if len(local_patterns[i][2]) > 1 or local_patterns[i][2][0] != 'year':
            #     continue
            T_set = set(t.keys()).difference(set(['lambda', uq['global_patterns'][i][2]]))
            psi.append(0)
            agg_col = local_patterns[i][3]
            start = time.time()
            
            t_t_list, agg_range, t_t_dict = get_tuples_by_F_V(local_patterns[i], local_patterns[i], 
                get_F_value(local_patterns[i][0], t), 
                #[get_V_value(local_patterns[i][2], t), [[-3, 3]]],
                None,
                conn, cur, res_table_name, cat_sim)
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
            
            # F union V \subsetneq G
            
            start = time.time()
            
            # print(t_coarser)
            t_coarser_copy = list(t_t_list)
            
            norm_lb = min(list(map(lambda x: x[agg_col], t_coarser_copy)))
            
            k_score = topK_heap.MinValue()
            
        
            if topK_heap.HeapSize() == TOP_K and 100 * float(dev_ub) / (dist_lb * float(norm_lb)) <= k_score:
                continue
            top_k_lists[i][-1] += DrillDown(global_patterns_dict, local_patterns[i], 
                F_set, T_set.difference(F_set.union(V_set)), V_set, t_coarser_copy, t_coarser_copy, t, t,
                conn, cur, pat_table_name, res_table_name, cat_sim, num_dis_norm,
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
        start = time.time()
        answer[j] = [{} for i in range(TOP_K)]
        score_computing_time_list.append([t, score_computing_time_end - score_computing_time_start])
        
        answer[j] = topK_heap.TopK()
        end = time.time()
        result_merging_time += end - start
        
    return answer, local_patterns_list, score_computing_time_list

def load_user_question(global_patterns, global_patterns_dict, uq_path=DEFAULT_QUESTION_PATH, schema=None, cur=None, pattern_table='', query_result_table=''):
    '''
        load user questions
    '''
    # print(global_patterns_dict)
    uq = []

    with open(uq_path, 'rt') as uqfile:
        reader = csv.DictReader(uqfile, quotechar='\'')
        headers = reader.fieldnames
        #temp_data = csv.reader(uqfile, delimiter=',', quotechar='\'')
        #for row in temp_data:
        for row in reader:
            row_data = {}
            raw_row_data = {}
            agg_col = None
            for v in row:
                # print(k, v)
                if schema is None or v not in schema:
                    if v != 'direction':
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
            if row['direction'][0] == 'h':
                dir = 1
            else:
                dir = -1
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
        
        f_key = str(sorted(patterns[-1][0]))
        v_key = str(sorted(patterns[-1][1]))
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

def compute_expl_quality_gt(cur, expl_list):    
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
    resultfile = './output/expl_params_top_{}_delta_{}.txt'.format(str(TOP_K), str(DEFAULT_LOCAL_SUPPORT))
    host = 'localhost'
    port = DEFAULT_PORT
    aggregate_column = DEFAULT_AGGREGATE_COLUMN
    
    
    try:
        opts, args = getopt.getopt(argv,"h:q:p:u:o:r:P:a",["host=", "qtable=", "ptable=", "ufile=","ofile=", "rtfile=", "port=", "aggregate_column="])
    except getopt.GetoptError:
        print('params_exp.py -h <host> -q <query_result_table> -p <pattern_table> -u <user_question_file> -o <outputfile> -r <resultfile> -P <port> -a <aggregate_column>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--host"):
            host = arg
        elif opt in ("-q", "--qtable"):
            query_result_table = arg
        elif opt in ("-p", "--ptable"):
            pattern_table = arg
        elif opt in ("-u", "--ufile"):
            user_question_file = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-r", "--rtfile"):
            resultfile = arg
        elif opt in ("-P", "--port"):
            port = float(arg)
        elif opt in ("-a", "--aggcolumn"):
            aggregate_column = arg

    try:
        conn = psycopg2.connect("host={} port={} dbname=antiprov user=antiprov password=antiprov".format(host, str(port)))
        cur = conn.cursor()
    except psycopg2.OperationalError:
        print('Connection to database failed！')


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
                print('Lambda, Theta, Global Support: {}, {}, {}'.format(str(lam), str(the), str(gs)))
                ofile.write('Lambda, Theta, Global Support: {}, {}, {}\n'.format(str(lam), str(the), str(gs)))
                expl_start = time.time()
                regression_package = 'statsmodels'
                explanations_list, local_patterns_list, score_computing_time_list = find_explanation_regression_based(
                    Q, global_patterns, global_patterns_dict, category_similarity, 
                    num_dis_norm, 
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
                # print(quality_dict[exp_key], expl_time_dict[exp_key])
                ofile.write('Running time: {}\n'.format(
                    # str(quality_dict[exp_key]),
                    exp_key,
                    str(expl_time_dict[exp_key])
                ))
                ofile.write('-------------------------------------------\n')
            #     break
            # break
    
    
    for g_key in global_vars.MATERIALIZED_DICT:
        for fv_key in global_vars.MATERIALIZED_DICT[g_key]:
            dv_query = '''DROP VIEW IF EXISTS MV_{};'''.format(str(global_vars.MATERIALIZED_DICT[g_key][fv_key]))
            cur.execute(dv_query)
            conn.commit()

    gt_ofile = open(resultfile, 'w')
    for ek in quality_dict:
        gt_ofile.write(ek + '\n')
        gt_ofile.write(str(quality_dict[ek]) + '\n')
        gt_ofile.write(str(expl_time_dict[ek]) + '\n')
    gt_ofile.close()


if __name__ == "__main__":
    main(sys.argv[1:])

