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
# import matplotlib.pyplot as plt

sys.path.append('./')
# sys.path.append('../')
from similarity_calculation.category_similarity_matrix import *
from similarity_calculation.category_network_embedding import *
from similarity_calculation.category_similarity_naive import *
from utils import *
from constraint_definition.LocalRegressionConstraint import *
from PatternDetection.PatternFinderForExpl import PatternFinder

TEST_ID = '_1'

class TopkHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []
 
    def Push(self, elem):
        if len(self.data) < self.k:

            heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem[0] > topk_small[0]:
                # for i in range(1, len(elem)):
                    # if elem[i] is not None:
                    #     elem[i] = str(elem[i])
                heapreplace(self.data, elem)
    def MinValue(self):
        return min(list(map(lambda x: x[0], self.data)))
    def MaxValue(self):
        return max(list(map(lambda x: x[0], self.data)))
    def TopK(self):
        return [x for x in reversed([heappop(self.data) for x in range(len(self.data))])]
    def HeapSize(self):
        return len(self.data)


# DEFAULT_QUERY_RESULT_TABLE = 'crime_2017'

# DEFAULT_QUERY_RESULT_TABLE = 'pub_large_no_domain'
# DEFAULT_PATTERN_TABLE = 'pub.pub_large_no_domain'
# DEFAULT_QUESTION_PATH = '../input/user_question_pub_large.csv'

# DEFAULT_QUERY_RESULT_TABLE = 'publication_vldb'
# DEFAULT_PATTERN_TABLE = 'crime.publication_vldb'
# DEFAULT_QUESTION_PATH = '../input/user_question_pub_small.csv'
# DEFAULT_QUESTION_PATH = '../input/user_question_pub_large_gen.csv'

DEFAULT_QUERY_RESULT_TABLE = 'crime_subset'
DEFAULT_PATTERN_TABLE = 'dev.crime_subset'
# DEFAULT_QUERY_RESULT_TABLE = 'crime_clean_100000_2'
# DEFAULT_PATTERN_TABLE = 'dev.crime_clean_100000'
# DEFAULT_QUERY_RESULT_TABLE = 'crime_partial'
# DEFAULT_PATTERN_TABLE = 'dev.crime_partial'
# DEFAULT_PATTERN_TABLE = 'crime_2017_2'
# DEFAULT_QUESTION_PATH = '../input/user_question_crime_qual_small.csv'
# DEFAULT_QUESTION_PATH = '../input/user_question_crime_subset_gen_ordered_6.csv'
DEFAULT_QUESTION_PATH = '../input/user_question_crime_subset_gen_new_6.csv'
# DEFAULT_QUERY_RESULT_TABLE = 'crime_exp'
# DEFAULT_PATTERN_TABLE = 'dev.crime_exp'
# # DEFAULT_PATTERN_TABLE = 'crime_2017_2'
# DEFAULT_QUESTION_PATH = '../input/user_question_crime_partial_1.csv'

EXAMPLE_NETWORK_EMBEDDING_PATH = '../input/NETWORK_EMBEDDING'
EXAMPLE_SIMILARITY_MATRIX_PATH = '../input/SIMILARITY_DEFINITION'
DEFAULT_USER_QUESTION_NUMBER = 5
DEFAULT_AGGREGATE_COLUMN = '*'
DEFAULT_EPSILON = 0.25
DEFAULT_LAMBDA = 0.5
TOP_K = 3
PARAMETER_DEV_WEIGHT = 1.0
global MATERIALIZED_CNT
MATERIALIZED_CNT = 0
global MATERIALIZED_DICT
MATERIALIZED_DICT = dict()
global VISITED_DICT
VISITED_DICT = dict()


def build_local_pattern(g_pat, t, tF, conn, cur, table_name):
    """Build local regression constraint from Q(R), t, and global regression constraint

    Args:
        data: result of Q(R)
        column_index: index for values in each column
        t: target tuple in Q(R)
        con: con[0] is the list of fixed attributes in Q(R), con[1] is the list of variable attributes in Q(R)
        epsilon: threshold for local regression constraint
        regression_package: which package is used to compute regression 
    Returns:
        A LocalRegressionConstraint object whose model is trained on \pi_{con[1]}(Q_{t[con[0]]}(R))
    """
    # agg_col = '{}({})'.format(g_pat[3], g_pat[2])
    agg_col = g_pat[2]
    data_points_query = 'SELECT {},{},{} as {} FROM {} WHERE {} GROUP BY {},{}'.format(
        ','.join(g_pat[0]), ','.join(g_pat[1]), agg_col, g_pat[3], 
        table_name, 
        ' AND '.join(list(map(lambda x: "{}='{}'".format(x, str(t[x])), g_pat[0]))),
        ','.join(g_pat[0]), ','.join(g_pat[1])
    )
    df=pandas.read_sql(data_points_query, con=conn)

    print(data_points_query)
    print(df)
    if df.empty:
        return None
    formula = g_pat[3] + ' ~ ' + ' + '.join(g_pat[1])
    print(formula)
    lr=sm.ols(formula, data=df).fit()
    theta_l=lr.rsquared_adj

    describe=[mean(df[g_pat[3]]),mode(df[g_pat[3]]),percentile(df[g_pat[3]],25)
                      ,percentile(df[g_pat[3]],50),percentile(df[g_pat[3]],75)]

    l_pat = [g_pat[0], tF, g_pat[1], g_pat[2], g_pat[3], 
        'linear', theta_l, str(describe).replace("'",""), str(lr.params).replace("'", "")]
    print(l_pat)
    return l_pat

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

def validate_local_regression_pattern(local_pattern, epsilon, t, dir, agg_col, cur, table_name):
    """Check the validicity of the user question under a local regression constraint

    Args:
        local_pattern:
        t: target tuple in Q(R)
        dir: whether user thinks t[agg_col] is high or low
        agg_col: the column of aggregated value
    Returns:
        the actual direction that t[agg_col] compares to its expected value wrt to local_pattern
    """
    test_tuple = {}
    # print('PAT', local_pattern)
    if isinstance(local_pattern, dict):
        return -2, 0
    for v in local_pattern[2]:
        test_tuple[v] = t[v.replace(' ', '')]
    # if regression_package == 'scikit-learn':
    #     for v in local_con.var_attr:
    #         if v in data['le']:
    #             test_tuple[v] = data['le'][v].transform(test_tuple[v])
    #             test_tuple[v] = data['ohe'][v].transform(test_tuple[v].reshape(-1, 1))
    #         else:
    #             test_tuple[v] = np.array(test_tuple[v]).reshape(-1, 1)
        
    #     test_tuple = np.concatenate(list(test_tuple.values()), axis=-1)
    #     predictY = local_con.predict_sklearn(test_tuple)
    # else:
    #     predictY = local_con.predict(pandas.DataFrame(test_tuple))
    
    if agg_col not in t:
        # agg_col = '{}({})'.format(local_pattern[4], local_pattern[3])
        agg_col = local_pattern[3]
    predictY = predict(local_pattern, test_tuple)
    # print(local_pattern)
    # print(181, t, test_tuple, predictY)
    if t[agg_col] < (1-epsilon) * predictY:
        # print(test_tuple, predictY)
        return -dir, predictY
    elif t[agg_col] > (1+epsilon) * predictY:
        # print(test_tuple, predictY)
        return dir, predictY
    else:
        return 0, predictY
        
def tuple_similarity(t1, t2, var_attr, cat_sim, num_dis_norm, agg_col):
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
    sim = 0.0
    cnt = 0
    if var_attr is None:
        var_attr = t1.keys()
    for v_col in var_attr:
        col = v_col.replace(' ', '')
        
        if (col not in t1 or col not in t2) and col != agg_col and col != 'lambda':
            # if col == 'name':
            #     cnt += 3
            # else:
                # cnt += 1
            cnt += 1
            continue
        if cat_sim.is_categorical(col):
            t1_key = t1[col].replace("'", '').replace(' ', '')
            t2_key = t2[col].replace("'", '').replace(' ', '')
            s = cat_sim.compute_similarity(col, t1_key, t2_key, agg_col)
            w = 1
            if col == 'name':
                w  = 5
            sim += w * s
            cnt += w * 1
            # print(1, col, sim, t1[col], t2[col])
        else:
            # print( num_dis_norm[col])
            if col not in num_dis_norm or num_dis_norm[col]['range'] is None:
                if t1[col] == t2[col]:
                    sim += 1
                    cnt += 1
            else:
                if col != agg_col and col != 'index':
                    # temp = (t1[col] - t2[col]) * (t1[col] - t2[col]) / (num_dis_norm[col]['range'] * num_dis_norm[col]['range'])
                    temp = abs(t1[col] - t2[col]) / (num_dis_norm[col]['range'])
                    # temp = abs(t1[col] - t2[col])
                    # if t1[col] == t2[col]:
                    #     sim += 1
                    # else:
                    #     sim += max(math.log(1-temp), 0)
                    # ((E^x-x) -1) / (E - 2)
                    # x = 1 - temp
                    # y = ((math.exp(x) - x) - 1) / (math.exp(1) - 2)
                    x = temp
                    if x == 0:
                        y = 1
                    else:
                        y = (1-x) * (1-x)
                        # y = 1-1/(1+(math.exp(-x+1)))
                        # y = (1-1/(1+(math.exp(-x+1)))) * (x / (num_dis_norm[col]['range']))
                    w = 5
                    sim += w * y
                    cnt += w
            
                    # sim += x * x * x * x
            # print(2, col, sim, t1[col], t2[col])
        
    # print(t1, t2, var_attr)
    return (sim) / (cnt )


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
    
        if cat_sim.is_categorical(col):
            
            t1_key = str(t1[col]).replace("'", '').replace(' ', '')
            t2_key = str(t2[col]).replace("'", '').replace(' ', '')
            s = 0
            if t1[col] == t2[col]:
                s = 1
            else:
                s = cat_sim.compute_similarity(col, t1_key, t2_key, agg_col)
            # print(359, col, t1, t2, s)
            # if col == 'community_area':
            #     s = 
            if s == 0:
                dis += 1
                max_dis = 1
            else:
                dis += (((1.0/s)) * ((1.0/s))) / 100
                # dis += (1-s) * (1-s)
                if math.sqrt((((1.0/s)) * ((1.0/s)) - 1) / 100) > max_dis:
                    max_dis = math.sqrt((((1.0/s)) * ((1.0/s)) - 1) / 100)
            # if s == 0:
            #     dis += 1
            #     max_dis = 1
            # else:
            #     dis += (((1.0/s)) * ((1.0/s))) / 100
            #     # dis += (1-s) * (1-s)
            #     if math.sqrt((((1.0/s)) * ((1.0/s)) - 1) / 100) > max_dis:
            #         max_dis = math.sqrt((((1.0/s)) * ((1.0/s)) - 1) / 100)
            cnt += 1
        else:
            # print( num_dis_norm[col])
            if col not in num_dis_norm or num_dis_norm[col]['range'] is None:
                if t1[col] == t2[col]:
                    dis += 0
                else:
                    dis += 1
            else:
                if col != agg_col and col != 'index':
                    # temp = (t1[col] - t2[col]) * (t1[col] - t2[col]) / (num_dis_norm[col]['range'] * num_dis_norm[col]['range'])
                    # temp = abs(t1[col] - t2[col]) / (num_dis_norm[col]['range'])
                    if isinstance(t1[col], datetime.date):
                        # print(398, t1, t2)
                        # print(col, t1[col], t2[col])
                        diff = datetime.datetime(t1[col].year, t1[col].month, t1[col].day) - datetime.datetime.strptime(t2[col], "%Y-%m-%d")
                        temp = diff.days
                    else:
                        # print(398, t1, t2)
                        # print(col, t1[col], t2[col])
                        temp = abs(float(t1[col]) - float(t2[col]))
                    # if temp != 0:
                    #     temp = 1/(1+(math.exp(-temp+2)))

                    dis += 0.5 * math.pow(temp, 8)
                    if temp > max_dis:
                        max_dis = temp
                cnt += 1
                    # sim += x * x * x * x
            # print(2, col, sim, t1[col], t2[col])
        
    # print(t1, t2, var_attr)
    return math.pow(dis, 0.5)
    # return max_dis

def get_local_patterns(F, Fv, V, agg_col, model_type, t, conn, cur, pat_table_name, res_table_name):
    local_patterns = []
    local_patterns_dict = {}
    
    tF = get_F_value(F, t)
        
    if model_type is not None:
        mt_predicate = " AND model='{}'".format(model_type)
    else:
        mt_predicate = ''
    if Fv is not None:
        local_pattern_query = '''SELECT * FROM {} WHERE array_to_string(fixed, ', ')='{}' AND 
            REPLACE(array_to_string(fixed_value, ', '), '"', '') LIKE '%{}%' AND 
            array_to_string(variable, ', ') = '{}' AND
            agg='{}'{};'''.format(
                pat_table_name + '_local'+TEST_ID, str(F).replace("\'", '').replace('[', '').replace(']', ''), 
                str(Fv).replace("\'", '').replace('[', '').replace(']', ''),
                str(V).replace("\'", '').replace('[', '').replace(']', ''), 
                agg_col, mt_predicate
            )
        # print(406, Fv, local_pattern_query)
    else:
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
                pat_table_name + '_local'+TEST_ID, str(F).replace("\'", '').replace('[', '').replace(']', ''), 
                '%'.join(list(map(str, tF))),
                str(V).replace("\'", '').replace('[', '').replace(']', ''), 
                agg_col, mt_predicate
            )
        # print(local_pattern_query)
        
    cur.execute(local_pattern_query)
    local_patterns = cur.fetchall()
        
    return local_patterns




def get_tuples_by_F(local_pattern, f_value, cur, table_name, cat_sim):
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

    F = str(local_pattern[0]).replace("\'", '')[1:-1]
    V = str(local_pattern[2]).replace("\'", '')[1:-1]
    F_list = F.split(', ')
    V_list = V.split(', ')
    where_clause = ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]), zip(local_pattern[0], map(tuple_column_to_str_in_where_clause_2, zip(F_list, f_value))))))
    tuples_query = '''SELECT {},{},{} as {} FROM {} WHERE {} GROUP BY {}, {};'''.format(
            F, V, local_pattern[3].replace('_', '(') + ')', local_pattern[3], table_name, where_clause, F, V
        )
    # tuples_query = "SELECT * FROM {} WHERE {};".format(table_name, where_clause)
    
    # column_name_query = "SELECT column_name FROM information_schema.columns where table_name='{}';".format(table_name)
    # print(column_name_query)
    # cur.execute(column_name_query)
    # column_name = cur.fetchall()
    column_name = F_list + V_list + [local_pattern[3]]
    cur.execute(tuples_query)
    # print(tuples_query)
    tuples = []
    res = cur.fetchall()
    min_agg = 1e10
    max_agg = -1e10
    for row in res:
        min_agg = min(min_agg, row[-1])
        max_agg = max(max_agg, row[-1])
        tuples.append(dict(zip(map(lambda x: x, column_name), row)))
        # row_data = {}
        # cnt = 0 
        # for f_attr in F_list:    
        #     if is_float(row[cnt]):
        #         row_data[f_attr] = float(row[cnt])
        #     elif is_integer(row[cnt]):
        #         row_data[f_attr] = float(int(row[cnt]))
        #     else:
        #         row_data[f_attr] = t[cnt]
        #     cnt += 1
        # for v_attr in V_list:
        #     if is_float(row[cnt]):
        #         row_data[v_attr] = float(row[cnt])
        #     elif is_integer(row[cnt]):
        #         row_data[v_attr] = float(int(row[cnt]))
        #     else:
        #         row_data[v_attr] = row[cnt]
        #     cnt += 1
        # row_data[pat[4] + '(' + pat[3] + ')'] = row[-1]
        # tuples.append(row_data)
    return tuples, max_agg - min_agg


def get_tuples_by_F_V(lp1, lp2, f_value, v_value, conn, cur, table_name, cat_sim):
    def tuple_column_to_str_in_where_clause_2(col_value):
        # print(col_value, cat_sim.is_categorical(col_value[0]))
        if cat_sim.is_categorical(col_value[0]) or col_value[0] == 'year':
            # return "like '%" + str(col_value[1]) + "%'"
            return "= '" + str(col_value[1]) + "'"
        else:
            if is_float(col_value[1]):
                return '=' + str(col_value[1])
            else:
                # return "like '%" + str(col_value[1]) + "%'"
                return "= '" + str(col_value[1]) + "'"
    def tuple_column_to_str_in_where_clause_3(col_value):
        # print(col_value, cat_sim.is_categorical(col_value[0]))
        if cat_sim.is_categorical(col_value[0]) or col_value[0] == 'year':
            # return "like '%" + str(col_value[1]) + "%'"
            return "= '" + str(col_value[1]) + "'"
        else:
            if is_float(col_value[1]):
                return '>=' + str(col_value[1])
            else:
                # return "like '%" + str(col_value[1]) + "%'"
                return "= '" + str(col_value[1]) + "'"

    def tuple_column_to_str_in_where_clause_4(col_value):
        # print(col_value, cat_sim.is_categorical(col_value[0]))
        if cat_sim.is_categorical(col_value[0]) or col_value[0] == 'year':
            # return "like '%" + str(col_value[1]) + "%'"
            return "= '" + str(col_value[1]) + "'"
        else:
            if is_float(col_value[1]):
                return '<=' + str(col_value[1])
            else:
                # return "like '%" + str(col_value[1]) + "%'"
                return "= '" + str(col_value[1]) + "'"

    F1 = str(lp1[0]).replace("\'", '')[1:-1]
    V1 = str(lp1[2]).replace("\'", '')[1:-1]
    F1_list = F1.split(', ')
    V1_list = V1.split(', ')

    F2 = str(lp2[0]).replace("\'", '')[1:-1]
    V2 = str(lp2[2]).replace("\'", '')[1:-1]
    F2_list = F2.split(', ')
    V2_list = V2.split(', ')
    G_list = sorted(F2_list + V2_list)
    G_key = str(G_list).replace("\'", '')[1:-1]
    f_value_key = str(f_value).replace("\'", '')[1:-1]

    global MATERIALIZED_DICT
    global MATERIALIZED_CNT
    if G_key not in MATERIALIZED_DICT:
        MATERIALIZED_DICT[G_key] = dict()
        
    if f_value_key not in MATERIALIZED_DICT[G_key]:
        MATERIALIZED_DICT[G_key][f_value_key] = MATERIALIZED_CNT
        dv_query = '''DROP VIEW IF EXISTS MV_{};'''.format(str(MATERIALIZED_CNT))
        cur.execute(dv_query)
        if lp2[3] == 'count':
            agg_fun = 'count(*)'
        else:
            agg_fun = lp2[3].replace('_', '(') + ')'

        cmv_query = '''
            CREATE VIEW MV_{} AS SELECT {}, {} as {} FROM {} WHERE {} GROUP BY {};
        '''.format(
            str(MATERIALIZED_CNT), G_key, agg_fun, lp2[3], table_name,
            ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]), 
                zip(lp1[0], map(tuple_column_to_str_in_where_clause_2, zip(F1_list, f_value)))))),
            G_key
        )
        cur.execute(cmv_query)
        conn.commit()
        MATERIALIZED_CNT += 1


    where_clause = ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]), zip(lp1[0], map(tuple_column_to_str_in_where_clause_2, zip(F1_list, f_value))))))
    if v_value is not None:
        where_clause += ' AND ' 
        v_range_l = map(lambda x: v_value[0][x] + v_value[1][x][0], range(len(v_value[0])))
        v_range_r = map(lambda x: v_value[0][x] + v_value[1][x][1], range(len(v_value[0])))
        where_clause += ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]), zip(lp1[2], 
            map(tuple_column_to_str_in_where_clause_3, zip(V1_list, v_range_l))))))
        where_clause += ' AND ' 
        where_clause += ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]), zip(lp1[2], 
            map(tuple_column_to_str_in_where_clause_4, zip(V1_list, v_range_r))))))
    tuples_query = '''SELECT {},{},{} FROM MV_{} WHERE {};'''.format(
            F2, V2, lp2[3], str(MATERIALIZED_DICT[G_key][f_value_key]), where_clause
        )
    column_name = F2_list + V2_list + [lp2[3]]
    cur.execute(tuples_query)
    # print(tuples_query)
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
        
    return tuples, (min_agg, max_agg), tuples_dict


def find_global_patterns_exact_match(global_patterns_dict, F_prime_set, V_set, agg_col, reg_type):
    gp_list = []
    for v_key in global_patterns_dict:
        F_key_set = set(f_key[1:-1].replace("'", '').split(', '))
        # print(545, f_key)
        # F_set = set(f_key)
        # print(579, f_key, F_set, t_set)
        if F_key_set != F_prime_set:
            continue
        
        for v_key in global_patterns_dict[f_key]:
            V_key_set = set(v_key[1:-1].replace("'", '').split(', '))
            if V_key_set != V_set:
                continue
            for pat in global_patterns_dict[f_key][v_key]:
                if pat[2] != agg_col or pat[3] != reg_type:
                    continue
                gp_list.append(pat)
    return gp_list

def find_patterns_refinement(global_patterns_dict, F_prime_set, V_set, agg_col, reg_type):
    # pattern refinement can have different model types
    # e.g., JH’s #pub increases linearly, but JH’s #pub on VLDB remains a constant
    gp_list = []
    v_key = str(sorted(list(V_set)))
    for f_key in global_patterns_dict[0][v_key]:
        if f_key.find('arrest') != -1 or f_key.find('domestic') != -1:
            continue
        F_key_set = set(f_key[1:-1].replace("'", '').split(', '))
        # print(657, F_key_set)
        if F_prime_set.issubset(F_key_set):
            for pat in global_patterns_dict[0][v_key][f_key]:
                if pat[2] == agg_col:
                    pat_key = f_key + '|,|' + v_key + '|,|' + pat[2] + '|,|' + pat[3]
                    
                    gp_list.append(pat)
                    global VISITED_DICT
                    if pat_key not in VISITED_DICT:
                        VISITED_DICT[pat_key] = True
    # f_key = str(sorted(list(F_prime_set)))
    # for v_key in global_patterns_dict[1][f_key]:
    #     V_key_set = set(v_key[1:-1].replace("'", '').split(', '))
    #     if V_set.issubset(V_key_set):
    #         for pat in global_patterns_dict[1][f_key][v_key]:
    #             if pat[2] == agg_col:
    #                 pat_key = f_key + '|,|' + v_key + '|,|' + pat[2] + '|,|' + pat[3]
                    
    #                 gp_list.append(pat)
    #                 # global VISITED_DICT
    #                 if pat_key not in VISITED_DICT:
    #                     VISITED_DICT[pat_key] = True
    # print(679,gp_list)
    return gp_list


def score_of_explanation(t1, t2, cat_sim, num_dis_norm, dir, denominator = 1, lp1 = None, lp2 = None):
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
        t_dis_old = t_dis
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
        return [100 * score / float(denominator), t_dis, 0, deviation, float(denominator), t_dis_old]

def compare_tuple(t1, t2):
    flag1 = True
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
    
    if flag1 and flag2:
        return -1
    elif flag1:
        return -1
    else:
        return 0

def DrillDown(global_patterns_dict, local_pattern, F_set, U_set, V_set, t_prime_coarser, t_coarser, t_prime, target_tuple,
        conn, cur, pat_table_name, res_table_name, cat_sim, num_dis_norm, 
        epsilon, dir, query_result, norm_lb, dist_lb, tkheap):
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
        if dir == 1:
            dev_ub = abs(gp2[7])
        else:
            dev_ub = abs(gp2[6])
        # dev_ub = agg_maxmin[0][0] - agg_maxmin[0][1]
        k_score = tkheap.MinValue()
        # print(888, dev_ub, k_score, 100 * float(dev_ub) / (dist_lb * float(norm_lb)))
        if tkheap.HeapSize() == TOP_K and 100 * float(dev_ub) / (dist_lb * float(norm_lb)) <= k_score:
            print(890, dev_ub, dist_lb, norm_lb, gp2[0], gp2[1])
            continue
    
        # lp2_list = get_local_patterns(gp2[0], None, gp2[1], gp2[2], None, t_prime, conn, cur, pat_table_name, res_table_name)
        lp2_list = get_local_patterns(gp2[0], None, gp2[1], gp2[2], gp2[3], t_prime, conn, cur, pat_table_name, res_table_name)
        
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
        lp3_list = get_local_patterns(lp2[0], f_value, lp2[2], lp2[3], lp2[4], t_prime, conn, cur, pat_table_name, res_table_name)
        # tuples_same_F, agg_range = get_tuples_by_F(local_pattern, lp2, f_value, 
        #     conn, cur, res_table_name, cat_sim)
        # tuples_same_F, agg_range, tuples_same_F_dict = get_tuples_by_F_V(local_pattern, lp2, f_value, None,
        #     conn, cur, res_table_name, cat_sim)
        # print(725, local_pattern[0], local_pattern[1], local_pattern[2], lp2[0], lp2[1], lp2[2])
        # print(725, len(tuples_same_F), len(lp3_list))
        # print(tuples_same_F_dict.keys())
        for lp3 in lp3_list:
            if dir == 1:
                dev_ub = abs(lp3[9])
            else:
                dev_ub = abs(lp3[8])
            # dev_ub = agg_maxmin[0][0] - agg_maxmin[0][1]
            k_score = tkheap.MinValue()
            # print(919, dev_ub, k_score, 100 * float(dev_ub) / (dist_lb * float(norm_lb)))
            if tkheap.HeapSize() == TOP_K and 100 * float(dev_ub) / (dist_lb * float(norm_lb)) <= k_score:
                print(921, dev_ub, dist_lb, norm_lb, lp3[0], lp3[1])
                continue

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
                        reslist.append([s[0], s[1:], dict(row), local_pattern, lp3, 1])  
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
    cat_sim, num_dis_norm, epsilon, agg_col, conn, cur, pat_table_name, res_table_name):

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
        print('No. ' + str(j))
        dir = uq['dir']
        topK_heap = TopkHeap(TOP_K)
        marked = {}

        t = uq['target_tuple']
        print(505, t)
        uq['global_patterns'] = find_patterns_relevant(
                global_patterns_dict, uq['target_tuple'], conn, cur, res_table_name, pat_table_name, cat_sim)
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
        
        # global VISITED_DICT
        VISITED_DICT = dict()
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
            # global VISITED_DICT
            if pat_key in VISITED_DICT:
                continue
            # global VISITED_DICT
            VISITED_DICT[pat_key] = True

            
            tF = get_F_value(uq['global_patterns'][i][0], t)
            local_pattern_query_fixed = '''SELECT * FROM {} 
                    WHERE array_to_string(fixed, ', ')='{}' AND 
                    REPLACE(array_to_string(fixed_value, ', '), '.0', '')='{}' AND
                    array_to_string(variable, ', ')='{}' AND 
                    agg='{}' AND model='{}'
                ORDER BY theta;
            '''.format(
                pat_table_name + '_local' + TEST_ID, 
                str(uq['global_patterns'][i][0]).replace("\'", '').replace('[', '').replace(']', ''),
                str(tF)[1:-1].replace("\'", ''),
                str(uq['global_patterns'][i][1]).replace("\'", '').replace('[', '').replace(']', ''), 
                uq['global_patterns'][i][2], uq['global_patterns'][i][3] 
            )
            cur.execute(local_pattern_query_fixed)
            res_fixed = cur.fetchall()
            # print(local_pattern_query_fixed)
            # print(1008, res_fixed)
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
                    mark_key = str(list(map(lambda y: y[1], sorted(t_t.items(), key=lambda x: x[0]))))
                    if mark_key not in marked:
                        marked[mark_key] = True
                        topK_heap.Push([s[0], s[1:], 
                            # list(map(lambda y: str(y[1]), sorted(t_t.items(), key=lambda x: x[0]))), 
                            t_t.items(),
                            local_patterns[i], None, 0])
                    top_k_lists[i][-1].append([s[0], s[1:], dict(t_t), local_patterns[i], None, 0])
                    if s[1] < dist_lb:
                        dist_lb = s[1]
                    # if s[-1] < dist_lb:
                    #     dist_lb = s[-1]
                    if abs(s[3]) > dev_ub:
                        dev_ub = abs(s[3])
            if dist_lb < 1e-10:
                dist_lb = 0.01

            
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
                dir, uq['query_result'], 
                norm_lb, dist_lb, topK_heap)

            for tk in top_k_lists[i][-1]:
                mark_key = str(list(map(lambda y: y[1], sorted(tk[2].items(), key=lambda x: x[0]))))
                if mark_key not in marked:
                    marked[mark_key] = True
                    topK_heap.Push([tk[0], tk[1], 
                        # list(map(lambda y: str(y[1]), sorted(tk[2].items(), key=lambda x: x[0]))), 
                        tk[2].items(),
                        tk[3], tk[4], tk[5]])
            
            end = time.time()
            
            score_computing_time += end - start
                
                
        score_computing_end = time.time()             
        score_computing_time_cur_uq = score_computing_end - score_computing_start
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
        score_computing_time_list.append([t, score_computing_time_cur_uq])
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

def load_query_result(t, cur, query_result_table, agg_col):
    agg_arr = agg_col.split('_')
    agg = agg_arr[0]
    a = agg_arr[1]
    group_arr = list(t.keys())
    group_arr.remove('lambda')
    group_arr.remove('direction')
    group_arr.remove('agg_col')
    group = ', '.join(group_arr)
    aggregate_query = 'SELECT {}, {}({}) as {} FROM {} GROUP BY {};'.format(
            group, agg, a, agg_col, query_result_table, group
        )
    cur.execute(aggregate_query)
    res = cur.fetchall()
    qr = list(map(lambda y: dict(zip(map(lambda x: x, column_name), y)), res))
    
    return qr

def find_patterns_relevant_old(global_patterns_dict, t, cur, table_name):
    
    g_pat_list = []
    l_pat_list = []
    res_list = []
    t_set = set(t.keys())
    # print(global_patterns_dict.keys())
    for v_key in global_patterns_dict[0]:
        # print(pat, pat[0])
        V_set = set(v_key[1:-1].replace("'", '').split(', '))
        # print(545, f_key)
        # F_set = set(f_key)
        # print(579, v_key, V_set, t_set)
        if not V_set.issubset(t_set):
            continue
        
        for f_key in global_patterns_dict[0][v_key]:
            for pat in global_patterns_dict[0][v_key][f_key]:
                # F_set = set(f_key.split(','))
                F_set = set(f_key[1:-1].replace("'", '').split(', '))
                # print(587, f_key, F_set, V_set, t_set)
                if not F_set.issubset(t_set):
                    continue
                # print(pat)
                if pat[2] not in t:
                    continue

                tF = get_F_value(pat[0], t)
                local_pattern_query_fixed = '''SELECT * FROM {} 
                        WHERE array_to_string(fixed, ', ')='{}' AND 
                        array_to_string(fixed_value, ', ')='{}' AND
                        array_to_string(variable, ', ')='{}' AND 
                        agg='{}' AND model='{}'
                    ORDER BY theta;
                '''.format(
                    table_name + '_local'+TEST_ID, str(pat[0]).replace("\'", '').replace('[', '').replace(']', ''),
                    str(tF)[1:-1].replace("\'", ''),
                    str(pat[1]).replace("\'", '').replace('[', '').replace(']', ''), 
                    pat[2], pat[3] 
                )
                cur.execute(local_pattern_query_fixed)
                print(1195, local_pattern_query_fixed)
                res_fixed = cur.fetchall()
                # print(res_fixed)
                pat_list = res_fixed
                if len(pat_list) > 0:
                    g_pat_list.append(pat)
                    l_pat_list.append(pat_list[0])
                    res_list.append([pat, pat_list[0], predict(pat_list[0], t)])
    # g_pat_list = sorted(g_pat_list, key=lambda x: len(x[0])+len(x[1]))
    # l_pat_list = sorted(l_pat_list, key=lambda x: len(x[0])+len(x[2]))
    res_list = sorted(res_list, key = lambda x: (len(x[0][0]) + len(x[0][1]), x[2]))
    g_pat_list = list(map(lambda x: x[0], res_list))
    l_pat_list = list(map(lambda x: x[1], res_list))
    print(1175, len(g_pat_list))
    return g_pat_list, l_pat_list

    # g_pat_list = []
    # l_pat_list = []
    # res_list = []
    # t_set = set(t.keys())
    # t_list = list(t.keys())
    # size_t = len(t_list) - 1
    # for i in range(1, 1 << size_t):
    #     for j in range(1, 1 << size_t):
    #         if i & j == 0:
    #             F_list = []
    #             V_list = []
    #             for k in range(0, size_t):
    #                 if ((1 << k) & i) > 0:
    #                     F_list.append(t_list[k])
    #                 if ((1 << k) & j) > 0:
    #                     V_list.append(t_list[k])
    #             v_key = str(sorted(V_list))
    #             f_key = str(sorted(F_list))
    #             if v_key in global_patterns_dict[0] and f_key in global_patterns_dict[0][v_key]:
    #                 for pat in global_patterns_dict[0][v_key][f_key]:
    #                     print(pat)
    #                     if pat[2] not in t:
    #                         continue

    #                     tF = get_F_value(pat[0], t)
    #                     local_pattern_query_fixed = '''SELECT * FROM {} 
    #                             WHERE array_to_string(fixed, ', ')='{}' AND 
    #                             array_to_string(fixed_value, ', ')='{}' AND
    #                             array_to_string(variable, ', ')='{}' AND 
    #                             agg='{}' AND model='{}'
    #                         ORDER BY theta;
    #                     '''.format(
    #                         table_name + '_local', str(pat[0]).replace("\'", '').replace('[', '').replace(']', ''),
    #                         str(tF)[1:-1].replace("\'", ''),
    #                         str(pat[1]).replace("\'", '').replace('[', '').replace(']', ''), 
    #                         pat[2], pat[3] 
    #                     )
    #                     cur.execute(local_pattern_query_fixed)
    #                     print(local_pattern_query_fixed)
    #                     res_fixed = cur.fetchall()
    #                     print(res_fixed)
    #                     pat_list = res_fixed
    #                     if len(pat_list) > 0:
    #                         g_pat_list.append(pat)
    #                         l_pat_list.append(pat_list[0])
    #                         res_list.append([pat, pat_list[0], predict(pat_list[0], t)])
    # # g_pat_list = sorted(g_pat_list, key=lambda x: len(x[0])+len(x[1]))
    # # l_pat_list = sorted(l_pat_list, key=lambda x: len(x[0])+len(x[2]))
    # res_list = sorted(res_list, key = lambda x: (len(x[0][0]) + len(x[0][1]), x[2]))
    # g_pat_list = list(map(lambda x: x[0], res_list))
    # l_pat_list = list(map(lambda x: x[1], res_list))
    # print(1175, len(g_pat_list))
    # return g_pat_list, l_pat_list

def get_tuples_by_gp_uq(gp, f_value, v_value, conn, cur, table_name, cat_sim):
    def tuple_column_to_str_in_where_clause_2(col_value):
        # print(col_value, cat_sim.is_categorical(col_value[0]))
        if cat_sim.is_categorical(col_value[0]) or col_value[0] == 'year':
            # return "like '%" + str(col_value[1]) + "%'"
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
    v_value_key = str(v_value).replace("\'", '')[1:-1]

    global MATERIALIZED_DICT
    global MATERIALIZED_CNT
    if G_key not in MATERIALIZED_DICT:
        MATERIALIZED_DICT[G_key] = dict()
        
    if f_value_key not in MATERIALIZED_DICT[G_key]:
        MATERIALIZED_DICT[G_key][f_value_key] = MATERIALIZED_CNT
        dv_query = '''DROP VIEW IF EXISTS MV_{};'''.format(str(MATERIALIZED_CNT))
        cur.execute(dv_query)
        # print(504, MATERIALIZED_CNT)
        if gp[2] == 'count':
            agg_fun = 'count(*)'
        else:
            agg_fun = gp[2].replace('_', '(') + ')'
        cmv_query = '''
            CREATE VIEW MV_{} AS SELECT {}, {} as {} FROM {} WHERE {} GROUP BY {};
        '''.format(
            str(MATERIALIZED_CNT), G_key, agg_fun, gp[2], table_name,
            ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]), 
                zip(gp[0], map(tuple_column_to_str_in_where_clause_2, zip(F1_list, f_value)))))),
            G_key
        )
        # print(585, cmv_query)
        cur.execute(cmv_query)
        conn.commit()
        MATERIALIZED_CNT += 1


    where_clause = ' AND '.join(
        list(map(lambda x: "{} {}".format(x[0], x[1]), zip(gp[0], map(
            tuple_column_to_str_in_where_clause_2, zip(F1_list, f_value)))))) + ' AND ' + \
        ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]), 
                zip(gp[1], map(tuple_column_to_str_in_where_clause_2, zip(V1_list, v_value))))))

    tuples_query = '''SELECT {} FROM MV_{} WHERE {};'''.format(
            gp[2], str(MATERIALIZED_DICT[G_key][f_value_key]), where_clause
        )
    # print(604, tuples_query)
    cur.execute(tuples_query)
    res = cur.fetchall()
    if len(res) == 0:
        return []
    return res[0]



def find_patterns_relevant(global_patterns_dict, t, conn, cur, query_table_name, pattern_table_name, cat_sim):
    
    g_pat_list = []
    l_pat_list = []
    res_list = []
    t_set = set(t.keys())
    # print(global_patterns_dict.keys())
    for v_key in global_patterns_dict[0]:
        # print(pat, pat[0])
        V_set = set(v_key[1:-1].replace("'", '').split(', '))
        # print(545, f_key)
        # F_set = set(f_key)
        # print(579, v_key, V_set, t_set)
        if not V_set.issubset(t_set):
            continue
        
        for f_key in global_patterns_dict[0][v_key]:
            for pat in global_patterns_dict[0][v_key][f_key]:
                # F_set = set(f_key.split(','))
                F_set = set(f_key[1:-1].replace("'", '').split(', '))
                # print(1382, F_set, V_set, t_set)
                # if len(list(F_set)) < 2:
                    # print(1337, f_key, F_set, V_set, t_set)
                if not F_set.issubset(t_set):
                    continue
                # print(pat)
                if pat[2] not in t:
                    continue

                agg_value = get_tuples_by_gp_uq(pat, get_F_value(pat[0], t), get_V_value(pat[1], t), 
                    conn, cur, query_table_name, cat_sim)
                if len(agg_value) > 0:
                    res_list.append([pat, agg_value[0]])

                
    res_list = sorted(res_list, key = lambda x: (len(x[0][0]) + len(x[0][1]), x[1]))
    g_pat_list = list(map(lambda x: x[0], res_list))
    return g_pat_list

def load_user_question(global_patterns, global_patterns_dict, uq_path=DEFAULT_QUESTION_PATH, schema=None, cur=None, pattern_table='', query_result_table='', pf=None):
    '''
        load user questions
    '''
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
            for k, v in enumerate(headers):
                # print(k, v)
                if schema is None or v not in schema:
                    if v != 'direction':
                        if is_float(row[v]):
                            row_data[v] = float(row[v])
                        elif is_integer(row[v]):
                            row_data[v] = float(long(row[v]))
                        else:
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
            # uq[-1]['global_patterns'], uq[-1]['local_patterns'] = find_patterns_relevant(
            #     global_patterns_dict, uq[-1]['target_tuple'], cur, pattern_table
            # )
            # uq[-1]['global_patterns'] = find_patterns_relevant(
            #     global_patterns_dict, uq[-1]['target_tuple'], cur, pattern_table
            # )
            
            # print(739, j, g_pat_cnt_by_theta[j])
            # print(list(map(lambda x: str(x[0]) + ' ' + str(x[1]) + '     ', g_pat_list_by_theta[j][0])))
            # print(list(map(lambda x: str(x[0]) + ' ' + str(x[1]) + '     ' if len(x) > 1 else '   empty   ', g_pat_list_by_theta[j][1])))
    
            # uq[-1]['query_result'] = load_query_result(uq[-1]['target_tuple'], cur, query_result_table, agg_col)
            uq[-1]['query_result'] = []
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

        
def main(argv=[]):
    query_result_table = DEFAULT_QUERY_RESULT_TABLE
    pattern_table = DEFAULT_PATTERN_TABLE
    user_question_file = DEFAULT_QUESTION_PATH
    outputfile = ''
    epsilon = DEFAULT_EPSILON
    aggregate_column = DEFAULT_AGGREGATE_COLUMN
    try:
        # conn = psycopg2.connect("host=216.47.152.61 port=5432 dbname=postgres user=antiprov password=test")
        # conn = psycopg2.connect("host=localhost port=5432 dbname=antiprov user=zjmiao password=keertijeff")
        conn = psycopg2.connect("host=localhost port=5436 dbname=antiprov user=antiprov")
        cur = conn.cursor()
    except psycopg2.OperationalError:
        print('数据库连接失败！')

    # config=['localhost','5432','antiprov','zjmiao','keertijeff']
    config=['localhost','5436','antiprov','antiprov','']
    try:
        engine = sa.create_engine(
                'postgresql://' + config[3] + ':' + config[4] + '@' + config[0]+':'+config[1]+'/'+config[2],
                # 'postgresql://' + config[3] + ':@' + config[0]+':'+config[1]+'/'+config[2],
                echo=True)
    except Exception as ex:
        print(ex)
        sys.exit(1)

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

    load_start = time.time()
    global_patterns, schema, global_patterns_dict = load_patterns(cur, pattern_table, query_result_table)
    pf = PatternFinder(engine.connect(), query_result_table, fit=True, theta_c=0.5, theta_l=0.25, lamb=DEFAULT_LAMBDA, dist_thre=0.9,  supp_l=10,supp_g=1)
    Q, global_patterns, global_patterns_dict = load_user_question(global_patterns, global_patterns_dict, user_question_file, schema, cur, pattern_table, query_result_table, pf)
        

    # category_similarity = CategorySimilarityMatrix(EXAMPLE_SIMILARITY_MATRIX_PATH, schema)
    category_similarity = CategorySimilarityNaive(cur=cur, table_name=query_result_table, embedding_table_list=[('community_area', 'community_area_loc')])
    # category_similarity = CategoryNetworkEmbedding(EXAMPLE_NETWORK_EMBEDDING_PATH, data['df'])
    #num_dis_norm = normalize_numerical_distance(data['df'])
    num_dis_norm = normalize_numerical_distance(cur=cur, table_name=query_result_table)
    load_end = time.time()
    load_time = load_end-load_start

    
    expl_start = time.time()
    #regression_package = 'scikit-learn'
    regression_package = 'statsmodels'
    explanations_list, local_patterns_list, score_computing_time_list = find_explanation_regression_based(
        Q, global_patterns, global_patterns_dict, category_similarity, 
        num_dis_norm, epsilon, 
        aggregate_column, 
        conn, cur, 
        pattern_table, query_result_table)
    expl_end = time.time()
    query_time = expl_end-expl_start
    print('User question and global pattern loading time: ' + str(load_time) + 'seconds')
    print('Total querying time: ' + str(query_time) + 'seconds')

    cur.execute('create table IF NOT EXISTS explanation_time(id serial primary key, pattern_id varchar(50), load_time float, query_time float);')
    cur.execute("INSERT INTO explanation_time(pattern_id, load_time, query_time) VALUES('{}', {}, {})".format(
        pattern_table, str(load_time), str(query_time)))
    conn.commit()

    ofile = sys.stdout
    if outputfile != '':
        ofile = open(outputfile, 'w')

    # for i, explanations in enumerate(explanations_list):
    #     ofile.write('User question ' + str(i+1) + ':\n')
    #     for j, e in enumerate(explanations):
    #         print_str = ''
    #         e_tuple = data['df'].loc[data['df']['index'] == e[2]]
    #         e_tuple_str = ','.join(e_tuple.to_string(header=False,index=False,index_names=False).split('  ')[1:])
    #         ofile.write('Top ' + str(j+1) + ' explanation:\n')
    #         ofile.write('Constraint ' + str(e[1]+1) + ': [' + ','.join(constraints[e[1]][0]) + ']' + '[' + ','.join(constraints[e[1]][1]) + ']')
    #         ofile.write('\n')
    #         ofile.write('Score: ' + str(e[0]))
    #         ofile.write('\n')
    #         ofile.write('(' + e_tuple_str + ')')
    #         ofile.write('\n')
    #     ofile.write('------------------------\n')

    for i, top_k_list in enumerate(explanations_list):
        ofile.write('User question {} in direction {}: {}\n'.format(
            str(i+1), 'high' if Q[i]['dir'] > 0 else 'low', str(Q[i]['target_tuple']))
        )

        print(1217, len(top_k_list))
        for j, e in enumerate(top_k_list):
            ofile.write('------------------------\n')
            print_str = ''
            # e_tuple_str = ','.join(e_tuple.to_string(header=False,index=False,index_names=False).split('  ')[1:])
            print(e)
            if isinstance(e, dict):
                continue
            e_tuple_str = ','.join(map(str, e[2]))
            ofile.write('Top ' + str(j+1) + ' explanation:\n')
            # ofile.write('Constraint ' + str(e[1]+1) + ': [' + ','.join(global_patterns[e[1]][0]) + ']' + '[' + ','.join(global_patterns[e[1]][1]) + ']')
            # print(827, e[1], local_patterns_list[i][e[1]][0])
            # print(828, local_patterns_list[i][e[1]][1])
            # print(829, local_patterns_list[i][e[1]][2])
            # ofile.write('Constraint ' + str(e[1]+1) + ': [' + ','.join(local_patterns_list[i][e[1]][0]) + ']' + 
            #     '[' + ','.join(list(map(str, local_patterns_list[i][e[1]][1]))) + ']' +
            #     '[' + ','.join(list(map(str, local_patterns_list[i][e[1]][2]))) + ']')
            
            if e[5] == 1:
                ofile.write('From local pattern' + ': [' + ','.join(e[3][0]) + ']' + 
                    '[' + ','.join(list(map(str, e[3][1]))) + ']' +
                    '[' + ','.join(list(map(str, e[3][2]))) + ']')
                ofile.write(' drill down to' + ': [' + ','.join(e[4][0]) + ']' + 
                    '[' + ','.join(list(map(str, e[4][1]))) + ']' +
                    '[' + ','.join(list(map(str, e[4][2]))) + ']')
            else:
                ofile.write('Directly from local pattern ' + ': [' + ','.join(e[3][0]) + ']' + 
                    '[' + ','.join(list(map(str, e[3][1]))) + ']' +
                    '[' + ','.join(list(map(str, e[3][2]))) + ']')
            ofile.write('\n')
            ofile.write('Score: ' + str(e[0]))
            ofile.write('\n')
            ofile.write('Distance: ' + str(e[1][0]))
            ofile.write('\n')
            # ofile.write('Simialriry: ' + str(e[1][1]))
            # ofile.write('\n')
            ofile.write('Outlierness: ' + str(e[1][2]))
            ofile.write('\n')
            ofile.write('Denominator: ' + str(e[1][3]))
            ofile.write('\n')
            ofile.write('(' + e_tuple_str + ')')
            ofile.write('\n')
        # else:
            #     ofile.write('------------------------\n')
            #     ofile.write('Explanation:\n')
            #     ofile.write(str(list_by_pat) + '\n')
        ofile.write('------------------------\n\n')
    ofile.close()

    for g_key in MATERIALIZED_DICT:
        for fv_key in MATERIALIZED_DICT[g_key]:
            dv_query = '''DROP VIEW IF EXISTS MV_{};'''.format(str(MATERIALIZED_DICT[g_key][fv_key]))
            cur.execute(dv_query)
            conn.commit()

    att_size_list = []
    sct_list = []
    time_o_file = open('../time_record_rev/crime_perf/crime_expl_time_prune_sort_{}{}_new.csv'.format(str(TOP_K), TEST_ID), 'w')
    for sct in score_computing_time_list:
        att_size_list.append(len(list(sct[0].keys())) - 2)
        sct_list.append(sct[1])
        time_o_file.write(str(att_size_list[-1]) + ',' + str(sct[1]) + '\n')
    time_o_file.close()
    # sct_df = pandas.DataFrame({'#attr': att_size_list, 'time':sct_list})
    # sct_df.boxplot(by='#attr')
    # plt.savefig('time1.png')


if __name__ == "__main__":
    main(sys.argv[1:])


