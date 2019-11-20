#!/usr/bin/python
# -*- coding:utf-8 -*- 

from heapq import *
import re
import logging
from capexplain.similarity.category_similarity_matrix import *
from capexplain.similarity.category_similarity_naive import *
from capexplain.similarity.category_network_embedding import *
from capexplain.utils import *
from capexplain.pattern_model.LocalRegressionPattern import *
from capexplain.cl.cfgoption import DictLike
from capexplain.explanation_model.explanation_model import *

# from build.lib.capexplain.database.dbaccess import DBConnection
# setup logging
# log = logging.getLogger(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s line %(lineno)d: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# ********************************************************************************
# Configuration for Explanation generation
class ExplConfig(DictLike):
    # DEFAULT_RESULT_TABLE = 'pub_large_no_domain'
    DEFAULT_RESULT_TABLE = 'crime_clean_100000_2'
    # DEFAULT_PATTERN_TABLE = 'dev.pub'
    DEFAULT_PATTERN_TABLE = 'dev.crime_clean_100000'
    # DEFAULT_RESULT_TABLE = 'crime_exp'
    # DEFAULT_PATTERN_TABLE = 'dev.crime_exp'
    DEFAULT_QUESTION_PATH = './input/user_question.csv'

    DEFAULT_NETWORK_EMBEDDING_PATH = './input/NETWORK_EMBEDDING'
    DEFAULT_SIMILARITY_MATRIX_PATH = './input/SIMILARITY_DEFINITION'
    DEFAULT_AGGREGATE_COLUMN = '*'
    DEFAULT_THETA = 0.1
    DEFAULT_LAMBDA = 0.1
    DEFAULT_TOP_K = 10

    # global MATERIALIZED_CNT
    MATERIALIZED_CNT = 0
    # global MATERIALIZED_DICT
    MATERIALIZED_DICT = dict()
    # global VISITED_DICT
    VISITED_DICT = dict()

    REGRESSION_PACKAGES = ['scikit-learn', 'statsmodels']

    def __init__(self,
                 query_result_table=DEFAULT_RESULT_TABLE,
                 pattern_table=DEFAULT_PATTERN_TABLE,
                 user_question_file=DEFAULT_QUESTION_PATH,
                 similarity_matrix_file=None,
                 outfile='',
                 runtime_outfile=None,
                 aggregate_column=DEFAULT_AGGREGATE_COLUMN,
                 pattern_theta=DEFAULT_THETA,
                 pattern_lambda=DEFAULT_LAMBDA,
                 expl_topk=DEFAULT_TOP_K,
                 regression_package='statsmodels',
                 exp_id=None,
                 pruning=False
                 ):
        self.pattern_table = pattern_table
        self.query_result_table = query_result_table
        self.user_question_file = user_question_file
        self.similarity_matrix_file = similarity_matrix_file
        self.outfile = outfile
        self.aggregate_column = aggregate_column
        self.pattern_theta = pattern_theta
        self.pattern_lambda = pattern_lambda
        self.expl_topk = expl_topk
        self.exp_id = exp_id
        self.runtime_outfile = runtime_outfile
        self.pruning = pruning
        self.regression_package = regression_package
        self.global_patterns = None
        self.schema = None
        self.global_patterns_dict = None
        self.conn = self.cur = None

    def __str__(self):
        return self.__dict__.__str__()


from capexplain.explain.pattern_retrieval import get_local_patterns, find_patterns_relevant, \
    find_patterns_refinement, load_patterns
from capexplain.explain.tuple_retrieval import get_tuples_by_F_V


class TopkHeap(object):
    def __init__(self, k):
        self.topk = k
        self.data = []

    def Push(self, elem):
        if len(self.data) < self.topk:
            heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem.score > topk_small.score:
                heapreplace(self.data, elem)

    def MinValue(self):
        return min(list(map(lambda x: x.score, self.data)))

    def MaxValue(self):
        return max(list(map(lambda x: x.score, self.data)))

    def TopK(self):
        return [x for x in reversed([heappop(self.data) for x in range(len(self.data))])]

    def HeapSize(self):
        return len(self.data)


def predict(local_pattern, t):
    if local_pattern[4] == 'const':
        predict_y = float(local_pattern[6][1:-1].split(',')[0])
    elif local_pattern[4] == 'linear':
        if isinstance(local_pattern[7], str):
            params_str = local_pattern[7].split('\n')
            params_dict = {}
            for i in range(0, len(params_str) - 1):
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
            params_dict = local_pattern[7]

        predict_y = params_dict['Intercept']
        for v_attr in t:
            v_key = '{}[T.{}]'.format(v_attr, t[v_attr])
            if v_key in params_dict:
                predict_y += params_dict[v_key]
            else:
                if v_attr in params_dict:
                    predict_y += params_dict[v_attr] * float(t[v_attr])

    return predict_y


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
            if col not in num_dis_norm or num_dis_norm[col]['range'] is None:
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

                    dis += 0.5 * math.pow(temp, 8)
                    if temp > max_dis:
                        max_dis = temp
                cnt += 1

    return math.pow(dis, 0.5)


def score_of_explanation(t1, t2, cat_sim, num_dis_norm, dir, denominator=1, lp1=None, lp2=None):
    if lp1 is None:
        return 1.0
    else:
        # print(lp1, lp2, t1, t2)
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

        for col in t1:
            if col not in t2:
                diff += 1
        for col in t2:
            if col not in t1:
                diff += 1

        w = 5
        t_dis = math.sqrt(t_dis_raw * t_dis_raw + w * diff * diff)

        t1v = dict(zip(lp1[2], map(lambda x: x, get_V_value(lp1[2], t1))))
        predicted_agg1 = predict(lp1, t1v)
        # t2v = dict(zip(lp2[2], map(lambda x: x, get_V_value(lp2[2], t2))))

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
        if a != 'lambda' and not a.startswith('count_') and not a.startswith('sum_'):
            if a not in t2:
                flag1 = False
            elif t1[a] != t2[a]:
                return 0
    flag2 = True
    for a in t2:
        # if (a != 'lambda' and a.find('_') == -1):
        if a != 'lambda' and not a.startswith('count_') and not a.startswith('sum_'):
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


def DrillDown(global_patterns_dict, local_pattern, F_set, U_set, V_set, t_prime_coarser, t_coarser, t_prime,
              target_tuple,
              ecf, cat_sim, num_dis_norm,
              dir, query_result, norm_lb, dist_lb, tkheap):
    reslist = []
    agg_col = local_pattern[3]

    gp2_list = find_patterns_refinement(global_patterns_dict, F_set, V_set, local_pattern[3], local_pattern[4])
    if len(gp2_list) == 0:
        return []
    for gp2 in gp2_list:
        # if str(gp2[0]).find('primary_type') == -1 or str(gp2[0]).find('community_area') == -1 or str(gp2[1]).find('year') == -1:
        #     continue

        if dir == 1:
            dev_ub = abs(gp2[7])
        else:
            dev_ub = abs(gp2[6])
        k_score = tkheap.MinValue()
        if ecf.pruning and tkheap.HeapSize() == ecf.expl_topk and 100 * float(dev_ub) / (dist_lb * float(norm_lb)) <= k_score:
            # prune
            continue

        lp2_list = get_local_patterns(gp2[0], None, gp2[1], gp2[2], gp2[3], t_prime, ecf.conn, ecf.cur, 
            (ecf.pattern_table + '_local') + ('_' + str(ecf.exp_id)) if ecf.exp_id is not None else '')
        if len(lp2_list) == 0:
            continue
        lp2 = lp2_list[0]

        if len(lp2[0]) == 2 and len(lp2[2]) == 1:
            logger.debug(lp2)
        f_value = get_F_value(local_pattern[0], t_prime)

        tuples_same_F, agg_range, tuples_same_F_dict = get_tuples_by_F_V(local_pattern, lp2, f_value,
                                                                         None,
                                                                         ecf.conn, ecf.cur, ecf.query_result_table, cat_sim)

        lp3_list = get_local_patterns(lp2[0], f_value, lp2[2], lp2[3], lp2[4], t_prime, ecf.conn, ecf.cur, 
            (ecf.pattern_table + '_local') + ('_' + str(ecf.exp_id)) if ecf.exp_id is not None else '')

        for lp3 in lp3_list:

            if dir == 1:
                dev_ub = abs(lp3[9])
            else:
                dev_ub = abs(lp3[8])
            k_score = tkheap.MinValue()

            if ecf.pruning and tkheap.HeapSize() == ecf.expl_topk and 100 * float(dev_ub) / (dist_lb * float(norm_lb)) <= k_score:
                # prune
                continue
            f_key = str(lp3[1]).replace('\'', '')[1:-1]
            f_key = f_key.replace('.0', '')

            if f_key in tuples_same_F_dict:
                for idx, row in enumerate(tuples_same_F_dict[f_key]):
                    for idx2, row2 in enumerate(t_coarser):
                        if get_V_value(local_pattern[2], row2) == get_V_value(local_pattern[2], row):
                            s = score_of_explanation(row, target_tuple, cat_sim, num_dis_norm, dir,
                                                     float(row2[agg_col]), lp3, lp2)
                            break

                    # e.g. U = {Venue}, u = {ICDE}, do not need to check whether {Author, Venue} {Year} holds on (JH, ICDE)
                    # expected values are replaced with the average across year for all (JH, ICDE, year) tuples
                    # s = score_of_explanation(row, t_prime, cat_sim)
                    cmp_res = compare_tuple(row, target_tuple)
                    if cmp_res == 0:  # row is not subset of target_tuple, target_tuple is not subset of row
                        reslist.append(
                            Explanation(1, s[0], s[1], s[2], s[3], dir, dict(row), ecf.expl_topk, local_pattern,
                                        lp3))

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
                                      cat_sim, num_dis_norm, agg_col, ecf):
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
    local_patterns_list = []

    for j, uq in enumerate(user_question_list):
        dir = uq['dir']
        topK_heap = TopkHeap(ecf.expl_topk)
        marked = {}

        t = dict(uq['target_tuple'])

        uq['global_patterns'] = find_patterns_relevant(
            global_patterns_dict, uq['target_tuple'], ecf.conn, ecf.cur, ecf.query_result_table, cat_sim)

        top_k_lists = [[] for i in range(len(uq['global_patterns']))]
        local_patterns = []

        ExplConfig.VISITED_DICT = dict()
        score_computing_start = time.time()

        for i in range(0, len(uq['global_patterns'])):
            top_k_lists[i] = [4, uq['global_patterns'][i], t, []]
            local_patterns.append(None)
            F_key = str(sorted(uq['global_patterns'][i][0]))
            V_key = str(sorted(uq['global_patterns'][i][1]))
            pat_key = F_key + '|,|' + V_key + '|,|' + uq['global_patterns'][i][2] + '|,|' + uq['global_patterns'][i][3]
            if pat_key in ExplConfig.VISITED_DICT:
                continue
            ExplConfig.VISITED_DICT[pat_key] = True

            tF = get_F_value(uq['global_patterns'][i][0], t)
            local_pattern_query_fixed = '''SELECT * FROM {} 
                                WHERE array_to_string(fixed, ', ')='{}' AND 
                                REPLACE(array_to_string(fixed_value, ', '), '.0', '')='{}' AND
                                array_to_string(variable, ', ')='{}' AND 
                                agg='{}' AND model='{}'
                            ORDER BY theta;
                        '''.format(
                (ecf.pattern_table + '_local') + (('_' + str(ecf.exp_id)) if ecf.exp_id is not None else ''),
                str(uq['global_patterns'][i][0]).replace("\'", '').replace('[', '').replace(']', ''),
                str(tF)[1:-1].replace("\'", ''),
                str(uq['global_patterns'][i][1]).replace("\'", '').replace('[', '').replace(']', ''),
                uq['global_patterns'][i][2], uq['global_patterns'][i][3]
            )
            ecf.cur.execute(local_pattern_query_fixed)
            res_fixed = ecf.cur.fetchall()

            if len(res_fixed) == 0:
                continue

            local_patterns[i] = res_fixed[0]
            T_set = set(t.keys()).difference(set(['lambda', uq['global_patterns'][i][2]]))
            agg_col = local_patterns[i][3]
            start = time.time()
            t_t_list, agg_range, t_t_dict = get_tuples_by_F_V(local_patterns[i], local_patterns[i],
                                                              get_F_value(local_patterns[i][0], t),
                                                              # [get_V_value(local_patterns[i][2], t), [[-3, 3]]],
                                                              None,
                                                              ecf.conn, ecf.cur, ecf.query_result_table, cat_sim)

            dist_lb = 1e10
            dev_ub = 0
            for t_t in t_t_list:
                if compare_tuple(t_t, t) == 0:
                    s = score_of_explanation(t_t, t, cat_sim, num_dis_norm, dir, t_t[agg_col], local_patterns[i],
                                             local_patterns[i])
                    expl_temp = Explanation(0, s[0], s[1], s[2], s[3], uq['dir'],
                                                   # list(map(lambda y: y[1], sorted(t_t.items(), key=lambda x: x[0]))),
                                                   dict(t_t),
                                                   ecf.expl_topk, local_patterns[i], None)
                    expl_temp_str = expl_temp.ordered_tuple_string()
                    # if str(t_t) not in marked:
                    #     marked[str(t_t)] = True
                    if expl_temp_str not in marked:
                        marked[expl_temp_str] = True
                        topK_heap.Push(expl_temp)
                        top_k_lists[i][-1].append(expl_temp)
                        print(t_t, t, compare_tuple(t_t, t))
                    if s[-1] < dist_lb:
                        dist_lb = s[-1]
                        # use raw distance (without penalty on missing attributes) as the lower bound
                    if abs(s[2]) > dev_ub:
                        dev_ub = abs(s[2])
            if dist_lb < 1e-10:
                dist_lb = 0.01

            end = time.time()
            question_validating_time += end - start

            F_set = set(local_patterns[i][0])
            V_set = set(local_patterns[i][2])

            # F union V \subsetneq G
            t_coarser_copy = list(t_t_list)
            norm_lb = min(list(map(lambda x: x[agg_col], t_coarser_copy)))

            k_score = topK_heap.MinValue()
            # prune
            if ecf.pruning and topK_heap.HeapSize() == ecf.expl_topk and 100 * float(dev_ub) / (dist_lb * float(norm_lb)) <= k_score:
                continue
            top_k_lists[i][-1] += DrillDown(global_patterns_dict, local_patterns[i],
                                            F_set, T_set.difference(F_set.union(V_set)), V_set, t_coarser_copy,
                                            t_coarser_copy, t, t,
                                            ecf, cat_sim, num_dis_norm,
                                            dir, uq['query_result'],
                                            norm_lb, dist_lb, topK_heap)
            for tk in top_k_lists[i][-1]:
                # if str(tk.tuple_value) not in marked:
                #    marked[str(tk.tuple_value)] = True
                tk_str = tk.ordered_tuple_string()
                if tk_str not in marked:
                    marked[tk_str] = True
                    topK_heap.Push(tk)

        score_computing_end = time.time()
        score_computing_time_cur_uq = score_computing_end - score_computing_start

        answer[j] = topK_heap.TopK()
        score_computing_time_list.append([t, score_computing_time_cur_uq])

    print('Local pattern loading time: ' + str(local_pattern_loading_time) + 'seconds')
    print('Score computing time: ' + str(sum(map(lambda x: x[1], score_computing_time_list))) + 'seconds')
    return answer, local_patterns_list, score_computing_time_list


def load_user_question_from_file(global_patterns, global_patterns_dict, uq_path, schema=None, conn=None, cur=None,
                                 pattern_table='', query_result_table='', pf=None, cat_sim=None):
    '''
        load user questions
    '''

    uq = []
    with open(uq_path, 'rt') as uqfile:
        reader = csv.DictReader(uqfile, quotechar='\'')
        headers = reader.fieldnames
        for row in reader:
            row_data = {}
            raw_row_data = {}
            agg_col = None
            for k, v in enumerate(headers):
                print(k, v)
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

            if row['direction'][0] == 'h':
                dir = 1
            else:
                dir = -1
            uq.append({'target_tuple': row_data, 'dir': dir})
            uq[-1]['query_result'] = []
    return uq, global_patterns, global_patterns_dict


class ExplanationGenerator:

    def __init__(self, config: ExplConfig, user_input_config=None):
        self.config = config
        if user_input_config is not None:
            if 'pattern_table' in user_input_config:
                self.config.pattern_table = user_input_config['pattern_table']
            if 'query_result_table' in user_input_config:
                self.config.query_result_table = user_input_config['query_result_table']
            if 'user_question_file' in user_input_config:
                self.config.user_question_file = user_input_config['user_question_file']
            if 'similarity_matrix_file' in user_input_config:
                self.config.similarity_matrix_file = user_input_config['similarity_matrix_file']
            if 'outfile' in user_input_config:
                self.config.outfile = user_input_config['outfile']
            if 'exp_id' in user_input_config:
                self.config.exp_id = user_input_config['exp_id']
            if 'runtime_outfile' in user_input_config:
                self.config.runtime_outfile = user_input_config['runtime_outfile']
            if 'pruning' in user_input_config:
                self.config.pruning = True
            if 'aggregate_column' in user_input_config:
                self.config.aggregate_column = user_input_config['aggregate_column']

    def initialize(self):
        ecf = self.config
        logger.debug(ecf)
        # logger.debug("pattern_table is")
        # logger.debug(pattern_table)

        # logger.debug("query_result_table is")
        # logger.debug(query_result_table)

        # print(opts)
        start = time.clock()
        logger.info("start explaining ...")
        self.global_patterns, self.schema, self.global_patterns_dict = load_patterns(
            ecf.cur, ecf.pattern_table, ecf.query_result_table,
            ecf.pattern_theta, ecf.pattern_lambda)
        logger.debug("loaded patterns from database")


        if ecf.similarity_matrix_file is None:
            if ecf.query_result_table.find('crime') == -1:
                self.category_similarity = CategorySimilarityNaive(cur=ecf.cur, table_name=ecf.query_result_table)
            else:
                self.category_similarity = CategorySimilarityNaive(cur=ecf.cur, table_name=ecf.query_result_table,
                                                                   embedding_table_list=[
                                                                       ('community_area', 'community_area_loc')])
        else:
            self.category_similarity = CategorySimilarityMatrix(inf=ecf.similarity_matrix_file)
        # category_similarity = CategoryNetworkEmbedding(EXAMPLE_NETWORK_EMBEDDING_PATH, data['df'])
        # num_dis_norm = normalize_numerical_distance(data['df'])
        self.num_dis_norm = normalize_numerical_distance(cur=ecf.cur, table_name=ecf.query_result_table)
        end = time.clock()
        print('Loading time: ' + str(end - start) + 'seconds')
        logger.debug(ExplConfig.MATERIALIZED_DICT)

    def wrap_user_question(self, global_patterns, global_patterns_dict, uq_tuple, schema=None):
        '''
            wrap user questions
        '''
        row_data = {}
        agg_col = None
        # if 'lambda' not in uq_tuple:
        #     uq_tuple['lambda'] = 0.1
        for k, v in enumerate(uq_tuple):
            # print(k, v)
            if schema is None or v not in schema:
                if v != 'direction':
                    if is_float(uq_tuple[v]):
                        row_data[v] = float(uq_tuple[v])
                    elif is_integer(uq_tuple[v]):
                        row_data[v] = float((uq_tuple[v]))
                    else:
                        row_data[v] = uq_tuple[v]
                if v not in schema and v != 'lambda' and v != 'direction':
                    agg_col = v
            else:
                if uq_tuple[v] != '*':
                    if v.startswith('count_') or v.startswith('sum_'):
                        agg_col = v
                    if schema[v] == 'integer':
                        row_data[v] = int(uq_tuple[v])
                    elif schema[v].startswith('double') or schema[v].startswith('float'):
                        row_data[v] = float(uq_tuple[v])
                    else:
                        row_data[v] = uq_tuple[v]

        if uq_tuple['direction'][0] == 'h':
            dir = 1
        else:
            dir = -1

        if 'count_*' in row_data:
            row_data['count'] = row_data['count_*']
        uq = {'target_tuple': row_data, 'dir': dir, 'query_result': []}

        logger.debug("uq is")
        logger.debug(uq)

        return [uq]

    def do_explain_online(self, uq_tuple):

        ecf = self.config
        query_result_table = ecf.query_result_table
        pattern_table = ecf.pattern_table
        aggregate_column = ecf.aggregate_column
       
        logger.debug("pattern_table is")
        logger.debug(pattern_table)

        logger.debug("query_result_table is")
        logger.debug(query_result_table)

        Q = self.wrap_user_question(self.global_patterns, self.global_patterns_dict, uq_tuple, self.schema)

        logger.debug("start finding explanations ...")

        start = time.clock()
        # regression_package = 'scikit-learn'
        # regression_package = 'statsmodels'

        explanations_list, local_patterns_list, score_computing_time_list = find_explanation_regression_based(
            Q, self.global_patterns, self.global_patterns_dict, self.category_similarity, self.num_dis_norm,
            aggregate_column, ecf
        )

        end = time.clock()
        logger.debug('Total querying time: ' + str(end-start) + 'seconds')
        logger.debug("finding explanations ... DONE")

        for g_key in ecf.MATERIALIZED_DICT:
            for fv_key in ecf.MATERIALIZED_DICT[g_key]:
                dv_query = '''DROP VIEW IF EXISTS MV_{};'''.format(str(ecf.MATERIALIZED_DICT[g_key][fv_key]))
                ecf.cur.execute(dv_query)
                ecf.conn.commit()
        return explanations_list[0]

    def do_batch_explain(self):

        ecf = self.config
        query_result_table = ecf.query_result_table
        pattern_table = ecf.pattern_table
        user_question_file = ecf.user_question_file
        outfile = ''
        aggregate_column = ecf.aggregate_column
        conn = ecf.conn
        cur = ecf.cur
        logger.debug(ExplConfig.MATERIALIZED_DICT)
        start = time.clock()
        logger.info("start explaining ...")
        global_patterns, schema, global_patterns_dict = load_patterns(cur, pattern_table, query_result_table)
        logger.debug("loaded patterns from database")
        logger.debug(ExplConfig.MATERIALIZED_DICT)

        # # category_similarity = CategorySimilarityMatrix(ecf.EXAMPLE_SIMILARITY_MATRIX_PATH, schema)
        # category_similarity = CategorySimilarityNaive(cur=cur, table_name=query_result_table)
        # # category_similarity = CategoryNetworkEmbedding(EXAMPLE_NETWORK_EMBEDDING_PATH, data['df'])
        # #num_dis_norm = normalize_numerical_distance(data['df'])
        # num_dis_norm = normalize_numerical_distance(cur=cur, table_name=query_result_table)
        if query_result_table.find('crime') == -1:
            category_similarity = CategorySimilarityNaive(cur=cur, table_name=query_result_table)
        else:
            category_similarity = CategorySimilarityNaive(cur=cur, table_name=query_result_table, embedding_table_list=[
                ('community_area', 'community_area_loc')])

        num_dis_norm = normalize_numerical_distance(cur=cur, table_name=query_result_table)
        logger.debug(ExplConfig.MATERIALIZED_DICT)

        # pf = PatternFinder(engine.connect(), query_result_table, fit=True, theta_c=0.5, theta_l=0.25,
        #                    lamb=DEFAULT_LAMBDA, dist_thre=0.9, supp_l=10, supp_g=1)

        Q, global_patterns, global_patterns_dict = load_user_question_from_file(
            global_patterns, global_patterns_dict, user_question_file,
            schema, conn, cur, pattern_table, query_result_table, None, category_similarity)

        logger.debug("loaded user question from file")
        logger.debug(ExplConfig.MATERIALIZED_DICT)
        end = time.clock()
        print('Loading time: ' + str(end - start) + 'seconds')

        logger.debug("start finding explanations ...")

        start = time.clock()
        # regression_package = 'scikit-learn'
        # regression_package = 'statsmodels'

        explanations_list, local_patterns_list, score_computing_time_list = find_explanation_regression_based(
            Q, global_patterns, global_patterns_dict, category_similarity, num_dis_norm,
            aggregate_column, ecf
        )
        logger.debug(ExplConfig.MATERIALIZED_DICT)
        end = time.clock()
        print('Total querying time: ' + str(end - start) + 'seconds')
        logger.debug("finding explanations ... DONE")

        ofile = sys.stdout
        if outfile != '':
            ofile = open(outfile, 'w')

        for i, top_k_list in enumerate(explanations_list):
            ofile.write('User question {} in direction {}: {}\n'.format(
                str(i + 1), 'high' if Q[i]['dir'] > 0 else 'low', str(Q[i]['target_tuple']))
            )

            for j, e in enumerate(top_k_list):
                ofile.write('------------------------\n')
                ofile.write('Top ' + str(j + 1) + ' explanation:\n')
                ofile.write(e.to_string())
                ofile.write('------------------------\n')

        # for g_key in ecf.MATERIALIZED_DICT:
        #     for fv_key in ecf.MATERIALIZED_DICT[g_key]:
        #         dv_query = '''DROP VIEW IF EXISTS MV_{};'''.format(str(ecf.MATERIALIZED_DICT[g_key][fv_key]))
        #         cur.execute(dv_query)
        #         conn.commit()
        # ecf.MATERIALIZED_DICT = dict()
        # ecf.MATERIALIZED_CNT = 0


def main(argv=[]):
    try:
        opts, args = getopt.getopt(argv, "h:p:q:u:o:a",
                                   ["help", "ptable=", "qtable=", "ufile=", "ofile=", "aggregate_column="])
    except getopt.GetoptError:
        print('explanation.py -p <pattern_table> -q <query_result_table>  -u <user_question_file>\
         -o <outfile> -a <aggregate_column>')
        sys.exit(2)
    # user_input_config = dict()
    user_input_config = ExplConfig()
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('explanation.py -p <pattern_table> -q <query_result_table> -u <user_question_file> \
                -o <outfile> -a <aggregate_column>')
            sys.exit(2)
        elif opt in ("-p", "--ptable"):
            user_input_config['pattern_table'] = arg
        elif opt in ("-q", "--qtable"):
            user_input_config['query_result_table'] = arg
        elif opt in ("-u", "--ufile"):
            user_input_config['user_question_file'] = arg
        elif opt in ("-o", "--ofile"):
            user_input_config['outfile'] = arg
        elif opt in ("-a", "--aggcolumn"):
            user_input_config['aggregate_column'] = arg

    eg = ExplanationGenerator(user_input_config)
    # eg.doExplain()
    eg.initialize()
    elist = eg.do_explain_online(
        {'name': 'Jiawei Han', 'venue': 'kdd', 'year': 2007, 'sum_pubcount': 1,  'direction': 'low'})
    # elist = eg.do_explain_online({'name': 'Kirsten Bergmann', 'venue': 'iva', 'sum_pubcount': 6.0, 'direction': 'high', 'lambda': 0.2})

    # elist = eg.do_explain_online({'primary_type': 'BATTERY', 'community_area': '26', 'year': '2011', 'count': 16, 'lambda': 0.2, 'direction': 'low'})
    for e in elist:
        print(e.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
