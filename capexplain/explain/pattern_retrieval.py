from capexplain.utils import *
from capexplain.explain.explanation import logger, ExplConfig
from capexplain.explain.tuple_retrieval import get_tuples_by_gp_uq


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
                    if pat_key not in ExplConfig.VISITED_DICT:
                        ExplConfig.VISITED_DICT[pat_key] = True
    return gp_list


def find_patterns_relevant(global_patterns_dict, t, conn, cur, query_table_name, cat_sim):
    res_list = []
    t_set = set(t.keys())
    # logger.debug(global_patterns_dict[1].keys())
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
        # if 'primary_type' in pat[1] or 'description' in pat[1] or 'location_description' in pat[1] or 'community_area' in pat[1] or 'beat' in pat[1]:
        #     continue

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
