from capexplain.utils import *
from capexplain.explain.explanation import logger, ExplConfig


def get_tuples_by_F_V(lp1, lp2, f_value, v_value, conn, cur, table_name, cat_sim):
    def tuple_column_to_str_in_where_clause_2(col_value):
        # logger.debug(col_value)
        # logger.debug(cat_sim.is_categorical(col_value[0]))
        if cat_sim.is_categorical(col_value[0]):
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
        if cat_sim.is_categorical(col_value[0]):
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
        if cat_sim.is_categorical(col_value[0]):
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

    if G_key not in MATERIALIZED_DICT:
        MATERIALIZED_DICT[G_key] = dict()

    if f_value_key not in MATERIALIZED_DICT[G_key]:
        MATERIALIZED_DICT[G_key][f_value_key] = MATERIALIZED_CNT
        dv_query = '''DROP VIEW IF EXISTS MV_{};'''.format(str(MATERIALIZED_CNT))
        cur.execute(dv_query)
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
        F2, V2, lp2[3], str(MATERIALIZED_DICT[G_key][f_value_key]), where_clause
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
        if cat_sim.is_categorical(col_value[0]):
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

    if G_key not in MATERIALIZED_DICT:
        MATERIALIZED_DICT[G_key] = dict()

    if f_value_key not in MATERIALIZED_DICT[G_key]:
        MATERIALIZED_DICT[G_key][f_value_key] = MATERIALIZED_CNT
        dv_query = '''DROP VIEW IF EXISTS MV_{};'''.format(str(MATERIALIZED_CNT))
        cur.execute(dv_query)

        cmv_query = '''
            CREATE VIEW MV_{} AS SELECT {}, {} as {} FROM {} WHERE {} GROUP BY {};
        '''.format(
            str(MATERIALIZED_CNT), G_key, agg_fun, gp[2], table_name,
            ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]),
                                  zip(gp[0], map(tuple_column_to_str_in_where_clause_2, zip(F1_list, f_value)))))),
            G_key
        )
        # logger.debug(cmv_query)
        cur.execute(cmv_query)
        conn.commit()
        MATERIALIZED_CNT += 1

    where_clause = ' AND '.join(
        list(map(lambda x: "{} {}".format(x[0], x[1]), zip(gp[0], map(
            tuple_column_to_str_in_where_clause_2, zip(F1_list, f_value)))))) + ' AND ' + \
                   ' AND '.join(list(map(lambda x: "{} {}".format(x[0], x[1]),
                                         zip(gp[1],
                                             map(tuple_column_to_str_in_where_clause_2, zip(V1_list, v_value))))))

    tuples_query = '''SELECT {} FROM MV_{} WHERE {};'''.format(
        gp[2], str(MATERIALIZED_DICT[G_key][f_value_key]), where_clause
    )
    # logger.debug(tuples_query)
    cur.execute(tuples_query)
    res = cur.fetchall()
    if len(res) == 0:
        return []
    return res[0]

