import datetime

def projection(t, cols):
    res = []
    for c in cols:
        if c in t:
            res.append(t[c])
    return res
    # return list(map(lambda x: t[x], cols))

def get_F_value(F, t):
    return projection(t, F)

def get_V_value(V, t):
    return projection(t, V)

def is_float(s):
    try:
        if isinstance(s, float) or isinstance(s, int) or isinstance(s, str):
            float(s)
            return True
        else:
            return False
    except ValueError:
        return False

def is_integer(s):
    try:
        if isinstance(s, str) and s[-1] == 'L':
            int(s[0:-1])
        else:
            int(s)
        return True
    except ValueError:
        return False

def float_or_integer(s):
    if is_float(s):
        return float(s)
    elif is_integer(s):
        return int(s[0:-1]) if s[-1] == 'L' else int(s)
    else:
        return s

def tuple_column_to_str_in_where_clause(col_value):
    if is_float(col_value):
        return '=' + str(col_value)
    else:
        return "like '%" + col_value + "%'"

def normalize_numerical_distance(df=None, cur=None, table_name=''):
    '''
        get the min value, max value, and the range of each column in the data frame
    '''
    if df is not None:
        res = {}
        max_vals = df.max()
        min_vals = df.min()
        for col in df:
            if col == 'index':
                continue
            if df[col].dtype.kind != 'S' and df[col].dtype.kind != 'O':
                res[col] = {'max':{}, 'min':{}, 'range':{}}
                
                res[col]['max'] = float(max_vals[col])
                res[col]['min'] = float(min_vals[col])
                res[col]['range'] = res[col]['max'] - res[col]['min']
        return res
    else:
        res = {}
        column_name_query = "SELECT column_name FROM information_schema.columns where table_name='{}';".format(table_name)
        cur.execute(column_name_query)
        column_name = cur.fetchall()
        
        max_clause = ', '.join(map(lambda x: 'MAX(' + x + ')', 
            map(lambda x: x[0] if x[0] != 'arrest' and x[0] != 'domestic' else '1', column_name)))
        max_query = "SELECT {} FROM {};".format(max_clause, table_name)
        cur.execute(max_query)
        max_vals = cur.fetchall()[0]
        min_clause = ', '.join(map(lambda x: 'MIN(' + x + ')',
            map(lambda x: x[0] if x[0] != 'arrest' and x[0] != 'domestic' else '1', column_name)))
        min_query = "SELECT {} FROM {};".format(min_clause, table_name)
        cur.execute(min_query)
        min_vals = cur.fetchall()[0]
        for idx, col_res in enumerate(column_name):
            col = col_res[0]
            res[col] = {'max':None, 'min':None, 'range':None}
            print('IN NORM', col, max_vals[idx], min_vals[idx])
            if is_float(max_vals[idx]) and is_float(min_vals[idx]):
                res[col]['max'] = float(max_vals[idx])
                res[col]['min'] = float(min_vals[idx])
                res[col]['range'] = res[col]['max'] - res[col]['min']
            else:
                if isinstance(max_vals[idx], datetime.date):
                    res[col]['max'] = max_vals[idx]
                    res[col]['min'] = min_vals[idx]
                    res[col]['range'] = res[col]['max'] - res[col]['min']
        return res

