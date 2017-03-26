import utils

def fix_categorical(train, test, option, nullcol=True):
    cats = [c for c, d in zip(train.columns, train.dtypes) if str(d) == 'object']
    if nullcol:
        for c in cats:
            train[c] = train[c].apply(str)
            test[c] = test[c].apply(str)
    if option == "one_hot":
        return one_hot(train, test)
    if option == "to_int":
        return to_int(train, test)

def one_hot(train, test, nullcol=False):
    cats = [c for c, d in zip(train.columns, train.dtypes) if str(d) == 'object']
    onehotvals = utils.get_onehots(train, cats)
    utils.set_onehots(train, onehotvals, drop=True)
    utils.set_onehots(test, onehotvals, drop=True)
    return train, test

def to_int(train, test):
    cats = [c for c, d in zip(train.columns, train.dtypes) if str(d) == 'object']
    for col in cats:
        vals = list(train[col].unique()) + list(test[col].unique())
        train[col] = train[col].apply(lambda x: vals.index(x))
        test[col] = test[col].apply(lambda x: vals.index(x))
    return train, test

def fix_numeric(train, test, cols, fillna_mode='zero', scaling='normal'):
    if fillna_mode == 'zero':
        for c in cols:
            train[c].fillna(0, inplace=True)
            test[c].fillna(0, inplace=True)
    elif fillna_mode == 'mean':
        means = [train[c].mean() for c in cols]
        for c, m in zip(cols, means):
            train[c].fillna(m, inplace=True)
            test[c].fillna(m, inplace=True)
    elif fillna_mode == 'median':
        meds = [train[c].median() for c in cols]
        for c, m in zip(cols, meds):
            train[c].fillna(m, inplace=True)
            test[c].fillna(m, inplace=True)
    else:
        raise ValueError('Unknown value for fillna_mode')

    if scaling == 'normal':
        means = [train[c].mean() for c in cols]
        stds = [train[c].std() for c in cols]
        for c, m, s in zip(cols, means, stds):
            train[c] = (train[c] - m) / s
            test[c] = (test[c] - m) / s
    elif scaling == 'uniform':
        maxs = [train[c].max() for c in cols]
        mins = [train[c].min() for c in cols]
        for c, mx, mn in zip(cols, maxs, mins):
            train[c] = (train[c] - mn) / (mx - mn)
            test[c] = (test[c] - mn) / (mx - mn)
    elif scaling is None or scaling == 'none':
        pass
    else:
        raise ValueError('Unknown value for scaling')

    return train, test
