import utils
import pandas

def encode(train, test, option):
    cats = [c for c, d in zip(train.columns, train.dtypes) if str(d) == 'object']
    for c in cats:
        train[c] = train[c].apply(str)
        test[c] = test[c].apply(str)
    if option == "one_hot":
        return one_hot(train, test)
    if option == "to_int":
        return to_int(train, test)

def one_hot(train, test):
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

#train = pandas.DataFrame({"colA" : ["A", "B", "C", "A", "A"]})
#test = pandas.DataFrame({"colA" : ["E", "F", "C", "A", "A"]})

#print(encode(train, test, "one_hot"))

