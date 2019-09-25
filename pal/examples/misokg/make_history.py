import cPickle as pickle

a = pickle.load(open("data_dumps/0.history", 'r'))
a = [x for x in a if x[0] == 0]
pickle.dump(a, open("data_dumps/0.history2", 'w'))
