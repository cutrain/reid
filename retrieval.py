import xxhash
import falconn
import numpy as np
from collections import defaultdict


class RetrievalTable(object):
    def __init__(self, seed=20180101):
        self.seed = seed
        self.tables = {}
        self.last_table = None

    def retrival(self, query, dataset=None, *, k=None, threshold=None):
        if dataset is None:
            table = self.last_table
        else:
            hashint = xxhash.xxh64(dataset[:,0].copy(), self.seed).intdigest()
            if hashint in self.tables:
                table = self.tables[hashint]
            else:
                print('find a new dataset')
                dataset = dataset.astype(np.float32)
                mean = np.mean(dataset, axis=0)
                dataset -= mean
                params = falconn.get_default_parameters(dataset.shape[0], dataset.shape[1])
                falconn.compute_number_of_hash_functions(7, params)
                lsh_index = falconn.LSHIndex(params)
                lsh_index.setup(dataset)
                qtable = lsh_index.construct_query_object()
                qtable.set_num_probes(10000)
                table = (mean, qtable)
                self.tables[hashint] = table
        if table is None:
            raise Exception("Dataset not specific")
        query -= table[0]
        if k is not None and threshold is not None:
            raise ValueError("k and threshold should not pass simultaneously")
        self.last_table = table
        if k is not None:
            return table[1].find_k_nearest_neighbors(query, k)
        if threshold is not None:
            return table[1].find_near_neighbors(query, threshold)
        return table[1].find_nearest_neighbor(query)

def force_retrieval(query, dataset, *, k=None, threshold=None):
    temp = np.sum((dataset-query)**2, axis=1)
    rank = np.argsort(temp)
    if k is not None:
        return list(rank[:k])
    if threshold is not None:
        k = np.count_nonzero(temp<=threshold)
        return list(rank[:k])
    return list(rank)


rt = RetrievalTable()


def retrieval(query, dataset, *, k=None, threshold=None, mode='force'):
    global rt
    if isinstance(query, list):
        query = np.stack(query)
    if isinstance(dataset, list):
        dataset = np.stack(dataset)
    assert query.shape[0:] == dataset.shape[1:], "shape not match"
    query = query.astype(np.float32)
    dataset = dataset.astype(np.float32)
    if mode == 'falconn':
        return rt.retrival(query, dataset, k=k, threshold=threshold)
    elif mode == 'force':
        return force_retrieval(query, dataset, k=k, threshold=threshold)
    else:
        raise NotImplemented


