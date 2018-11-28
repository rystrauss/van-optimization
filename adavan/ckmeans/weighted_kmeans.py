"""Weighted kmeans clustering with size constraints.

This implementation comes from https://github.com/Behrouz-Babaki/MinSizeKmeans.
"""

import random

import numpy as np
import pulp


def cluster(data, k=3, min_weight=0, max_weight=None, weights=None, max_iter=1000):
    if weights is None:
        weights = np.ones((data.shape[0],), dtype=np.uint8)
    best = None
    best_clusters = None
    for i in range(max_iter):
        clusters, centers = _minsize_kmeans_weighted(data, k=k, min_weight=min_weight, max_weight=max_weight,
                                                     weights=weights)
        if clusters:
            quality = _compute_quality(data, clusters)
            if not best or (quality < best):
                best = quality
                best_clusters = clusters

    return best_clusters


def _l2_distance(point1, point2):
    return sum([(float(i) - float(j)) ** 2 for (i, j) in zip(point1, point2)])


class _SubProblem(object):
    def __init__(self, centroids, data, weights, min_weight, max_weight):

        self.centroids = centroids
        self.data = data
        self.weights = weights
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.n = len(data)
        self.k = len(centroids)
        self.y = None
        self.status = None
        self.model = self._create_model()

    def _create_model(self):
        def distances(assignment):
            return _l2_distance(self.data[assignment[0]], self.centroids[assignment[1]])

        assignments = [(i, j) for i in range(self.n) for j in range(self.k)]

        # assignment variables
        self.y = pulp.LpVariable.dicts('data-to-cluster assignments',
                                       assignments,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)

        # create the model
        model = pulp.LpProblem("Model for assignment subproblem", pulp.LpMinimize)

        # objective function
        model += pulp.lpSum(
            [distances(assignment) * self.weights[assignment[0]] * self.y[assignment] for assignment in
             assignments]), 'Objective Function - sum weighted squared distances to assigned centroid'
        # this is also weighted, otherwise the weighted centroid computation don't make sense.

        # constraints on the total weights of clusters
        for j in range(self.k):
            model += pulp.lpSum([self.weights[i] * self.y[(i, j)] for i in
                                 range(self.n)]) >= self.min_weight, "minimum weight for cluster {}".format(j)
            model += pulp.lpSum([self.weights[i] * self.y[(i, j)] for i in
                                 range(self.n)]) <= self.max_weight, "maximum weight for cluster {}".format(j)

        # make sure each point is assigned at least once, and only once
        for i in range(self.n):
            model += pulp.lpSum([self.y[(i, j)] for j in range(self.k)]) == 1, "must assign point {}".format(i)

        return model

    def solve(self):
        self.status = self.model.solve()

        clusters = None
        if self.status == 1:
            clusters = [-1 for i in range(self.n)]
            for i in range(self.n):
                for j in range(self.k):
                    if self.y[(i, j)].value() > 0:
                        clusters[i] = j
        return clusters


def _compute_centers(clusters, dataset, weights=None):
    """
    weighted average of datapoints to determine centroids
    """
    if weights is None:
        weights = [1] * len(dataset)
    # canonical labeling of clusters
    ids = list(set(clusters))
    c_to_id = dict()
    for j, c in enumerate(ids):
        c_to_id[c] = j
    for j, c in enumerate(clusters):
        clusters[j] = c_to_id[c]

    k = len(ids)
    dim = len(dataset[0])
    cluster_centers = [[0.0] * dim for i in range(k)]
    cluster_weights = [0] * k
    for j, c in enumerate(clusters):
        for i in range(dim):
            cluster_centers[c][i] += dataset[j][i] * weights[j]
        cluster_weights[c] += weights[j]
    for j in range(k):
        for i in range(dim):
            cluster_centers[j][i] = cluster_centers[j][i] / float(cluster_weights[j])
    return clusters, cluster_centers


def _initialize_centers(dataset, k):
    ids = list(range(len(dataset)))
    random.shuffle(ids)
    return [dataset[id] for id in ids[:k]]


def _minsize_kmeans_weighted(data, k=3, weights=None, min_weight=0, max_weight=None, max_iter=1000):
    assert isinstance(data, np.ndarray), 'data must be a numpy array'

    n = data.shape[0]
    if weights is None:
        weights = np.full((n,), fill_value=-1, dtype=np.int8)
    if max_weight is None:
        max_weight = sum(weights)

    centers = _initialize_centers(data, k)
    clusters = np.full((n,), fill_value=-1)

    for step in range(max_iter):
        m = _SubProblem(centers, data, weights, min_weight, max_weight)
        clusters_ = m.solve()
        if not clusters_:
            return None, None
        clusters_, centers = _compute_centers(clusters_, data)

        converged = all([clusters[i] == clusters_[i] for i in range(n)])
        clusters = clusters_
        if converged:
            break

    return clusters, centers


def _cluster_quality(cluster):
    if len(cluster) == 0:
        return 0.0

    quality = 0.0
    for i in range(len(cluster)):
        for j in range(i, len(cluster)):
            quality += _l2_distance(cluster[i], cluster[j])
    return quality / len(cluster)


def _compute_quality(data, cluster_indices):
    clusters = dict()
    for i, c in enumerate(cluster_indices):
        if c in clusters:
            clusters[c].append(data[i])
        else:
            clusters[c] = [data[i]]
    return sum(_cluster_quality(c) for c in clusters.values())
