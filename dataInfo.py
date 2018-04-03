class dataInfo:
    def __init__(self, inst, clsDist = None, indices = None, clusters = None, centroids = None,
        N = None, K = None, k = None, itert = None, dist_Cnt = None, SSN = None):
        self.data = inst
        self.clsDist = clsDist
        self.indices = indices
        self.clusters = clusters
        self.centroids = centroids
        self.N = N
        self.K = K
        self.k = k
        self.itert = itert
        self.dist_Cnt = dist_Cnt
        self.SSN = SSN