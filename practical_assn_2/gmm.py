import numpy as np

##   Violet Rally, 260772375   ##
## collaborated with Alex Gass ##


# extract data from tsv file
X_data = np.genfromtxt('Data.tsv', delimiter='\t')

# function to initialize vectors 
def initialize(X_data, k=3):
    # parse data into k=3 clusters
    X = np.array_split(X_data, k)

    # initializing parameters: mean and covariance matrices
    mean_vec = []
    cov_vec = []
    for x in X:
        mean_vec.append(np.mean(x, axis = 0))
        cov_vec.append(np.cov(x.T))
    # initializing pi values for k=3 clusters of assumed equal size
    pi_vec = [1/k for i in range(k)]
    
    # return parameters initialized here
    return mean_vec, cov_vec, pi_vec

# function to handle long calculation for multivariate norm 
def multivariate_norm(X, mean_vec, cov_vec):
    return (2*np.pi)**(-len(X)/2)*np.linalg.det(cov_vec)**(-1/2)*np.exp(-np.dot(np.dot((X - mean_vec).T, np.linalg.inv(cov_vec)), (X - mean_vec))/2)

def expectation_step(X, mean_vec, cov_vec, pi_vec, k=3):
    # initialize score matrix
    r_score = np.zeros((len(X), k))
    
    # update score matrix
    for n in range(len(X)):
        for z in range(k):
            r_score[n][z] = pi_vec[z] * multivariate_norm(X[n], mean_vec[z], cov_vec[z])
            r_score[n][z] = r_score[n][z] / sum([pi_vec[j] * multivariate_norm(X[n], mean_vec[j], cov_vec[j]) for j in range(k)])
   
    # sum of scores for points x
    N = np.sum(r_score, axis=0)
    return r_score, N

def maximization_step(X, r_score, N, k=3):
    mean_vec = np.zeros((k, len(X[0])))
    # resetting covariance matrix and repopulating with zeroes
    cov_vec = []
    for i in range(k):
        cov_vec.append(np.zeros((len(X[0]), len(X[0]))))
    pi_vec = np.zeros(k)

    # updating covariance, mean, and pi values 
    for z in range(k):
        cov_vec[z] = np.cov(X.T, aweights=(r_score[:, z]), ddof=0) / N[z]
        pi_vec[z] = N[z] / len(X)
        for i in range(len(X)):
            mean_vec[z] += r_score[i][z] * X[i]
        mean_vec[z] = mean_vec[z] / N[z]
    
    # returning values updated here 
    return mean_vec, cov_vec, pi_vec

def likelihoods(X, old_logl, logl, pi_vec, mean_vec, cov_vec, k=3):
    # computes updated likelihood value
    for i in range(len(X)):
        pre_log = 0
        for z in range(k):
            pre_log +=  pi_vec[z] * multivariate_norm(X[i], mean_vec[z], cov_vec[z])
        logl += np.log(pre_log)
    logl_diff = np.abs(logl - old_logl)
    old_logl = logl
    return logl_diff, old_logl 

def assign_clusters(X, r_score, mean_vec, cov_vec, k=3):
    probabilities = []
    clusters = []
    ind = [index for index in range(k)]
    for i in range(len(X)):
        probabilities.append([multivariate_norm(X[i], mean_vec[z], cov_vec[z]) for z in range(k)])
    for p in probabilities:
        clusters.append(ind[p.index(max(p))])
    return clusters

def write_to_tsv(clusters, filename):
    np.savetxt(filename, clusters, newline="\t", delimiter= "\t", fmt='%i')

def compute_GMM(X, tolerance, k=3):
    # populate matrices by initialization function, initialize other values for new iteration
    means, covs, pis = initialize(X)
    old_logl = 0
    logl_diff = 1
    r_score = []
    i = 0
    # iterate over expectation step and maximization step until convergence
    while (logl_diff > tolerance):
        i += 1
        logl = 0
        logl_diff, old_logl = likelihoods(X, old_logl, logl, pis, means, covs)
        r_score, N = expectation_step(X, means, covs, pis)
        means, covs, pis = maximization_step(X, r_score, N)
    print(i)
    # assign final cluster values and output result
    clusters = assign_clusters(X, r_score, means, covs)
    write_to_tsv(clusters, "gmm_output.tsv")

compute_GMM(X_data, 10e-5)