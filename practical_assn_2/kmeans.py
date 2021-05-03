
import numpy as np

X = np.genfromtxt('Data.tsv', delimiter='\t')

# step 1: for each of z element_of(1,2,...,k) (k=3) start with random initial guess for cluster centroid mew(C_z)
centroids = [[1.03800476, 0.09821729, 1.0469454, 1.58046376], [0.18982966, -1.97355361, 0.70592084, 0.3957741], [1.2803405, 0.09821729, 0.76275827, 1.44883158]]
prev_centroids = [[1.03800476, 0.09821729, 1.0469454, 1.58046376], [0.18982966, -1.97355361, 0.70592084, 0.3957741], [1.2803405, 0.09821729, 0.76275827, 1.44883158]]


# initialize clusters
prev_clusters = [0]*150
clusters = [0]*150
k = 3
closest = -1
cluster_count = [0, 0, 0]
condition = False
temp = [[],[],[]]
counter = 0

# function to find closest centroid
while (condition == False):
    flag = 1
    temp = [[],[],[]]
    for j in range(len(X)):
        sum_prev = 10000.0
        # i is each of the three cluster centroids
        for i in range(0,3):
            sum_dist = ((X[j][0] - centroids[i][0]) ** 2) + ((X[j][1] - centroids[i][1]) ** 2) + ((X[j][2] - centroids[i][2]) ** 2) + ((X[j][3] - centroids[i][3]) ** 2)
            if sum_dist < sum_prev:
                closest = i
                sum_prev = sum_dist
        print(closest)
        cluster_count[closest] = cluster_count[closest] + 1
        clusters[j] = closest
        temp[closest] = (centroids[closest] + X[j])
    for k in range(0,3): 
        if (cluster_count[k] != 0):
            list1 = [(temp[k][z] / cluster_count[k]) for z in range(0,4)]
            centroids[k] = list1
        else: 
            centroids[k] = [0.0, 0.0, 0.0, 0.0]
   
    if (prev_clusters == clusters):
        condition = True
        print("stop condition")
    prev_clusters = clusters.copy()
arrayclusters = np.array(clusters)
print(clusters)
np.savetxt('kmeans_output.tsv', clusters, delimiter="\t", newline = "\t", fmt = '%i')
      
 
    
    
    
    
