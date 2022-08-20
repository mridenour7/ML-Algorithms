import numpy as np
import matplotlib.pyplot as plt

class kmeans:
    def __init__(self,k,data):
        self.nPoints = np.shape(data)[0] # number of rows
        self.nDim = np.shape(data)[1] # number of columns
        self.k = k
        self.centers = np.zeros([self.k, self.nDim]) # centers 

    def train(self,data,initializations):
        finalcenters = np.zeros([self.k, self.nDim, initializations]) # final centers for each iteration
        finalerrors = np.zeros([initializations]) # final sum of squares error for each iteration
        for r in range(initializations):
            self.centers = np.random.rand(self.k, self.nDim) * (data.max(0)-data.min(0)) + data.min(0) # start with random centers
            i=0
            while True:
                #print("Iteration %i:"%(i+1))
                #self.classify(data) # plot each training iteration
                distance = np.zeros([self.k, self.nPoints]) # matrix of distances
                for row in range(self.k):
                    distance[row,:] = np.sum(np.square(data-self.centers[row,:]),1)
                cluster = distance.argmin(0) # closest cluster
                error = np.sum(np.amin(distance, 0)) # sum of squares error

                newCenters = np.zeros([self.k, self.nDim])
                for c in range(self.k):
                    clusterSize = np.shape(cluster[cluster == c])[0]
                    if clusterSize == 0:
                        continue
                    for d in range(self.nDim):
                        newCenters[c, d] = np.sum(data[cluster == c , d]) / clusterSize # new cluster centers

                if(np.sum(newCenters - self.centers) == 0): # converged
                    break
                self.centers = newCenters
                i+=1

            finalerrors[r] = error
            finalcenters[:,:,r] = self.centers # save the final center for the rth iteration
            #print("Iteration %i:"%(r+1))
            #self.classify(data) # plot each initialization iteration
            
        self.centers = finalcenters[:,:,finalerrors.argmin(0)] # centers with lowest mean square error
        #print("The iteration with the lowest average mean square error was %i:"%(finalerrors.argmin(0)+1))
        self.classify(data) 
            
    def classify(self,data):
        distance = np.zeros([self.k, self.nPoints]) # matrix of distances
        for row in range(self.k):
            distance[row,:] = np.sum(np.square(data-self.centers[row,:]),1) # matrix of distances
        cluster = distance.argmin(0) # closest cluster
        error = np.sum(np.amin(distance, 0)) # sum of squares error
        # plot clustered data points
        if self.nDim == 2:
            for c in range(self.k):
                plt.scatter(data[cluster == c, 0], data[cluster == c, 1], label = c)
                plt.scatter(self.centers[c,0], self.centers[c,1], c="black")
        else:
            ax = plt.axes(projection ="3d")
            for c in range(self.k):
                ax.scatter(data[cluster == c, 0], data[cluster == c, 1], data[cluster == c, 2], label = c)
        plt.title('k=%i'%self.k)
        plt.legend()
        plt.show()
