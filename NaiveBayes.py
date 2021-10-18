# Gaussian Naive Bayes classifier trained on the Spambase dataset from the UCI ML repository
# dataset: https://archive.ics.uci.edu/ml/datasets/spambase
import numpy as np
import math # used for pi

data = np.loadtxt("spambase.data",delimiter=",")

trainPos = data[:906, :] # 906 spam
testData = data[906:1813, :] # 907 spam
trainNeg = data[1813:3207, :] # 1394 not-spam
testData = np.vstack([testData, data[3207:4600, :]]) # 1393 not-spam

class classifier:

    def __init__(self,trainPos,trainNeg):
        self.nFeatures = np.shape(trainPos)[1] - 1 # number of columns in training data (minus the label)
        self.nClasses = 2
        self.nPos = np.shape(trainPos)[0] # number of rows
        self.nNeg = np.shape(trainNeg)[0] # number of rows
        self.posPrior = self.nPos / (self.nPos + self.nNeg) # prior
        self.negPrior = self.nNeg / (self.nPos + self.nNeg) # prior
        self.deviations = np.zeros([self.nFeatures, self.nClasses]) + 0.0001   # standard deviations for each feature (columns are classes)
        self.means = np.zeros([self.nFeatures, self.nClasses])   # means for each feature (columns are classes)

    def train(self,trainPos,trainNeg):

        for f in range(self.nFeatures): # compute means and deviations given negative class
            mean = sum(i for i in trainNeg[:, f]) / self.nNeg # scalar
            self.means[f, 0] = mean
            deviations = trainNeg[:, f] - mean
            deviations = np.square(deviations)
            deviation = sum(d for d in deviations) / self.nNeg # scalar
            deviation = np.sqrt(deviation)
            if deviation < 0.0001:
                deviation = 0.0001  # avoid a divide-by-zero error
            self.deviations[f, 0] = deviation

        for f in range(self.nFeatures): # compute means and deviations given positive class
            mean = sum(i for i in trainPos[:, f]) / self.nPos # scalar
            self.means[f, 1] = mean
            deviations = trainPos[:, f] - mean
            deviations = np.square(deviations)
            deviation = sum(d for d in deviations) / self.nPos  # scalar
            deviation = np.sqrt(deviation)
            if deviation < 0.0001:
                deviation = 0.0001  # avoid a divide-by-zero error
            self.deviations[f, 1] = deviation

    def test(self,testData):

        posterior = np.zeros(self.nClasses) # posterior for each class
        predicted = np.zeros(len(testData)) # predicted class
        targets = testData[:, 57]

        for i in range(len(testData)):
            x = np.transpose(testData[i, :57]) # a test data point
            for c in range(self.nClasses):
                likelihoods = np.square(x - self.means[:, c]) / (2*np.square(self.deviations[:, c]))
                likelihoods = np.exp(-likelihoods) / (np.sqrt(2*math.pi)*self.deviations[:, c])
                likelihoods = np.where(likelihoods==0, 0.1**300, likelihoods) # avoid log(0)
                likelihoods = np.log(likelihoods) # log-likelihoods

                posterior[c] = sum(l for l in likelihoods) # add the log-likelihoods
                posterior[c] += np.log(self.negPrior) if c == 0 else np.log(self.posPrior) # add the log prior

            predicted[i] = 1 if posterior[1] > posterior[0] else 0 # find the most probable class value

        # create the confusion matrix
        confm = np.zeros((self.nClasses,self.nClasses))
        for i in range(self.nClasses):
            for j in range(self.nClasses):
                confm[i,j] = np.sum(np.where(predicted==i,1,0)*np.where(targets==j,1,0))

        print(confm.astype(int)) # print the confusion matrix
        print("Accuracy: {:.2f}%".format(100*np.trace(confm)/np.sum(confm))) # print the accuracy
        print("Precision: {:.2f}%".format(100*confm[0,0]/np.sum(confm[:,0]))) # print the precision
        print("Recall: {:.2f}%".format(100*confm[0,0]/np.sum(confm[0,:]))) # print the recall

c = classifier(trainPos, trainNeg)
c.train(trainPos,trainNeg)
c.test(testData)
