# multilayer perceptron trained on the mnist dataset
# dataset: http://yann.lecun.com/exdb/mnist/
# 20 hidden nodes produces a 94% accuracy on the test set after 20 epochs

import numpy as np
import matplotlib.pyplot as plt # used to plot accuracies
import sys # used to prevent float overflow from exp()

data_path = "datasets/"
train_data = np.loadtxt(data_path + "mnist_train.csv",delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

inp = np.asfarray(train_data[:, 1:])/255 # inputs
targ = np.int_(train_data[:, 0]) # targets
targ = np.eye(10)[targ] # one-hot encoding
targ = targ*0.8 + 0.1 # change 1 and 0 to 0.9 and 0.1 
del train_data # free up memory

testInp = np.asfarray(test_data[:, 1:])/255 # test inputs
testTarg = np.int_(test_data[:, 0])  # test targets
testTarg = np.eye(10)[testTarg] # one-hot encoding
del test_data

class neuralnet:
    
        def __init__(self,inputs,targets, nHidden):
                self.nIn = np.shape(inputs)[1] # number of columns in training data
                self.nTrain = np.shape(inputs)[0] # number of rows in training data
                self.nOut = np.shape(targets)[1] # number of columns in targets
                self.nHidden = nHidden # number of hidden nodes
                self.hWeights = np.random.rand(self.nIn+1,self.nHidden)*0.1-0.05   # initialize hidden weights in range [-.05, .05]
                self.oWeights = np.random.rand(self.nHidden+1,self.nOut)*0.1-0.05   # initialize output weights in range [-.05, .05]
                self.momentum = 0.9
                
        def sigmoid(self, x):
            # prevent float overflow from exp()
            # if x contains a very large negative number, replace it with -np.log(sys.float_info.max)+0.001
            safeX = np.where(-x < np.log(sys.float_info.max), x, -np.log(sys.float_info.max)+0.001)
            return 1.0/(1.0+np.exp(-safeX))

        def train(self,trainInputs,trainTargets,testTargets,testInputs,eta,nEpochs, batchSize):
                trainInputs = np.concatenate((trainInputs,np.ones((np.shape(trainInputs)[0],1))),1) # bias node = 1
                testInputs = np.concatenate((testInputs,np.ones((np.shape(testInputs)[0],1))),1) # bias node = 1
                trainAcc = np.array([]) # array of training accuracy per epoch
                testAcc = np.array([]) # array of test accuracies
                
                # keep track of previous change in weights (needed for momentum)
                hWeightUpdate = np.zeros((np.shape(self.hWeights))) # initialized to zero
                oWeightUpdate = np.zeros((np.shape(self.oWeights)))
                
                #trainAcc = np.append(trainAcc, self.accuracy(trainInputs,trainTargets,False)) # epoch 0
                #testAcc = np.append(testAcc, self.accuracy(testInputs,testTargets,False)) # epoch 0
                
                for n in range(nEpochs):
                        for b in range(int(self.nTrain/batchSize)):
                                inputs = trainInputs[(b*batchSize):((b+1)*batchSize),:] # split into batches
                                targets = trainTargets[(b*batchSize):((b+1)*batchSize),:]
                                
                                # forwards
                                hActivations = np.dot(inputs,self.hWeights) # dimensions: (batchSize,785) * (785, nHidden)
                                hActivations = self.sigmoid(hActivations) # sigmoid activation function
                                hActivations = np.concatenate((hActivations,np.ones((np.shape(inputs)[0],1))),1) # bias node = 1
                                oActivations = np.dot(hActivations, self.oWeights) # dimensions: (batchSize,nHidden+1) * (nHidden+1,10)
                                oActivations = self.sigmoid(oActivations) # sigmoid activation function
                                
                                # backwards
                                # calculate error terms
                                oDelta = np.multiply(oActivations, np.multiply((1-oActivations),(targets-oActivations)))
                                hDelta = np.dot(oDelta,np.transpose(self.oWeights)) # dimensions: (batchSize,10) * (10,nHidden+1)
                                hDelta = np.multiply(hActivations, np.multiply((1-hActivations),hDelta))
                                
                                # update the weights
                                oWeightUpdate = eta*(1/batchSize)*np.dot(np.transpose(hActivations),oDelta) + self.momentum*oWeightUpdate
                                self.oWeights += oWeightUpdate
                                hWeightUpdate = eta*(1/batchSize)*np.dot(np.transpose(inputs),hDelta[:,:-1]) + self.momentum*hWeightUpdate
                                self.hWeights += hWeightUpdate
                                
                        trainAcc = np.append(trainAcc, self.accuracy(trainInputs,trainTargets,False))
                        testAcc = np.append(testAcc, self.accuracy(testInputs,testTargets,False))
                        #if(abs(trainAcc[n]-trainAcc[n+1])<0.00001): # terminating condition
                        #        break
                        
                        # switch around the order of inputs after each epoch
                        change = list(range(self.nTrain))
                        np.random.shuffle(change)
                        trainInputs = trainInputs[change,:]
                        trainTargets = trainTargets[change,:]
                
                self.plotAccuracy(trainAcc,testAcc) # plot the accuracies
                self.accuracy(testInputs,testTargets,True) # print confusion matrix

        def accuracy(self,inputs,targets,printConfm):
                # forwards
                hActivations = np.dot(inputs,self.hWeights)
                hActivations = self.sigmoid(hActivations)
                hActivations = np.concatenate((hActivations,np.ones((np.shape(inputs)[0],1))),1) # bias node = 1
                oActivations = np.dot(hActivations, self.oWeights)
                oActivations = self.sigmoid(oActivations)
                
                outputs = np.argmax(oActivations, 1) # encode as 1-of-N
                targets = np.argmax(targets, 1) # encode as 1-of-N

                # create the confusion matrix
                confm = np.zeros((self.nOut,self.nOut))
                for i in range(self.nOut):
                        for j in range(self.nOut):
                                confm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

                if(printConfm):
                        print(confm.astype(int)) # print the confusion matrix
                return np.trace(confm)/np.sum(confm) # return the accuracy
            
        def plotAccuracy(self, trainAcc,testAcc):
                epochs = np.zeros(np.shape(trainAcc)[0])
                for n in range(np.shape(trainAcc)[0]):
                    epochs[n] = n+1 # exclude epoch 0
                plt.plot(epochs,trainAcc,'o-g')
                plt.plot(epochs,testAcc,'o-b')
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.title('Accuracy vs Epochs (Hidden=%s)' %self.nHidden)
                plt.legend(["Training", "Testing"], loc="lower right")
                plt.show()
                
p = neuralnet(inp,targ,nHidden=20)
p.train(inp,targ,testTarg,testInp,eta=0.1,nEpochs=20,batchSize=60)
