# soft-margin linear kernel SVM  with an adjustable C parameter

import numpy as np
import matplotlib.pyplot as plt

import cvxopt as cvxopt # convex optimization library
from cvxopt import solvers # quadratic programmer
# pip install cvxopt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

class svm:

    def __init__(self,C=0.1):
        self.threshold = 0.00001 # threshold for the Lagrange multipliers
        self.C = C # hyperparameter that trades off margin width with misclassifications

    def train(self,data,targets):
        self.K = np.dot(data,data.transpose()) # linear kernel
        self.num = np.shape(data)[0] # number of data points

        # I Followed this guide to Quadratic Programming with cvxopt:
        # https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf

        # equation (8.26) in the Marsland textbook:
        # solve for x: min (1/2)x^T*P*x + q^T*x subject to Gx <= h, Ax = b
        # x is the vector of Lagrange multipliers (lambdas)
        P = np.dot(targets,targets.transpose())*self.K
        q = -np.ones((self.num,1)) # q^T*x = -sum(x)

        # Constraints:
        A = targets.reshape(1,self.num)
        b = 0.0
        # Ax = b is equivalent to sum(lambdas*targets) = 0

        # Gx <= h:
        if self.C is None or self.C == 0: # hard-margin
            # lambdas >= 0
            G = np.eye(self.num)*-1 # multiply by -1 for >= constraint
            h = np.zeros((self.num,1))
            # thus Gx = -x and h = 0 vector
            # x >= 0 becomes -x <= 0
        else:
            # 0 <= lambdas <= C
            G = np.concatenate((np.eye(self.num),np.eye(self.num)*-1))
            # G is an identity matrix concatenated with a negative identity matrix
            # Gx = x concatenated with -x
            h = np.concatenate((np.ones((self.num,1))*self.C,np.zeros((self.num,1))))
            # h = [C, C, C, ..., 0, 0, 0]
            # thus x <= C and x >= 0

        solvers.options['show_progress'] = False # silences cvxopt solver so it won't print to terminal
        sol = cvxopt.solvers.qp(cvxopt.matrix(P),cvxopt.matrix(q),\
            cvxopt.matrix(G),cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b), sym_proj=False)
        del G, P, A
        lambdas = np.array(sol['x']) # x is the vector of Lagrange multipliers (lambdas)
        self.svIndex = np.where(lambdas>self.threshold)[0] # indices of support vectors
        self.supports = data[self.svIndex,:]
        self.alphas = lambdas[self.svIndex] * targets[self.svIndex] # alpha values for each support vector

        # I followed equation (8.10) in the Marsland textbook for calculating b
        self.b = np.sum(targets[self.svIndex])
        for i in range(len(self.svIndex)):
            for j in range(len(self.svIndex)):
                self.b -= self.alphas[i]*self.K[self.svIndex[i],self.svIndex[j]]
        self.b = self.b / len(self.supports) # average over all the support vectors

    def classify(self,data,hard=True):
        K = np.dot(data,self.supports.transpose()) # linear kernel

        output = np.zeros((np.shape(data)[0],1))
        for i in range(np.shape(data)[0]):
            for j in range(len(self.svIndex)):
                #output[i] += self.alphas[j]*np.dot(data[i],self.supports[j].transpose())
                output[i] += self.alphas[j]*K[i,j]
            output[i] += self.b
        if hard:
            return np.sign(output)
        else:
            return output

def test(trainInp,trainTarg,testInp,testTarg,nClasses,hard=False,C=0.1):

    targets = -np.ones((np.shape(trainTarg)[0],nClasses),dtype=float);

    for c in range(nClasses):
        targets[:,c] = np.where(trainTarg==c, 1, -1)
    output = np.zeros((np.shape(testTarg)[0],nClasses))

    for c in range(nClasses):
        svc = svm(C=C)
        svc.train(trainInp,np.reshape(targets[:,c],(np.shape(trainInp[:,:])[0],1)))
        output[:,c] = svc.classify(testInp,hard).transpose()
        del svc

    bestclass = np.argmax(output,axis=1)

    confm = np.zeros((nClasses,nClasses))
    for i in range(nClasses):
        for j in range(nClasses):
            confm[i,j] = np.sum(np.where(bestclass==i,1,0)*np.where(testTarg==j,1,0))

    print(confm) # print confusion matrix
    print("Accuracy: {:.2f}%".format(100*np.trace(confm)/np.sum(confm))) # print the accuracy

 
# Load IRIS Data Set
iris = datasets.load_iris()
x = iris.data
y = iris.target
 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify = y)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

test(x_train_std,y_train,x_test_std,y_test,nClasses=3,hard=False,C=0.1) # achieves 88.89% accuracy
