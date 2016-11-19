"""
 Anomaly Detection in Human Dynamics Temporal Data Using Gaussian-HMM model 
 -------------------------------------------------------------------------
 2016/10/29 
"""
from __future__ import print_function
import datetime
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
try:
    from matplotlib.finance import quotes_historical_yahoo_ochl
except ImportError:
    # For Matplotlib prior to 1.5.
    from matplotlib.finance import (
        quotes_historical_yahoo as quotes_historical_yahoo_ochl
    )
from hmmlearn.hmm import GaussianHMM
import sys, os
from sklearn.cluster import KMeans


class HMMModel:
    def __init__(self):
        self.train = []
        self.test = []
        self.true_anomaly = []

    def setPath(self, path):
        self.OUTPATH = '%s'%path

    def addData(self, train=[], test=[]):
        if not train==[]:
            self.train = train
            self.T = len(train)
        if not test==[]:
            self.test = test

    def setModel(self, n_components=4, covariance_type='full', n_iter=1000):
        # Make an HMM instance and execute fit
        self.model = GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter)

    def trainHMM(self):
        # Run Gaussian HMM
        print("fitting to HMM and decoding ...", end="")
        self.model.fit(self.train)
        
        # Print trained parameters and plot
        print("Transition matrix")
        print(self.model.transmat_)
        print()

        print("Means and vars of each hidden state")
        for i in range(self.model.n_components):
            print("{0}th hidden state".format(i))
            print("mean = ", self.model.means_[i])
            print("var = ", np.diag(self.model.covars_[i]))
            print()

        # Predict the optimal sequence of internal hidden state
        self.hidden_states = self.model.predict(self.train)
        print("Hidden states of TRAIN sequence")
        print(self.hidden_states)
        print()

    def testHMM(self):
        # Predict the optimal sequence of internal hidden state
        self.test_hidden_states = self.model.predict(self.test)
        print("Hidden states of TEST sequence")
        print(self.test_hidden_states)
        print()

    def detectAnomaly(self, true_anomaly=[]):
        # detect anomaly by comparing state sequences of train and test data           
        self.estimated_anomaly = []
        for t in range(self.T):
            if self.hidden_states[t] == self.test_hidden_states[t]:
                self.estimated_anomaly.append(0)
            else:
                self.estimated_anomaly.append(1)
        print("Anomalies")
        print(self.estimated_anomaly)
        print()

        #################
        ## calculate anomaly detection performance (detection rate and precision)
        if not true_anomaly == []:
            self.true_anomaly = true_anomaly
            TP=0
            TN=0
            FP=0
            FN=0
            for t in range(self.T):
                if self.estimated_anomaly[t] == 1:
                    if self.true_anomaly[t] == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if self.true_anomaly[t] == 1:
                        FN += 1
                    else:
                        TN += 1
            # calculate stats
            recall = float(TP) / float(TP+FN)
            precision = float(TP) / float(TP+FP)
            accuracy = float(TP+TN) / float(TP+TN+FP+FN)

            # set dictionary
            self.stats = dict({'TP':TP,'TN':TN,'FP':FP,'FN':FN,'recall':recall,'precision':precision,'accuracy':accuracy})
            print("recall, precision")
            print(self.stats['recall'], self.stats['precision'])

    def drawGraph(self):
        time = np.array(range(self.T))

        fig, axs = plt.subplots(4, sharex=True, sharey=True)
        colours = cm.rainbow(np.linspace(0, 1, self.model.n_components))
        
        for i, colour in enumerate(colours):
        # Use fancy indexing to plot data in each state.
            mask = self.hidden_states == i
            axs[0].plot(time[mask],self.train[mask], ".-", c=colour)           
            axs[0].set_title("Train Data HiddenState")
            axs[0].grid(True)

        for i, colour in enumerate(colours):
        # Use fancy indexing to plot data in each state.
            mask = self.test_hidden_states == i
            axs[1].plot(time[mask],self.test[mask], ".-", c=colour)
            axs[1].set_title("Test Data HiddenState")
            axs[1].grid(True)

        axs[2].stem(self.estimated_anomaly)
        axs[2].set_title("Estimated Anomaly")
        axs[3].stem(self.true_anomaly)
        axs[3].set_title("True Anomaly")

        #plt.show()
        plt.savefig('%s/result.jpg'%self.OUTPATH)




def main():
    ## load data and anomalies
    data = np.loadtxt('data/data.csv', delimiter=',')
    anomaly = np.loadtxt('data/anomaly.csv', delimiter=',')
    
    ## set model
    hmmmodel = HMMModel()
    hmmmodel.addData(train = data[0,:][:,np.newaxis], test = data[1,:][:,np.newaxis])
    hmmmodel.setModel(10,'full',10000)
   
    sys.stdout = open("tmep.txt","w")

    ## infer HMM parameters and estimate HMM states
    hmmmodel.trainHMM()
    hmmmodel.testHMM()
    
    ## do anomaly detection
    hmmmodel.detectAnomaly(anomaly)

    sys.stdout.close()
    sys.stdout = sys.__stdout__

    ## drae eresults
    hmmmodel.drawGraph()


def main2D():
    ## load data and anomalies
    data = np.loadtxt('data/data2.csv', delimiter=',')
    anomaly = np.loadtxt('data/anomaly.csv', delimiter=',')
    
    ## set model
    hmmmodel = HMMModel()
    hmmmodel.addData(train = data[[0,1],:].T, test = data[[2,3],:].T)
    hmmmodel.setModel(5,'full',10000)
    hmmmodel.setPath('2Dtest')

    sys.stdout = open("%s/log.txt"%hmmmodel.OUTPATH,"w")

    ## infer HMM parameters and estimate HMM states
    hmmmodel.trainHMM()
    hmmmodel.testHMM()
    
    ## do anomaly detection
    hmmmodel.detectAnomaly(anomaly)

    sys.stdout.close()
    sys.stdout = sys.__stdout__

    ## drae eresults
    hmmmodel.drawGraph()





def modelSelection():
    ## load data and anomalies
    data = np.loadtxt('data/data.csv', delimiter=',')
    anomaly = np.loadtxt('data/anomaly.csv', delimiter=',')

    ns = [3,4,5,6,7,8,9,10]    
    
    for n in ns:
        directory = '%s_components'%(n)
        if not os.path.exists(directory):
            os.makedirs(directory)

        ## set model
        hmmmodel = HMMModel()
        hmmmodel.setPath(directory)
        hmmmodel.addData(train = data[0,:][:,np.newaxis], test = data[1,:][:,np.newaxis])
        hmmmodel.setModel(n,'full',10000)
   
        sys.stdout = open("%s/tmep.txt"%hmmmodel.OUTPATH,"w")

        ## infer HMM parameters and estimate HMM states
        hmmmodel.trainHMM()
        hmmmodel.testHMM()
    
        ## do anomaly detection
        hmmmodel.detectAnomaly(anomaly)

        sys.stdout.close()
        sys.stdout = sys.__stdout__

        ## drae eresults
        hmmmodel.drawGraph()

def modelSelection2D():
    ## load data and anomalies
    data = np.loadtxt('data/dataD3.csv', delimiter=',')
    anomaly = np.loadtxt('data/anomaly.csv', delimiter=',')

    ns = [3,4,5,6,7,8,9,10]    
    
    for n in ns:
        directory = '%s_components'%(n)
        if not os.path.exists(directory):
            os.makedirs(directory)

        ## set model
        hmmmodel = HMMModel()
        hmmmodel.setPath(directory)
        hmmmodel.addData(train = data[[0,1],:].T, test = data[[2,3],:].T)
        hmmmodel.setModel(n,'full',10000)
   
        sys.stdout = open("%s/tmep.txt"%hmmmodel.OUTPATH,"w")

        ## infer HMM parameters and estimate HMM states
        hmmmodel.trainHMM()
        hmmmodel.testHMM()
    
        ## do anomaly detection
        hmmmodel.detectAnomaly(anomaly)

        sys.stdout.close()
        sys.stdout = sys.__stdout__

        ## drae eresults
        hmmmodel.drawGraph()

if __name__=='__main__':

    #testYahoo()
    #main()
    #modelSelection()
    #main2D()
    modelSelection2D()