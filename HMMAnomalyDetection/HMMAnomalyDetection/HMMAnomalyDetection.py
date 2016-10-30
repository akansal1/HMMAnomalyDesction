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


def testYahoo():
    """
    Gaussian HMM of stock data
    --------------------------

    This script shows how to use Gaussian HMM on stock price data from
    Yahoo! finance. For more information on how to visualize stock prices
    with matplotlib, please refer to ``date_demo1.py`` of matplotlib.
    """
    ###############################################################################
    # Get quotes from Yahoo! finance
    quotes = quotes_historical_yahoo_ochl(
        "INTC", datetime.date(1995, 1, 1), datetime.date(2012, 1, 6))

    # Unpack quotes
    dates = np.array([q[0] for q in quotes], dtype=int)
    close_v = np.array([q[2] for q in quotes])
    volume = np.array([q[5] for q in quotes])[1:]

    # Take diff of close value. Note that this makes
    # ``len(diff) = len(close_t) - 1``, therefore, other quantities also
    # need to be shifted by 1.
    diff = np.diff(close_v)
    dates = dates[1:]
    close_v = close_v[1:]

    # Pack diff and volume for training.
    X = np.column_stack([diff, volume])

    ###############################################################################
    # Run Gaussian HMM
    print("fitting to HMM and decoding ...", end="")

    # Make an HMM instance and execute fit
    model = GaussianHMM(n_components=5, covariance_type="full", n_iter=1000).fit(X)

    # Predict the optimal sequence of internal hidden state
    hidden_states = model.predict(X)

    print("done")

    ###############################################################################
    # Print trained parameters and plot
    print("Transition matrix")
    print(model.transmat_)
    print()

    print("Means and vars of each hidden state")
    for i in range(model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means_[i])
        print("var = ", np.diag(model.covars_[i]))
        print()

    fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, model.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        # Use fancy indexing to plot data in each state.
        mask = hidden_states == i
        ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
        ax.set_title("{0}th hidden state".format(i))

        # Format the ticks.
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())

        ax.grid(True)

    plt.show()


def sampleFunc():
    ## load data and anomalies
    data = np.loadtxt('data/data.csv', delimiter=',')
    anomaly = np.loadtxt('data/anomaly.csv', delimiter=',')
    train = data[0,:][:,np.newaxis]
    test = data[1,:][:,np.newaxis]
    time = np.array(range(len(train)))

    ###############################################################################
    # Run Gaussian HMM
    print("fitting to HMM and decoding ...", end="")

    # Make an HMM instance and execute fit
    model = GaussianHMM(n_components=4, covariance_type="full", n_iter=1000).fit(train)

    # Predict the optimal sequence of internal hidden state
    hidden_states = model.predict(train)

    print("done")
    

    ###############################################################################
    # Print trained parameters and plot
    print("Transition matrix")
    print(model.transmat_)
    print()

    print("Means and vars of each hidden state")
    for i in range(model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means_[i])
        print("var = ", np.diag(model.covars_[i]))
        print()

    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    colours = cm.rainbow(np.linspace(0, 1, model.n_components))
    for i, colour in enumerate(colours):
        # Use fancy indexing to plot data in each state.
        mask = hidden_states == i
        ax.plot(time[mask],train[mask], ".-", c=colour)
        ax.set_title("{0}th hidden state".format(i))

        ax.grid(True)

    plt.show()


class HMMModel:
    def __init__(self):
        self.train = []
        self.test = []
        self.true_anomaly = []

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

        print("done")

    def testHMM(self):
        # Predict the optimal sequence of internal hidden state
        self.test_hidden_states = self.model.predict(self.test)

    def detectAnomaly(self, true_anomaly=[]):

        self.estimated_anomaly = []
        for t in range(self.T):
            if self.hidden_states[t] == self.test_hidden_states[t]:
                self.estimated_anomaly.append(0)
            else:
                self.estimated_anomaly.append(1)


        print(self.estimated_anomaly)


    def drawGraph(self):
        time = np.array(range(self.T))

        fig, axs = plt.subplots(3, sharex=True, sharey=True)
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

        axs[2].stem(time, self.estimated_anomaly)

        plt.show()




def main():
    ## load data and anomalies
    data = np.loadtxt('data/data.csv', delimiter=',')
    anomaly = np.loadtxt('data/anomaly.csv', delimiter=',')
    
    ## set model
    hmmmodel = HMMModel()
    hmmmodel.addData(train = data[0,:][:,np.newaxis], test = data[1,:][:,np.newaxis])
    hmmmodel.setModel(4,'full',1000)
    
    ## infer HMM parameters and estimate HMM states
    hmmmodel.trainHMM()
    hmmmodel.testHMM()
    
    ## do anomaly detection
    hmmmodel.detectAnomaly(anomaly)

    ## drae eresults
    hmmmodel.drawGraph()
    
    print(hmmmodel.hidden_states)
    print(hmmmodel.test_hidden_states)




if __name__=='__main__':

    #testYahoo()
    main()
    