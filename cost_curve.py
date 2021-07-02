"""
=======================================
      Cost Curve - version 1.0
=======================================

Author: Song Junru from Central University of Finance and Economics, Beijing, China. 

Cost curve is a way of measuring the performance of predictive models, based on 
  expected costs of misclassification.
This module contains an implementation of cost curves, which includes -- 
(1) graphing of cost curves;
(2) computation of AUCC(area under the cost curve), or rather, total expected cost, 
   and its Bootstrap confidence interval;
(3) suggestion of optimal probability threshold, given misclassification costs 
   and prior probabilities;
(4) comparison between two models' AUCCs based on a Bootstrap confidence interval.

Note: This is a beta version and is still under continuous development. If you have any
    suggestion or better idea about the implementation of cost curves, please feel free 
    to contact me! This is my email: woodysjr@foxmail.com. :)
    
----------------
Reference:
  Drummond C, Holte R C. Cost curves: An improved method for visualizing classifier 
  performance[J]. Machine Learning, 2006, 65(1):95-130.
    
"""
class cost_curve(object):
    
    # initialization
    def __init__(self,predict,label,step=0.01):
        """
        Parameters
        ----------------
        predict: array-like
           The predictive probabilities of the positive class, which is 
            output by an arbitary classifier. A 1-d array is required.
        label:  array-like
           The true class labels of the same samples as in 'predict'.
        step:  float
           The computation of AUCC actually relies on an approximate method,
            which means the interval [0,1] is equally split into subintervals
            of length 'step'. step takes a default value of 0.01.
        """
        self.predict = predict
        self.label = label
        self.fpr, self.tpr, self.thresholds = roc_curve(self.label, self.predict) 
        self.step = step
        self.scores = None
        self.best_thresholds = None
        self.aucc = None
    
    
    # computation of AUCC
    def score(self,see_result = True):
        """
        Parameters
        ----------------
        see_result: boolean
            whether or nor to print the value of AUCC. 
            takes a default value of True. 
                
        Returns
        ----------------
        self.aucc: float
           the value of AUCC.
        
         """
        self.scores  = []
        self.best_thresholds = []
        self.aucc = 0
        for p in np.arange(0,1,self.step):
            costs = (1-self.tpr)*p + (self.fpr)*(1-p)
            min_score = np.min(costs)
            min_index = np.argmin(costs)
            min_threshold = self.thresholds[min_index]
            self.scores.append(min_score)
            self.best_thresholds.append(min_threshold)
            
            self.aucc += min_score*self.step
            
        if see_result:
            print('Area under the cost curve:',self.aucc)
        return(self.aucc)
    
    
    def graph(self,plotname):
        """
        graphing and storaging of cost curve plots.
        
        Parameters
        ----------------
        plotname: str
           the filename preferred for the cost curve plot.
           Note that the storage path and figure format can be
           informed in this parameter.
           For example: 'E:/working/my_cost_curve.svg'
              
        Returns
        ----------------
        self.scores: array-like, shape=[1/self.step]
            the minimum expected costs under different P(+)_cost values.
            P(+)_cost is the horizontal axis of the cost curve plot.
            For more details, please refer to the original paper by Drummond et al.
        """
        plt.figure(figsize=(8,8))
        plt.title('Cost curve',fontsize=20)
        plt.ylabel('Expected cost',fontsize=15)
        plt.xlabel(r'$P(+)_{cost}$',fontsize=15)
        
        for i in range(0,len(self.thresholds)):
            plt.plot([0,1],[self.fpr[i],1-self.tpr[i]],'lightgrey',alpha=0.5)
        
        plt.plot(np.arange(0,1,self.step),self.scores,'k')
        plt.axhline(y=0, xmin=0, xmax=1,color='k')
        plt.text(0.3,0.9,'Area under the cost curve: %0.4f' % self.aucc,fontsize=15)
        
        plt.savefig(fname=plotname)
        plt.show()
        return(self.scores)
    
    
    # cost10 is the cost of misclassifying 1 to 0, cost01 vice versa
    # prior is the prior probability of class 1
    # given cost10, cost01 and prior, the best threshold  and the corresponding predictive performance
    #  are output in a dataframe. 
    def threshold(self,cost10,cost01,prior):
        """
        suggests the optimal probability threshold given specific misclassification costs and
        prior probabilities, and also returns the corresponding predictive preformance.
        
        Parameters
        ----------------
        cost10: float
          the cost of misclassifying class 1 to 0. 
        cost01: float
          the cost of misclassifying class 0 to 1.
        prior: float
          the prior probability of class 1.
          
        Returns
        ----------------
        result.pd: dataframe
          which includes the following items: optimal threshold, minimum expected cost,
          normalized minimum expected cost, sensitivity, specificity, accuracy.
        """
        
        p = (prior*cost10)/(prior*cost10+(1-prior)*cost01)
        costs = (1-self.tpr)*p + (self.fpr)*(1-p)
        min_score = np.min(costs)
        min_index = np.argmin(costs)
        min_threshold = self.thresholds[min_index]
        FPR = self.fpr[min_index]
        TPR = self.tpr[min_index]
        
        pred_class = np.zeros(len(self.label))
        pred_class[self.predict>min_threshold]=1
        accuracy = np.mean(self.label==pred_class)
        
        result = {'$P(+)_{cost}$':prior,'Best threshold':min_threshold,
                 'Expected cost':(1-TPR)*prior*cost10 + FPR*(1-prior)*cost01,
                 'Normalized expected cost':min_score,
                 'Sensitivity':TPR,'Specificity':1-FPR,
                  'Accuracy': accuracy}
        
        result_pd = pd.DataFrame.from_dict(result,orient='index',columns=[' '])
        
        return(result_pd)
