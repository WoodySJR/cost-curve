
def cc_test(label,predict1,predict2=[],plotname=[],n_iter=1000,plot=False,alpha=0.05,precision=3):
    """
    This function computes a bootstrap confidence interval, either for a single model's AUCC,
    or for the difference between two models' AUCCs in order to draw comparison.
   
    Author: Song Junru from Central University of Finance and Economics, Beijing, China.
    
    Parameters
    ----------------
    label: 1-d array-like
      true class labels of the sample.
    predict1: 1-d array-like
      predicted possibilities of class 1 output by a classifier, on the same sample as 'label'.
    predict2: 1-d array-like
      predicted possibilities of class 1 output by another classifier, on the same sample.
      When predict 2 is specified, a bootstrap confidence interval of the difference between 
      two classifiers' AUCC is computed. When not specified, a bootstrap confidence interval
      of one single classifier's AUCC is computed. 
    plot: boolean
      whether or not to plot the histogram of the differences between two models or a single
      model's AUCC on bootstrap subsamples.
    plotname: str
      filename preferred for the histogram mentioned above.Note that the storage path and format
      can be informed in plotname.
    n_iter: int
      number of bootstrap subsamples preferred.takes a default value of 1,000.
    alpha: float between 0 and 1
      1-alpha is the confidence level for the bootstrap confidence interval.
    precision: int
      number of decimal places preferred to print the confidence interval
   
    Returns
    ----------------
    diffs: 1-d array-like of length n_iter
      differences between two models's AUCC or a single model's AUCC on bootstrap samples.
    lbound: float
      lower bound of the confidence interval
    ubound: float
      upper bound of the confidence interval 
    
    """
    indices = range(0,len(label))
    aucc_1 = []
    aucc_2 = []
    
    if len(predict2) == 0:
        aucc_2 = np.zeros(n_iter)
        for i in range(n_iter):
            sub_index = np.random.choice(indices,len(label))
            sub_label = np.array(label)[sub_index]
            sub_predict1 = predict1[sub_index]
            cc1 = cost_curve(predict = sub_predict1, label = sub_label)
            score_1 = cc1.score(see_result=False)
            aucc_1.append(score_1)
        
    else:
        for i in range(n_iter):
            sub_index = np.random.choice(indices,len(label))
            sub_label = np.array(label)[sub_index]
            sub_predict1 = predict1[sub_index]
            sub_predict2 = predict2[sub_index]
            cc1 = cost_curve(predict = sub_predict1, label = sub_label)
            cc2 = cost_curve(predict = sub_predict2, label = sub_label)
            score_1 = cc1.score(see_result=False)
            aucc_1.append(score_1)
            score_2 = cc2.score(see_result=False)
            aucc_2.append(score_2)
    
    diffs = np.array(aucc_1) - np.array(aucc_2)
    ubound = np.percentile(diffs,100*(1-alpha/2))
    lbound = np.percentile(diffs,100*(alpha/2))
    
    if plot:
        sns.distplot(diffs,color='r')
        plt.savefig(fname=plotname)
        plt.show()
    
    print(100*(1-alpha),'% confidence interval with bootstrapping: ',
          '[',round(lbound,precision),', ',round(ubound,precision),']')
    
    return(diffs,lbound,ubound)
