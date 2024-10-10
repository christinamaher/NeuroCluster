import numpy as np
import pandas as pd
from scipy.stats import  t
from joblib import Parallel, delayed
import statsmodels.api as sm 
from scipy.ndimage import label 

class TFR_Cluster_Test(object):
    '''
    Single-electrode neurophysiology object class to identify time-frequency resolved neural activity correlates of complex behavioral variables using non-parametric 
    cluster-based permutation testing.   

    Attributes
    ----------
    tfr_data       : (np.array) Single electrode tfr data matrix. Array of floats (n_epochs,n_freqs,n_times). 
    tfr_dims       : (tuple) Frequency and time dimensions of tfr_data. Tuple of integers (n_freq,n_times). 
    ch_name        : (str) Unique electrode identification label. String of characters.
    predictor_data : (pd.DataFrame) Regressors from task behavior with continuous, discreet, or categorical data. DataFrame of (rows=n_epochs,columns=n_regressors). 
    target_var     : (str) Column label for primary regressor of interest.
    alternative    : (str) Alternate hypothesis for t-test. Must be 'two-sided','greater', or 'less'. Default is 'two-sided'.
    alpha          : (float) Significance level - Threshold for allowable type 1 error rates. Default is 0.05.
      
    Methods
    ----------

    tfr_regression
    pixel_regression
    max_tfr_cluster
    compute_tcritical
    threshold_tfr_tstat
    compute_null_cluster_stats
    permuted_tfr_regression
    cluster_significance_test

    '''

    def __init__(self, tfr_data, predictor_data, target_var, ch_name, alternative='two-sided', alpha=0.05, **kwargs):

        '''
        Args:
        - tfr_data       : (np.array) Single electrode tfr data matrix. Array of floats (n_epochs,n_freqs,n_times). 
        - predictor_data : (pd.DataFrame) Task-based regressor data with dtypes continuous/discreet(int64/float) or categorical(pd.Categorical). DataFrame of (n_epochs,n_regressors).
        - target_var     : (str) Column label for primary regressor of interest. Array of 1d integers or floats (n_epochs,).
        - ch_name        : (str) Unique electrode identification label. String of characters.  
        - alternative    : (str) Alternate hypothesis for t-test. Must be 'two-sided','greater', or 'less'. Default is 'two-sided'.
        - alpha          : (float) Significance level - Threshold for allowable type 1 error rates. Default is 0.05.
        - **kwargs       : (optional) alternative, alpha, cluster_shape
        '''

        self.tfr_data        = tfr_data  # single electrode tfr data
        self.tfr_dims        = self.tfr_data.shape[1:] # time-frequency dims of electrode data (n_freqs x n_times)
        self.ch_name         = ch_name # channel name for single electrode tfr data
        self.predictor_data  = predictor_data # single subject behav data
        self.target_var      = target_var # variable to permute in regression model 
        self.ols_dmatrix     = pd.get_dummies(predictor_data,drop_first=True, dtype=float) # converts only categorical variables into one dummy coded vector
        self.target_var_idx  = np.where(self.ols_dmatrix.columns  == target_var)[0][0] # column index of regressor of interest in dummy coded dmatrix
        self.alternative     = alternative # Type of hypothesis test for t-distribution. Must be 'two-sided', 'greater', 'less'. Default is 'two-sided'.
        self.alpha           = alpha # Significance level 

    def tfr_regression(self):

        '''
        Performs univariate or multivariate OLS regression across tfr matrix for all pixel-level time-frequency power data and task-based predictor variables. Regressions are parallelized across pixels.

        Returns:
        - tfr_betas  : (np.array) Matrix of beta coefficients for predictor of interest for each pixel regression. Array of (n_freqs,n_times). 
        - tfr_tstats : (np.array) Matrix of t-statistics from coefficient estimates for predictor of interest for each pixel regression. Array of (n_freqs,n_times). 
        '''
        
        # run pixel permutations in parallel    
        expanded_results = Parallel(n_jobs=-1, verbose=5)(delayed(self.pixel_regression)(pixel_data)
                                                           for pixel_data in np.resize(self.tfr_data,(self.tfr_data.shape[0],np.prod(self.tfr_dims))).T) 
        
        tfr_betas,tfr_tstats = list(zip(*expanded_results))

        del expanded_results # improve speed, reduce memory load 

        return np.resize(np.array(tfr_betas),(self.tfr_data.shape[1],self.tfr_data.shape[2])), np.resize(np.array(tfr_tstats),
                                                                                                         (self.tfr_data.shape[1],self.tfr_data.shape[2]))

    def pixel_regression(self,pixel_data):
        
        '''        
        Fit pixel-wise univariate or multivariate OLS regression model and extract beta coefficient and t-statistic for predictor of interest (self.target_var). 

        Args:
        - pixel_data : (np.array) Array of power values for every epochs from single time-frequency pixel in tfr-data. Array of floats (num_epochs,)
        
        Returns:
        - pixel_beta : (np.array) Beta coefficient for predictor of interest from pixel-wise regression. Array of 1d float (1,)
        - pixel_tval : (np.array) Observed t-statistic for predictor of interest from pixel-wise regression. Array of 1d float (1,)
        '''
        
        # Fit pixel-wise regression model. If called in permuted_tfr_regressions, ols_dmatrix.target_var is permuted once per tfr, not for each pixel.
        pixel_model = sm.OLS(pixel_data,sm.add_constant(self.ols_dmatrix.to_numpy()),missing='drop').fit()
        
        # Return the estimated beta coefficient and tvalue for predictor of interest only (target_var)
        return (pixel_model.params[self.target_var_idx + 1],pixel_model.tvalues[self.target_var_idx + 1])

    def max_tfr_cluster(self,tfr_tstats,max_cluster_output='all',clust_struct=np.ones(shape=(3,3))):

        '''
        Identify time-frequency clusters of neural activity that are significantly correlated with the predictor of interest (self.target_var). Clusters are identified 
        from neighboring pixel regression t-statistics for the predictor of interest that exceed the tcritical threshold from the alternate hypothesis. 

        Args:
        - tfr_tstats         : (np.array) Pixel regression tstatistic from coefficient estimates for predictor of interest. Array of floats (n_freqs,n_times). 
        - max_cluster_output : (str) Output format for max cluster statistics. Must be 'all', 'cluster_stat', or 'freq_time'. Default is 'all'.
        - clust_struct       : (np.array) Binary matrix to specify cluster structure for scipy.ndimage.label. Array of (3,3). 
                                        Default is np.ones.shape(3,3), to allow diagonal cluster pixels (Not the scipy.ndimage.label default).
                                        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html

        Returns:
        - max_cluster_data : (list) Data for cluster with maximum statistic (sum of pixel tvals). List of dict(s). If alternative='two-sided':len=2,else: len=1. 
                                    If max_cluster_output = 'all', return dictionary of maximum cluster statistic ('cluster_stat' : sum of pixel t-statistics), 
                                    cluster frequency indices ('freq_idx':(freq_x,freq_y)), and cluster time indices ('time_idx':(time_x,time_y)). 
                                    If max_cluster_output = 'cluster_stat', return only [{cluster_stat}]. If max_cluster_output = 'freq_time', return only {freq_idx,time_idx}
                                    If max_cluster_output = 'expanded', return 'all' & 'all_clusters': 2D cluster ID matrix,'max_label': all_clusters label of max cluster
                                    ** If no clusters are found, max_cluster_data contains list of empty dictionaries
        '''
        
        max_cluster_data = []
        
        # Create binary matrix from tfr_tstats by thresholding pixel t-statistics by tcritical. (1 = pixel t-statistic exceeded tcritical threshold)
        for binary_mat in self.threshold_tfr_tstat(tfr_tstats):
            # test whether there are any pixels above tcritical threshold

            if np.sum(binary_mat) != 0:  
                
                # Find clusters of pixels with t-statistics exceeding tcritical
                cluster_label, num_clusters = label(binary_mat,clust_struct)
                del binary_mat # improve speed, reduce memory load
                # use argmax to find index of largest absolute value of cluster t statistic sums 
                max_label = np.argmax([np.abs(np.sum(tfr_tstats[cluster_label==i+1])) for i in range(num_clusters)])+1
                # use max_label index to compute cluster tstat sum (without absolute value)
                max_clust_stat = np.sum(tfr_tstats[cluster_label==max_label])
                # find 2D indices of minimum/maximum cluster frequencies and times 
                clust_freqs, clust_times = [(np.min(arr),np.max(arr)) for arr in np.where(cluster_label == max_label)]

                if max_cluster_output == 'all':
                    max_cluster_data.append({'cluster_stat':max_clust_stat,'freq_idx':clust_freqs,'time_idx':clust_times})
                elif max_cluster_output == 'cluster_stat':
                    max_cluster_data.append({'cluster_stat':max_clust_stat})
                elif max_cluster_output == 'freq_time':
                    max_cluster_data.append({'freq_idx':clust_freqs,'time_idx':clust_times})
                elif max_cluster_output == 'expanded':
                    max_cluster_data.append({'cluster_stat':max_clust_stat,'freq_idx':clust_freqs,'time_idx':clust_times,
                                            'all_clusters':cluster_label,'max_label':max_label})
            else: # if there is no cluster, return max_cluster_data with empty dictionaries
                
                if max_cluster_output == 'all':
                    max_cluster_data.append({'cluster_stat':0,'freq_idx':0,'time_idx':0})
                elif max_cluster_output == 'cluster_stat':
                    max_cluster_data.append({'cluster_stat':0})
                elif max_cluster_output == 'freq_time':
                    max_cluster_data.append({'freq_idx':0,'time_idx':0})
                elif max_cluster_output == 'expanded':
                    max_cluster_data.append({'cluster_stat':0,'freq_idx':0,'time_idx':0,'max_label':0,'all_clusters':0})            
        
        return max_cluster_data

    def compute_tcritical(self):

        '''
        Calculate critical t-values for regression model.

        Returns:
        - tcritical   : (float) Critical t-statistic for hypothesis test. Positive value when alternative = 'two-sided' or 'greater'. 
                                Negative when alternative = 'less'. 
        '''

        # Set number of tails for t-tests using 'alternative' parameter input string. 
            # tails = 2 if alternative = 'two-sided' (two tailed hypothesis test)
            # tails = 1 if alternative = 'greater' or 'less' (one tailed hypothesis test)
        tails = len(self.alternative.split('-')) 

        # Calculate degrees of freedom (N-k-1) 
        deg_free = float(len(self.ols_dmatrix)-len(self.ols_dmatrix.columns)-1) #### predictor data must only include regressors in columns

        # Return tcritical from t-distribution. Significance level is alpha/2 for two tailed hypothesis tests (alternative = 'two-sided').
        return (t.ppf(1-(self.alpha/tails),deg_free) if self.alternative != 'less' else np.negative(t.ppf(1-(self.alpha/tails),deg_free)))

    def threshold_tfr_tstat(self,tfr_tstats):

        '''
        Threshold tfr t-statistic matrix using tcritical.

        Args:
        - tfr_tstats  : (np.array) Matrix of t-statistics from pixel-wise regressions. Array of floats (n_freqs, n_times). 

        Returns:
        - binary_mat  : (np.array) Binary matrix results of pixel-wise t-tests. Pixel = 1 when tstatistic > tcritical, else pixel = 0. 
                                   List of array(s) (n_freqs, n_times).
        '''

        if self.alternative == 'two-sided': # return positive and negative t-critical for two-sided hypothesis test 
            return [(tfr_tstats>=self.compute_tcritical()).astype(int), 
                    (tfr_tstats<=np.negative(self.compute_tcritical())).astype(int)]

        elif self.alternative == 'greater': # return positive t-critical for one-sided hypothesis test 
            return [(tfr_tstats>=self.compute_tcritical()).astype(int)]

        elif self.alternative == 'less': # return negative t-critical for one-sided hypothesis test 
            return [(tfr_tstats<=self.compute_tcritical()).astype(int)] 
        else: 
            raise ValueError('Alternative hypothesis must be two-sided, greater, or less not {self.alternative}')

    def compute_null_cluster_stats(self, num_permutations,max_cluster_output='cluster_stat'):

        '''
        Compute null distribution (length = num_permutations) of maximum cluster statistics by running tfr regressions with permuted predictor of interest. 
        Note: only the predictor of interest (self.target_var) is permuted. We recommend only permuting the predictor of interest so your results are not
              confounded by covariates. 

        Args:
        - num_permutations   : (int) Number of permutation tests to perform. For every permutation iteration, predictor of interest is randomly permuted. 

        Returns:
        - null_cluster_stats : (list) List of maximum cluster statistics. If alternative = 'two-sided', returns two lists (positive & negative cluster stats).
                                    If alternative = 'greater' or 'less', returns one list with positive or negative cluster stats (length = num_permutations).  
        '''
        
        # run num_permutations repetitions of permuted_tfr_regression and using a generator expression to output a single generator object. 
        null_tstat_generator = (self.permuted_tfr_regression() for _ in range(num_permutations)) # generator saves memory and improves efficiency. 

        # Compute the maximum cluster statistics for each permutation by iterating through generator object and evaluating every permuted_tfr_regression. 
        permuted_cluster_stats = [self.max_tfr_cluster(tstat_gen,max_cluster_output) for tstat_gen in null_tstat_generator]

        # Return list of null maximum cluster statistics for each 
        if self.alternative == 'two-sided': 
            # Returns the positive and negative null distributions as nested lists. First element is always the positive null cluster stats. 
            return [[null_stat[0]['cluster_stat'] for null_stat in permuted_cluster_stats], [null_stat[1]['cluster_stat'] for null_stat in permuted_cluster_stats]]        

        elif (self.alternative == 'greater') | (self.alternative == 'less'):
            # Returns the null cluster stat distribution as a list. If alternative = 'greater', cluster stats are positive. If 'less', cluster stats are negative. 
            return [null_stat[0]['cluster_stat'] for null_stat in permuted_cluster_stats]

        else: 
            raise ValueError('Alternative hypothesis must be two-sided, greater, or less not {self.alternative}')


    def permuted_tfr_regression(self,n_jobs=-1,verbose=0):

        '''
        Run pixel-wise tfr regression with predictor data permuted with respect to predictor of interest (self.target_var). Covariates are not permuted.

        Args:
        - n_jobs  : (int) Number of CPUs used to run pixel-wise regressions in parallel. Default is -1 (use all available CPUs)
        - verbose : (int) Verbosity of parallelization function. Default is 0 (minimum progress messages printed). Minimal verbosity recommended for speed. 
        
        Returns:
        - permuted_tstats : (np.array) Matrix of null t-statistics from permuted pixel-wise regressions. Array of tfr_dims (n_freqs,n_times). 
        '''

        # Permute predictor data with respect to predictor of interest (target_var). Predictor data should only be permuted once for entire tfr (not for every pixel). 
        self.ols_dmatrix[self.target_var] = np.random.permutation(self.ols_dmatrix[self.target_var].values)

        # Run Parallelized pixel-wise regressions across 1d tfr_data expanded to shape (1,np.prod(self.tfr_dims)). 
        # permuted_results is a list of pixel-wise null betas and t-statistics with length = np.prod(self.tfr_dims) 
        permuted_results = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(self.pixel_regression)(pixel_data)
                                    for pixel_data in np.resize(self.tfr_data,(self.tfr_data.shape[0],np.prod(self.tfr_dims))).T) 
        
        # extract only permuted_tstats from all pixel_regression outputs
        _,permuted_tstats = list(zip(*permuted_results))
        
        del permuted_results # improve speed, reduce memory load
        
        # return permuted tstatistics as 2D array (shape = tfr_dims) to compute null tfr cluster statistics  
        return np.resize(np.array(permuted_tstats), (self.tfr_data.shape[1],self.tfr_data.shape[2]))

    def cluster_significance_test(self, max_cluster_data, null_distribution):

        '''
        Compute non-parametric pvalue of maximum tfr cluster statistic from distribution of null permutation distribution.

        Args:
        - max_cluster_data  : (list) Real tfr cluster data. If alternative = 'two-sided', len = 2. Else, length = 1. List of dict(s).
        - null_distribution : (list) Null cluster statistics from permutations. List of len=num_permutations. If alternative='two-sided', two nested lists
        
        Returns:
        - cluster_pvalue    : (list) P-value(s) for clusters in max_cluster_data (# null_stats > cluster_stat)/num permutations) List of float(s). 
        '''

        cluster_pvalue = []
        
        if self.alternative == 'two-sided':  

        # Iterate through real max cluster statistics info and null distribution simultaneously
            for cluster, null_stats in list(zip(max_cluster_data,null_distribution)): # cluster_stat and null_stats should have the same sign
                # check whether sign of cluster_stat and null_stats is the same
                if np.sign(cluster['cluster_stat']) == np.sign(null_stats[0]): 
                    # pvalue = (number of null stats more extreme than observed maximum clustre statistic)/(number of permutations)
                    pval = np.sum(np.abs(np.array(null_stats)) > np.abs(cluster['cluster_stat']))/len(null_stats)
                    cluster_pvalue.append(pval) # add pval to cluster_pvalue list
                else: # if the sign of the max cluster data is not the same as the corresponding null distribution, raise an error 
                    raise ValueError('Signs of max cluster stats and null distributions do not align') 
        
            return cluster_pvalue
        
        else: 
            # For one sided hypothesis tests, compute pvalue from inputs 
            pval = np.sum(np.abs(np.array(null_distribution)) > np.abs(max_cluster_data[0]['cluster_stat'])) / len(null_distribution)            
            
            # return as list to match two-sided test format
            return [pval]
