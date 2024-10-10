import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.text import OffsetFrom
from scipy.stats import  t

def plot_tcritical(cluster_test,figsize=(5.5,4.5),dpi=125,context='talk',tcrit_color='red'):

    """
    Plots the T-Distribution used to calculate t-critical for a given hypothesis test.  

    Args:
    - cluster_test : (NeuroCluster object): object containing the model specifications. 
    - figsize      : (tuple) size of the figure. Default is (8,4).
    - dpi          : (int) dots per inch for the plot. Default is 150.
    - context      : (str) seaborn context for the plot. Default is 'talk'.

    Returns:
    - None: displays a plot of the beta coefficients.

    """
    sns.set_context(context,rc={'axes.linewidth': 1.5})

    # Create t-distribution with N-K-1 degrees of freedom 
    df = int(float(len(cluster_test.ols_dmatrix)-len(cluster_test.ols_dmatrix.columns)-1))
    # x axis distribution (n=100)
    xs = np.linspace(-4,4,1000)
    # fitted t-distribution line
    y  = t.pdf(xs,df)
    # Calculate critical t-statistic (will be list len 2 if alternative = 'two-sided', else len=1)
    tcrit = cluster_test.compute_tcritical()
    
    # start plotting
    fig, ax = plt.subplots(1, 1, figsize=figsize,dpi=dpi,layout='constrained')

    # plot t distribution
    ax.plot(xs, y, 'k',linewidth=1.5)
    fig.suptitle(rf'   $T$-Distribution: $t_{{{{df={df}}}}}$')
    ax.tick_params(length=0)
    plt.ylabel('Probability Density')
    plt.xlabel(r'$t-values$')
    # Get all the lines used to draw the density curve (continuous values of t-distribution)
    lines = ax.get_lines()[-1]
    # x and y values of t-distribution line
    l_x, l_y = lines.get_data()
    
    # iterate through tcrit list and fill in tail(s) of t-distribution corresponding to significant t-values
    if cluster_test.alternative == 'two-sided':
        tcrit_posneg = [tcrit,np.negative(tcrit)]
        for tcrit_val in tcrit_posneg:
            # values of x that exceed t critical threshold (either positive or negative)
            if tcrit_val > 0:
                mask = l_x > tcrit_val
            else:
                mask = l_x < tcrit_val
        
            # x and y coordinates to fill in AUC for t-values exceeding t-critical
            filled_x, filled_y = l_x[mask], l_y[mask]
            # plot AUC
            ax.fill_between(filled_x, y1=filled_y,facecolor='red')

        # plt.title(rf'$H_{{{{a}}}}: \beta_{{{{t,f}}}}  \ne 0, t^*=\pm {np.round(tcrit,2)}$')
        ax.annotate(rf'$H_{{{{a}}}}:t_{{{{p}}}} \neq t^{{*}}$'+f'\n'+rf'$\alpha  = {cluster_test.alpha}$'+
            f'\n'+rf'$t^{{*}}\approx\pm{np.round(tcrit,2)}$',
            xy=(ax.get_xlim()[1],ax.get_ylim()[1]),xycoords='data',fontsize=12,
            xytext=(-100,-60),textcoords='offset points')

    elif cluster_test.alternative == 'greater':
        # x and y values of t-distribution line
        l_x, l_y = lines.get_data()
        # values of x that exceed t critical threshold (either positive or negative)
        mask = l_x > tcrit
        # x and y coordinates to fill in AUC for t-values exceeding t-critical
        filled_x, filled_y = l_x[mask], l_y[mask]
        # plot AUC
        ax.fill_between(filled_x, y1=filled_y,facecolor='red')
        ax.annotate(rf'$H_{{{{a}}}}:t_{{{{p}}}}>t^{{*}}$'+f'\n'+rf'$\alpha  = {cluster_test.alpha}$'+
                    f'\n'+rf'$t^{{*}}\approx{np.round(tcrit,2)}$',
                    xy=(ax.get_xlim()[1],ax.get_ylim()[1]),xycoords='data',fontsize=12,
                    xytext=(-100,-60),textcoords='offset points')
    else:
        # x and y values of t-distribution line
        l_x, l_y = lines.get_data()
        # values of x that exceed t critical threshold (either positive or negative)
        mask = l_x < tcrit
        # x and y coordinates to fill in AUC for t-values exceeding t-critical
        filled_x, filled_y = l_x[mask], l_y[mask]
        # plot AUC
        ax.fill_between(filled_x, y1=filled_y,facecolor='red')
        ax.annotate(rf'$H_{{{{a}}}}:t_{{{{p}}}}<t^{{*}}$'+f'\n'+rf'$\alpha  = {cluster_test.alpha}$'+
            f'\n'+rf'$t^{{*}}\approx{np.round(tcrit,2)}$',
            xy=(ax.get_xlim()[1],ax.get_ylim()[1]),xycoords='data',fontsize=12,
            xytext=(-100,-60),textcoords='offset points')
    
    sns.despine()
    plt.close(fig)
    return fig

def plot_beta_coef(betas, cluster_test,figsize=(7,5),dpi=125,context='talk',cmap='Spectral_r'):
    """
    Plots the beta coefficients for regressor of interest from a linear regression. 

    Args:
    - betas        : (np.array): 2D array of beta coefficients (frequency x time) from a linear regression.
    - cluster_test : (NeuroCluster object): object containing the model specifications. 
    - figsize      : (tuple) size of the figure. Default is (8,4).
    - dpi          : (int) dots per inch for the plot. Default is 150.
    - context      : (str) seaborn context for the plot. Default is 'talk'.
    - cmap         : (str) colormap for the plot. Default is 'Spectral_r'.

    Returns:
    - None: displays a plot of the beta coefficients.

    """
    sns.set_context(context,rc={'axes.linewidth': 1})
    fig = plt.figure(figsize=figsize,dpi = dpi,layout='constrained')
    plt.imshow(betas, interpolation = 'Bicubic', cmap=cmap, aspect='auto', origin='lower')  
    plt.tick_params(size=5,width=1)
    plt.suptitle(rf'TFR {cluster_test.ch_name} $\beta_{{{{target}}}}$ coefficients            ',fontsize=18)
    cbar = plt.colorbar()
    # cbar.set_label(r'$Beta $Coefficient')
    cbar.set_label(fr'$\beta_{{{cluster_test.target_var}}} coefficient$',fontsize=18)
    cbar.ax.tick_params(size=5,width=1,length=3)

    # make title dynamic depending on whether or not you are controlling for other variables
    if cluster_test.predictor_data.columns.tolist() == [cluster_test.target_var]:
        plt.title(rf'Target Variable: ${cluster_test.target_var}$',fontsize=12)
    else:
        beh_variables = cluster_test.predictor_data.columns.tolist().copy()
        control_variables = [var for var in beh_variables if var != cluster_test.target_var] 
        control_variables_str = ", ".join(control_variables)
        plt.title(rf'Target Variable: ${cluster_test.target_var}$, Covariate(s): ${control_variables_str}$',fontsize=12)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (ms)')
    plt.close(fig) 
    return fig
    

def plot_tstats(tstats, cluster_test, figsize=(7,5), dpi=125, context='talk', cmap='Spectral_r'):
    """
    Plots the t statistics for the regressor of interest.

    Args:
    - tstats       : (np.array) 2D array of t statistics (frequency x time) corresponding with the beta coefficients for a linear regression. 
    - cluster_test : (NeuroCluster object) object containing the model specifications.
    - figsize      : (tuple) size of the figure. Default is (8,4).
    - dpi          : (int) dots per inch for the plot. Default is 150.
    - context      : (str) seaborn context for the plot. Default is 'talk'.
    - cmap         :  (str) colormap for the plot. Default is 'Spectral_r'.

    Returns:
    - None: displays a plot of the t statistics.

    """
    
    sns.set_context(context,rc={'axes.linewidth': 1})
    fig = plt.figure(figsize=figsize,dpi=dpi,layout='constrained')
    plt.imshow(tstats, interpolation = 'Bicubic', cmap=cmap, aspect='auto', origin='lower')     
    plt.tick_params(size=5,width=1)
    plt.suptitle(rf'TFR {cluster_test.ch_name} $t$-statistics         ',fontsize=18)
    cbar = plt.colorbar()
    cbar.set_label(r'$t-statistic$',fontsize=18)
    cbar.ax.tick_params(size=5,width=1,length=3)
    # make title dynamic depending on whether or not you are controlling for other variables
    if cluster_test.predictor_data.columns.tolist() == [cluster_test.target_var]:
        plt.title(rf'Target Variable: ${cluster_test.target_var}$',fontsize=12)
        # plt.title(f'{cluster_test.ch_name} encoding \n{cluster_test.target_var}')
    else:
        beh_variables = cluster_test.predictor_data.columns.tolist().copy()
        control_variables = [var for var in beh_variables if var != cluster_test.target_var] 
        control_variables_str = ", ".join(control_variables)
        plt.title(rf'Target Variable: ${cluster_test.target_var}$, Covariate(s): ${control_variables_str}$',fontsize=12)

    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (ms)')
    plt.close(fig) 
    return fig

def plot_clusters(tstats,cluster_test,figsize=(9,4.5),dpi=125,context='talk'):
    """
    Plots clusters based on pixels with significant t-statistics for regressor of interest.

    Args:
    tstat_threshold : (list) List (either length 2 if alternative == 'two-sided' or length 1 if alternative == 'less' | 'greater') of binary matrices (frequency x time) for significant t-statistics.
    figsize         : (tuple) size of the figure. Default is (8,4).
    dpi             : (int) dots per inch for the plot. Default is 150.
    context         : (str) seaborn context for the plot. Default is 'talk'.

    Returns:
    - None: displays a plot of the significant t statistics.

    """
    
    sns.set_context(context,rc={'axes.linewidth': 1})

    if cluster_test.alternative == 'two-sided':
        # binary tstat matrices thresholded by tcritical value
        tstat_threshold = cluster_test.threshold_tfr_tstat(tstats)
        tcritical = cluster_test.compute_tcritical()

        fig, axs = plt.subplots(1,2,figsize=figsize,dpi=dpi,constrained_layout=True)
        fig.suptitle('Pixel-Wise Significance Tests')
        fig.supylabel('Frequency (Hz)',fontsize=20)
        fig.supxlabel('Time (ms)',fontsize=20)
        for i in range(len(tstat_threshold)):
            if i == 0:
                axs[i].imshow(tstat_threshold[i], interpolation = 'Bicubic',cmap='Reds', aspect='auto',origin='lower')
                axs[i].set_title(fr'$t_{{pixel}}>t_{{critical}}={np.round(tcritical,2)}$',fontsize=16)
                axs[i].tick_params(size=5,width=1)
            
            else:
                axs[i].imshow(tstat_threshold[i], interpolation = 'Bicubic',cmap='Blues', aspect='auto',origin='lower')
                axs[i].set_title(fr'$t_{{pixel}}<t_{{critical}}={np.negative(np.round(tcritical,2))}$',fontsize=16)
                axs[i].tick_params(size=5,width=1)

    elif cluster_test.alternative == 'greater':
        # binary tstat matrices thresholded by tcritical value
        tstat_threshold = cluster_test.threshold_tfr_tstat(tstats)
        tcritical = cluster_test.compute_tcritical()
    
        fig,ax = plt.subplots(1,1,figsize=figsize,dpi=dpi,constrained_layout=True)
        fig.suptitle('Pixel-Wise Significance Tests')
        fig.supylabel('Frequency (Hz)',fontsize=20)
        fig.supxlabel('Time (ms)',fontsize=20)
        ax.imshow(tstat_threshold[0], interpolation = 'Bicubic',cmap='Reds', aspect='auto',origin='lower')
        ax.set_title(fr'$t_{{pixel}}>t_{{critical}}={np.round(tcritical,2)}$',fontsize=16)
        ax.tick_params(size=5,width=1)
    
    elif cluster_test.alternative == 'less':
        # binary tstat matrices thresholded by tcritical value
        tstat_threshold = cluster_test.threshold_tfr_tstat(tstats)
        tcritical = cluster_test.compute_tcritical()
        fig,ax = plt.subplots(1,1,figsize=figsize,dpi=dpi,constrained_layout=True)
        fig.suptitle('Pixel-Wise Significance Tests')
        fig.supylabel('Frequency (Hz)',fontsize=20)
        fig.supxlabel('Time (ms)',fontsize=20)
        ax.imshow(tstat_threshold[0], interpolation = 'Bicubic',cmap='Blues', aspect='auto',origin='lower')
        ax.set_title(fr'$t_{{pixel}}<t_{{critical}}={np.negative(np.round(tcritical,2))}$',fontsize=16)
        ax.tick_params(size=5,width=1)

    plt.close(fig) 
    return fig


def plot_max_clusters(cluster_test,tstats,which_cluster='all',figsize=(9,4.5),dpi=125,context='talk'):
    """
    Plots significant clusters (positive and negative).

    Args:
    - tstats        : (np.array) 2D array of t statistics (frequency x time) corresponding with the beta coefficients for a linear regression.
    - which_cluster : (str) For two-sided tests, indicate which clusters to plot: 'all', 'positive', or 'negative'. Default is 'all'. 
    - figsize       : (tuple) size of the figure. Default is (8,4).
    - dpi           : (int) dots per inch for the plot. Default is 150.
    - context       : (str) seaborn context for the plot. Default is 'talk'.
    - sns_style     : (str)seaborn style for the plot. Default is 'white'.

    Returns:
    - None: displays a plot of the maximum cluster(s).

    """
    sns.set_context(context,rc={'axes.linewidth': 1})

    if cluster_test.alternative=='two-sided':
        # Compute the max cluster statistics with expanded output to get 2D cluster coordinates
        max_cluster_info = cluster_test.max_tfr_cluster(tstats,max_cluster_output='expanded')

        if which_cluster=='all':

            fig,axs = plt.subplots(1,len(max_cluster_info),figsize=figsize,dpi=dpi,constrained_layout=True)
            fig.suptitle('TFR-Level Threshold: Max Cluster Statistic')
            fig.supylabel('Frequency (Hz)',fontsize=20)
            fig.supxlabel('Time (ms)',fontsize=20)
            # Loop through the list of dictionaries
            for i,cluster in enumerate(max_cluster_info):

                # Extract max cluster pixel indices by finding 2D time-freq indices of max cluster 
                cluster_freqs,cluster_times = np.where(cluster['all_clusters']==cluster['max_label'])
                
                # Initialize an array the same shape as the tstat
                masked_tstat_plot = np.zeros_like(tstats)

                # Copy the values from tstat_plot to masked_tstat_plot for the significant cluster 
                masked_tstat_plot[cluster_freqs, cluster_times] = 1
                
                if cluster['cluster_stat'] > 0:
                    # Plot the masked tstat plot
                    axs[i].imshow(masked_tstat_plot, interpolation='bicubic', cmap='Reds', aspect='auto', origin='lower')
                    axs[i].text(0.95,0.95,('').join([r'$Max Cluster_{{statistic}}$',f'\n',
                        r'$\sum$ $t_{{pixel}} = $',f'{np.round(max_cluster_info[i]["cluster_stat"],2)}']),color='k',fontsize=11,
                        va='top',ha='right',transform=axs[i].transAxes)                
                    axs[i].set_title(r'Positive Cluster',fontsize=15)
                    axs[i].tick_params(size=5,width=1)

                elif cluster['cluster_stat'] < 0:
                    axs[i].imshow(masked_tstat_plot, interpolation='bicubic', cmap='Blues', aspect='auto', origin='lower')
                    axs[i].text(0.95,0.95,('').join([r'$Max Cluster_{{statistic}}$',f'\n',
                        r'$\sum$ $t_{{pixel}} = $',f'{np.round(max_cluster_info[i]["cluster_stat"],2)}']),color='k',fontsize=11,
                        va='top',ha='right',transform=axs[i].transAxes)
                    axs[i].set_title(r'Negative Cluster',fontsize=15)
                    axs[i].tick_params(size=5,width=1)

        elif which_cluster=='positive':
            cluster = max_cluster_info.copy()[0]

            # Extract max cluster pixel indices by finding 2D time-freq indices of max cluster 
            cluster_freqs,cluster_times = np.where(cluster['all_clusters']==cluster['max_label'])
            
            # Initialize an array the same shape as the tstat
            masked_tstat_plot = np.zeros_like(tstats)

            # Copy the values from tstat_plot to masked_tstat_plot for the significant cluster 
            masked_tstat_plot[cluster_freqs, cluster_times] = 1
            
            fig,ax = plt.subplots(1,1,figsize=figsize,dpi=dpi,constrained_layout=True)
            fig.suptitle('TFR-Level Threshold: Max Cluster Statistic')
            fig.supylabel('Frequency (Hz)',fontsize=20)
            fig.supxlabel('Time (ms)',fontsize=20)

            # Plot the masked tstat plot
            ax.imshow(masked_tstat_plot, interpolation='bicubic', cmap='Reds', aspect='auto', origin='lower')
            ax.text(0.95,0.95,('').join([r'$Max Cluster_{{statistic}}$',f'\n',
                r'$\sum$ $t_{{pixel}} = $',f'{np.round(cluster["cluster_stat"],2)}']),color='k',fontsize=11,
                va='top',ha='right',transform=ax.transAxes)                
            ax.set_title(r'Positive Cluster',fontsize=15)
            axs[i].tick_params(size=5,width=1)
        
        elif which_cluster=='negative':
            cluster = max_cluster_info.copy()[1]

            # Extract max cluster pixel indices by finding 2D time-freq indices of max cluster 
            cluster_freqs,cluster_times = np.where(cluster['all_clusters']==cluster['max_label'])
            
            # Initialize an array the same shape as the tstat
            masked_tstat_plot = np.zeros_like(tstats)

            # Copy the values from tstat_plot to masked_tstat_plot for the significant cluster 
            masked_tstat_plot[cluster_freqs, cluster_times] = 1
            
            # Plot the masked tstat plot
            fig,ax = plt.subplots(1,1,figsize=figsize,dpi=dpi,constrained_layout=True)
            fig.suptitle('TFR-Level Threshold: Max Cluster Statistic')
            fig.supylabel('Frequency (Hz)',fontsize=20)
            fig.supxlabel('Time (ms)',fontsize=20)

            ax.imshow(masked_tstat_plot, interpolation='bicubic', cmap='Blues', aspect='auto', origin='lower')
            ax.text(0.95,0.95,('').join([r'$Max Cluster_{{statistic}}$',f'\n',
                    r'$\sum$ $t_{{pixel}} = $',f'{np.round(cluster["cluster_stat"],2)}']),color='k',fontsize=11,
                    va='top',ha='right',transform=ax.transAxes)
            ax.set_title(r'Negative Cluster',fontsize=15)
            axs[i].tick_params(size=5,width=1)


    else:
        # Compute the max cluster statistics with expanded output to get 2D cluster coordinates
        max_cluster_info = cluster_test.max_tfr_cluster(tstats,max_cluster_output='expanded')
        
        # Initialize an array the same shape as the tstat
        masked_tstat_plot = np.zeros_like(tstats)

        # Extract max cluster pixel indices by finding 2D time-freq indices of max cluster 
        cluster_freqs,cluster_times = np.where(max_cluster_info[0]['all_clusters']==max_cluster_info[0]['max_label'])

        # Copy the values from tstat_plot to masked_tstat_plot for the significant cluster 
        masked_tstat_plot[cluster_freqs, cluster_times] = 1

        if max_cluster_info[0]['cluster_stat'] > 0:
            fig,ax = plt.subplots(1,1,figsize=figsize,dpi=dpi,constrained_layout=True)
            fig.suptitle('TFR-Level Threshold: Max Cluster Statistic')
            fig.supylabel('Frequency (Hz)',fontsize=20)
            fig.supxlabel('Time (ms)',fontsize=20)
            
            # Plot the masked tstat plot
            plt.imshow(masked_tstat_plot, interpolation='bicubic', cmap='Reds', aspect='auto', origin='lower')
            ax.text(0.95,0.95,('').join([r'$Max Cluster_{{statistic}}$',f'\n',
                    r'$\sum$ $t_{{pixel}} = $',f'{np.round(max_cluster_info[0]["cluster_stat"],2)}']),color='k',fontsize=11,
                    va='top',ha='right',transform=ax.transAxes)
            plt.title(r'Positive Cluster',fontsize=15)

        elif max_cluster_info[0]['cluster_stat'] < 0:
            fig,ax = plt.subplots(1,1,figsize=figsize,dpi=dpi,constrained_layout=True)
            fig.suptitle('TFR-Level Threshold: Max Cluster Statistic')
            fig.supylabel('Frequency (Hz)',fontsize=20)
            fig.supxlabel('Time (ms)',fontsize=20)
            plt.imshow(masked_tstat_plot, interpolation='bicubic', cmap='Blues', aspect='auto', origin='lower')
            ax.text(0.95,0.95,('').join([r'$Max Cluster_{{statistic}}$',f'\n',
                    r'$\sum$ $t_{{pixel}} = $',f'{np.round(max_cluster_info[0]["cluster_stat"],2)}']),color='k',fontsize=11,
                    va='top',ha='right',transform=ax.transAxes)
            plt.title(r'Negative Cluster',fontsize=15)

    plt.close(fig) 

    return fig


def plot_null_distribution(null_clusters, max_cluster_data, pvalue,figsize=(9,4.5),dpi=125,context='talk'):
    """
    Plots the null distribution of the cluster permutation test.

    Args:
    - null_clusters    : (np.array) 1D array of cluster statistics from the permutation test.
    - max_cluster_data : (list) List of dictionaries containing the significant cluster(s) statistics.
    - pvalue           : (float) p-value associated with the cluster permutation test.
    - figsize          : (tuple) size of the figure. Default is (12,4).
    - dpi              : (int) dots per inch for the plot. Default is 150.
    - context          : (str) seaborn context for the plot. Default is 'talk'.
    - sns_style        : (str) seaborn style for the plot. Default is 'white'.


    Returns:
    - None: displays a plot of the null distribution.

    """
    sns.set_context(context,rc={'axes.linewidth': 1.5})
    
    # initialize plots
    fig, axs = plt.subplots(1,len(max_cluster_data), figsize=figsize,dpi=dpi,layout='constrained')
    fig.suptitle('Null Cluster Distribution')
    
    for i, cluster in enumerate(max_cluster_data):
        # extract null data based on type of hypothesis test 
        l_fn = lambda x,i: x if len(x)>2 else x[i]
        null_data = l_fn(null_clusters,i)

        if cluster['cluster_stat'] > 0:
            if len(pvalue) == 1: 
                # plot null data
                plt.hist(null_data, bins=20, color='gray',edgecolor='black')
                plt.axvline(cluster['cluster_stat'], color='red', linestyle='dashed', linewidth=2)
                plt.tick_params(size=5)
                # 1d axes not scriptable
                axs.set_title(f'Positive Cluster\n'+r'$^\mathit{{pvalue \approxeq\ {}}}$'.format(np.round(pvalue[i],4)),
                             fontsize=16)
                axs.annotate(f'True Statistic',xy=(cluster['cluster_stat'],axs.get_ylim()[1]),xycoords='data',
                                xytext=(5,-10),color='red',
                                fontsize=10,textcoords='offset points')
            else:
                # plot null data
                axs[i].hist(null_data, bins=20, color='gray',edgecolor='black')
                axs[i].axvline(cluster['cluster_stat'], color='red', linestyle='dashed', linewidth=2)
                axs[i].tick_params(size=5)
                axs[i].set_title(f'Positive Cluster\n'+r'$^\mathit{{pvalue \approxeq\ {}}}$'.format(np.round(pvalue[i],4)),
                             fontsize=16)
                axs[i].annotate(f'True Statistic',xy=(cluster['cluster_stat'],axs[i].get_ylim()[1]),xycoords='data',
                                xytext=(5,-10),color='red',
                                fontsize=10,textcoords='offset points')
        else:
            if len(pvalue) == 1:
                # plot null data
                plt.hist(null_data, bins=20, color='gray',edgecolor='black')
                plt.axvline(cluster['cluster_stat'], color='red', linestyle='dashed', linewidth=2)
                plt.tick_params(size=5)
                # 1d axes not scriptable
                axs.set_title(f'Negative Cluster\n'+r'$^\mathit{{pvalue \approxeq\ {}}}$'.format(np.round(pvalue[i],4)),
                             fontsize=16)
                axs.annotate(f'True Statistic',xy=(cluster['cluster_stat'],axs.get_ylim()[1]),xycoords='data',
                                xytext=(-70,-10),color='red',
                                fontsize=10,textcoords='offset points')
            else:
                # plot null data
                axs[i].hist(null_data, bins=20, color='gray',edgecolor='black')
                axs[i].axvline(cluster['cluster_stat'], color='red', linestyle='dashed', linewidth=2)
                axs[i].tick_params(size=5)
                axs[i].set_title(f'Negative Cluster\n'+r'$^\mathit{{pvalue \approxeq \ {}}}$'.format(np.round(pvalue[i],4)),
                             fontsize=16)
                axs[i].annotate(f'True Statistic',xy=(cluster['cluster_stat'],axs[i].get_ylim()[1]),xycoords='data',
                                xytext=(-70,-10),color='red',
                                fontsize=10,textcoords='offset points')
    
    fig.supxlabel('Surrogate Test Statistics',fontsize=18)
    fig.supylabel('Count',fontsize=18)
    sns.despine()    
    plt.close(fig) 
    
    return fig


def plot_neurocluster_results(betas,cluster_test, max_cluster_data, null_clusters, tstats, tstat_threshold,cluster_pvalue):
    """
    Plots all the results from a NeuroCluster object.

    Args:
    - betas            : (np.array) 2D array of beta coefficients (frequency x time) from a linear regression.
    - cluster_test     : (NeuroCluster object) object containing the model specifications.
    - max_cluster_data : (list) List of dictionaries containing the significant cluster(s) statistics.
    - null_clusters    : (np.array) 1D array of cluster statistics from the permutation test.
    - tstats           : (np.array) 2D array of t statistics (frequency x time) corresponding with the beta coefficients for a linear regression.
    - tstat_threshold  : (list) List (either length 2 if alternative == 'two-sided' or length 1 if alternative == 'less' | 'greater') of binary matrices (frequency x time) for significant t-statistics.

    Returns:
    - None: displays beta coefficients, t statistics, significant clusters, maximum cluster(s), and null distribution plots.

    """
    tcrit_plot = plot_tcritical(cluster_test)
    beta_plot = plot_beta_coef(betas, cluster_test)
    tstat_plot = plot_tstats(tstats, cluster_test)
    cluster_plot = plot_clusters(tstats,cluster_test)
    max_cluster_plot= plot_max_clusters(cluster_test,tstats)
    null_distribution_plot = plot_null_distribution(null_clusters, max_cluster_data,cluster_pvalue)

    return tcrit_plot,beta_plot,tstat_plot,cluster_plot,max_cluster_plot,null_distribution_plot

