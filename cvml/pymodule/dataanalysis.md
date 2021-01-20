# 1. seaborn 数据可视化包

## 1.1. density plot AND Histogram 


A density plot is a smoothed, continuous version of a histogram estimated from the data.   


```py

sns.distplot(flights['arr_delay'], hist=True, kde=False, bins=int(180/5), color = 'blue',hist_kws={'edgecolor':'black'})
sns.distplot(flights['arr_delay'], hist=True, kde=True, bins=int(180/5), color = 'darkblue', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})

```

## 1.2. swarmplot 

Draw a categorical scatterplot with non-overlapping points.  
This function is similar to stripplot(), but the points are adjusted (only along the categorical axis) so that they don’t overlap.   
This gives a better representation of the `distribution of values` , but it does `not scale well to large numbers of observations` .   
This style of plot is sometimes called a “beeswarm”.  


