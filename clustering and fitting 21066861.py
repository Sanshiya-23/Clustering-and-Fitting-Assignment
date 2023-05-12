import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
from sklearn import cluster
import cluster_tools as ct
from scipy.optimize import curve_fit
import scipy.optimize as opt

df_gdp= pd.read_csv(r"C:\Users\SANSHIYA\Downloads\GDP-per-capita-in-the-uk-since-1270.csv")
print(df_gdp.describe())

corr = df_gdp.corr()
print(corr)

def map_corr(df_gdp, size=6): 
    corr = df_gdp.corr()
    plt.figure(figsize=(8, 8))
    # fig, ax = plt.subplots()
    plt.matshow(corr, cmap='coolwarm', origin="lower")
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.colorbar()
    # no plt.show() at the end
# call the map_corr() function with the DataFrame
map_corr(df_gdp)
# show the plot
plt.show()

pd.plotting.scatter_matrix(df_gdp, figsize=(12, 12), s=5, alpha=0.8)
plt.show()

# extract the two columns for clustering
df_ex = df_gdp[["Year", "Real GDP per capita "]]

# remove rows with NaN values
df_ex = df_ex.dropna()

# reset the index
df_ex = df_ex.reset_index(drop=True)
print(df_ex.head())

from sklearn.cluster import KMeans
import pandas as pd

def scaler(df):
    # normalize the dataframe to the 0-1 range
    df_norm = (df - df.min()) / (df.max() - df.min())
    # return the normalized dataframe along with the minimum and maximum values
    return df_norm, df.min(), df.max()

# read the dataframe from a CSV file
df_gdp = pd.read_csv(r"C:\Users\SANSHIYA\Downloads\GDP-per-capita-in-the-uk-since-1270.csv")

# extract the two columns for clustering and remove NaN entries
df_ex = df_gdp[["Year", "Real GDP per capita "]].dropna()

# normalize the data using the scaler function
df_norm, df_min, df_max = scaler(df_ex)

# loop over number of clusters
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = KMeans(n_clusters=ncluster)
    # fit the data, results are stored in the kmeans object
    kmeans.fit(df_norm)
    # get the cluster labels for the data points
    labels = kmeans.labels_

# print the results
print(df_gdp.head())
print(df_norm.head())
print(df_min)
print(df_max)


def backscale(arr, df_min, df_max):
        """ Expects an array of normalised cluster centres and scales
            it back. Returns numpy array.  """

        # convert to dataframe to enable pandas operations
        minima = df_min.to_numpy()
        maxima = df_max.to_numpy()

        # loop over the "columns" of the numpy array
        for i in range(len(minima)):
            arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

        return arr
    # extract the estimated cluster centres
cen = kmeans.cluster_centers_
    # calculate the silhouette score
print(ncluster, skmet.silhouette_score(df_ex, labels))
    


# extract the two columns for clustering
df_ex = df_gdp[["Year", "Real GDP per capita "]]
# remove NaN entries
df_ex = df_ex.dropna()
# normalize the data
df_norm, df_min, df_max = scaler(df_ex)
# loop over number of clusters
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = KMeans(n_clusters=ncluster)
    # fit the data, results are stored in the kmeans object
    kmeans.fit(df_norm) # fit done on x,y pairs
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    # calculate the silhouette score
    print(ncluster, skmet.silhouette_score(df_ex, labels))
    
ncluster = 7 # best number of clusters
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm) # fit done on x,y pairs
labels = kmeans.labels_
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_norm["Year"], df_norm["Real GDP per capita "], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("Year")
plt.ylabel("Real GDP per capita")
plt.show()

print(cen)
# Applying the backscale function to convert the cluster centre
scen = ct.backscale(cen, df_min, df_max)
print()
print(scen)
xcen = scen[:, 0]
ycen = scen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_ex["Year"], df_ex["Real GDP per capita "], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("Year")
plt.ylabel("Real GDP per capita ")
plt.show()

#fitting

df_gdp.plot("Year", "Real GDP per capita ")
plt.show()

def exponential(t, n0, g):

    t = t - 1990
    f = n0 * np.exp(g*t)
    return f

print(type(df_gdp["Year"].iloc[1]))
df_gdp["Year"] = pd.to_numeric(df_gdp["Year"])
print(type(df_gdp["Year"].iloc[1]))
param, covar = opt.curve_fit(exponential, df_gdp["Year"], df_gdp["Real GDP per capita "],p0 = (1.0e12, 0.03))
print("Real GDP per capita ", param[0]/1e9)
print("growth rate", param[1])

plt.figure()
plt.plot(df_gdp["Year"], exponential(df_gdp["Year"], 1.2e12, 0.03), label = "trail fit")
plt.plot(df_gdp["Year"], df_gdp["Real GDP per capita "])
plt.xlabel("Year")
plt.ylabel("Real GDP per capita ")
plt.legend()
plt.show()

df_gdp["fit"] = exponential(df_gdp["Year"], *param)
df_gdp.plot(x="Year", y=["Real GDP per capita ", "fit"])
plt.xlabel("Year")
plt.ylabel("UK GDP")
plt.title("UK GDP growth over time")
plt.legend(["Actual GDP", "Exponential fit"])
plt.show()

def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

param, covar = opt.curve_fit(logistic, df_gdp["Year"], df_gdp["Real GDP per capita "], p0=(1.2e12, 0.03, 1990.0), maxfev=2000)
sigma = np.sqrt(np.diag(covar))
df_gdp["fit"] = logistic(df_gdp["Year"], *param)
df_gdp.plot("Year", ["Real GDP per capita ", "fit"])
plt.show()
print("turning point", param[2], "+/-", sigma[2])
print("GDP at turning point", param[0]/1e9, "+/-", sigma[0]/1e9)
print("growth rate", param[1], "+/-", sigma[1])

df_gdp["trial"] = logistic(df_gdp["Year"], 3e12, 0.10, 1990)
df_gdp.plot("Year", ["Real GDP per capita ", "trial"])
plt.show()

year = np.arange(1930, 2015)
forecast = logistic(year, *param)
plt.figure()
plt.plot(df_gdp["Year"], df_gdp["Real GDP per capita "], label="Real GDP per capita ")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("Real GDP per capita ")
plt.legend()
plt.show()

# errors
def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper  

import errors as err

low, up = err.err_ranges(year, logistic, param, sigma)
plt.figure()
plt.plot(df_gdp["Year"], df_gdp["Real GDP per capita "], label="Real GDP per capita ")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("Real GDP per capita ")
plt.legend()
plt.show()