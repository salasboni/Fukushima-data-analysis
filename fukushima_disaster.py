import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import scipy, scipy.stats
from sklearn.cluster import KMeans

def getData(original_filename, modified_filename):

    if os.path.isfile(modified_filename):

        df = pd.read_csv(modified_filename)

    else:
        reader = pd.read_csv(original_filename , iterator = True, chunksize = 10000)

        # Filter for all the readings between dates 03-01-2012 to 03-31-2012.
        # Set Captured time as index
        reader = [chunk[(chunk['Captured Time']>'2012-03-01') & (chunk['Captured Time']<'2012-03-31')] for chunk in reader]

        df = pd.concat(reader)

        # Filter for all the readings from within 100 km of the Fukushima Diaichi Nuclear plant
        df['Distance To Fukushima in Km'] = df.apply( distanceToF, axis = 1)
        df = df[df['Distance To Fukushima in Km'] <= 100]

        # Write to csv file
        df.to_csv(modified_filename)

    return df

def distanceToF(x):

    R = 6373.0 # Earth's radius in Km
    lat_Fukishima = np.radians(37.421003)
    lon_Fukishima = np.radians(141.033206)

    lat = np.radians(x['Latitude'])
    lon = np.radians(x['Longitude'])

    dist_lon = lon - lon_Fukishima
    dist_lat = lat - lat_Fukishima

    a = (np.sin( dist_lat / 2)) ** 2 + np.cos( lat) * np.cos( lat_Fukishima) * ( np.sin( dist_lon / 2)) ** 2
    c = 2 * np.arctan2( np.sqrt( a), np.sqrt( 1 - a))
    distance = R * c # result in kilometers
    return distance


def Fstat_pval(y,y_hat):
    dfn = 1
    dfd = len(y) - dfn - 1
    MSM = (1/dfn) * (y - y.mean()).sum()
    MSE = (1/dfd) * (y - y_hat).sum()
    F = MSM / MSE
    p = 1.0 - scipy.stats.f.cdf(F,dfn,dfd)
    return p


# Load data for large files
original_filename =  "/Users/" + os.getlogin() + "/Desktop/measurements.csv"
modified_filename =  "/Users/" + os.getlogin() + "/Desktop/measurements_modified.csv"

df = getData(original_filename, modified_filename)
df = df.set_index('Captured Time')

# Show head of data frame
print(df.head())

# Plot distance to Fukushima vs Radiation level
plt.figure()
df.plot( x = 'Distance To Fukushima in Km', y = 'Value', style = 'o', label='All loaders')
plt.xlabel( 'Distance To Fukushima in Km' )
plt.ylabel( 'Value in CPM' )
plt.axis( 'tight' )
namefig =  "/Users/" + os.getlogin() + '/Desktop/Distance_vs_Radiation.png'
plt.savefig(namefig, bbox_inches='tight')

# Box Cox transform on the data
df['Value transf'], lambd = stats.boxcox(df['Value'])

# Check that there is only one radioactivity measure unit
print(pd.unique(df.Unit.ravel()))

# Make plots, using the box-cox transformed data
plt.figure()
df.plot( x = 'Distance To Fukushima in Km', y = 'Value transf', style = 'o', label='All loaders')
plt.xlabel( 'Distance To Fukushima in Km' )
plt.ylabel( 'Value in CPM (box cox transform)' )
plt.axis( 'tight' )
namefig = '/Users/' + os.getlogin() + '/Desktop/Distance_vs_Radiation_Box_Cox.png'
plt.savefig(namefig, bbox_inches='tight')

# OLS fit for radiation level as a function of distance from the plant
X = df[ 'Distance To Fukushima in Km']
X = sm.add_constant(X)
y = df['Value transf']
model = sm.OLS( y , X)
result = model.fit()

# Print summary of OLS fit
print(result.summary())

# Plot regression line
y_hat = result.predict(X)
plt.plot(X, y_hat, 'r', alpha=0.9)

# Check which Loader IDs have values that don't fit the model well
alpha = 0.05
group_Loader_ID = df.groupby('Loader ID')
list_outliers_Loader_ID = []

# Loop over each group
for key , gp in group_Loader_ID:

    # Compute p value for how the model computed in total dataset fits this group
    gp = gp.sort('Distance To Fukushima in Km')
    X = gp['Distance To Fukushima in Km']
    X = sm.add_constant(X)
    y = gp['Value transf']
    y_hat = result.predict(X)
    p_value = Fstat_pval(y,y_hat)

    if p_value > alpha:

        list_outliers_Loader_ID.append(key)

        # Plot data for outlier Loader IDs to visually inspect our results
        plt.figure()
        gp.plot( x = 'Distance To Fukushima in Km', y = 'Value transf', style = 'o', label='Loader ID %s, p_val = %f' % (key,p_value))
        plt.xlabel('Distance To Fukushima in Km')
        plt.ylabel('Value in CPM')
        plt.axis('tight')
        namefig = '/Users/' + os.getlogin() + 'Desktop/Loader_ID_%s.png' % key
        plt.savefig(namefig, bbox_inches = 'tight')


print("This is the list of outlier Loader ID's:")
print(list_outliers_Loader_ID)

# Perform kmeans clustering
data_for_cluster = df[['Distance To Fukushima in Km', 'Value transf']]
data_for_cluster = np.asarray(data_for_cluster)
kmeans = KMeans(init = 'k-means++' , n_clusters = 10 , n_init = 10)
kmeans.fit(data_for_cluster)


# Plot the decision boundary of the clusters. Create a mesh
x_min, x_max = data_for_cluster[:, 0].min() , data_for_cluster[:, 0].max()
y_min, y_max = data_for_cluster[:, 1].min() , data_for_cluster[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 0.05))

# Get labels for points in mesh
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot mesh
plt.figure()
plt.imshow( Z, interpolation = 'nearest', extent = (x_min, x_max, y_min, y_max),
           cmap = plt.cm.Paired, aspect = 'auto', origin = 'lower')
plt.plot(data_for_cluster[:, 0], data_for_cluster[:, 1], 'k.', markersize = 2)

# Plot the centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x',
            linewidths = 5, color = 'y', zorder = 10)

plt.title('K-means clustering on the dataset')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
namefig = '/Users/' + os.getlogin() + '/Desktop/Region_clusters.png'
plt.savefig(namefig, bbox_inches = 'tight')


# Add kmeans labels as a column to data frame. Since labels are assigned randomly,
# we re-name the labels so that Region 0 is the closest one to the plant, followed by Region 1, etc.

df['Cluster Labels'] = kmeans.labels_
centroids_x_axis = centroids[:, 0]
sort_index = np.argsort(centroids_x_axis)
for label in range(0,10):
    df.loc[df['Cluster Labels']==label, 'Cluster Labels'] = 'Region %s' % sort_index[label]


# Plot a time series and histogram for each Region
for label in range(0,10):

    ts = df[ df['Cluster Labels'] == 'Region %s' %label]['Value']
    ts = ts.sort_index()

    plt.figure()
    ts.plot()
    pd.rolling_mean(ts, 100).plot(style = 'k--', linewidth = 5)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=30)
    plt.title('Time series for Region %s'%label)
    namefig = '/Users/' + os.getlogin() + 'Desktop/Region_%s_timeseries.png' % label
    plt.savefig(namefig, bbox_inches='tight')

    plt.figure()
    ts.hist()
    plt.title('Histogram for Region %s'%label)
    namefig = '/Users/' + os.getlogin() + 'Desktop/Region_%s_histogram.png' % label
    plt.savefig(namefig, bbox_in ches='tight')
