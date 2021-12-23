# [William Chak Lim Chan]
# [20198113]
# [GMMA]
# [Inaugural]
# [GMMA 869]
# [July 5, 2020]


# Answer to Question [1], Part [a]

# Import Packages

# Import packages we need for preprocessing, clustering, and performance metrics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
import itertools

from sklearn.cluster import KMeans, DBSCAN
import sklearn.metrics
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance, KElbowVisualizer
from kmodes.kmodes import KModes

# Read .csv Data
df = pd.read_csv('/Users/williamchan/Desktop/jewelry_customers.csv')

# Data Profiling

# Understand the data shape
list(df)
df.shape
df.info()
df.describe().transpose()
df.head(n=10)
df.tail()

# Understand the data - pull stats
pandas_profiling.ProfileReport(df)

# Data Scaling

# Copy unscaled data into new data frame for scaling
df_scaled = df.copy()
df_scaled.head(10)

# Scale the data
scaler = StandardScaler()
features = ['Age',
 'Income',
 'SpendingScore',
 'Savings',]
df_scaled[features] = scaler.fit_transform(df_scaled[features])

print(df_scaled)

# Answer to Question [1], Part [b]

# Determine Number of Clusters using Elbow Method
inertias = {}
silhouettes = {}
for k in range(2, 10):
    kmeans = KMeans(init='k-means++', n_init=10, n_clusters=k, max_iter=1000, random_state=28).fit(df_scaled)
    inertias[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    silhouettes[k] = silhouette_score(df_scaled, kmeans.labels_, metric='euclidean')

plt.figure();
plt.grid(True);
plt.plot(list(inertias.keys()), list(inertias.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Inertia");
plt.savefig('/Users/williamchan/Desktop/plots/mall-kmeans-elbow-interia.png');

plt.figure();
plt.grid(True);
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Silhouette");
plt.savefig('/Users/williamchan/Desktop/plots/mall-kmeans-elbow-silhouette.png');

model = KMeans(init='k-means++', n_init=10, max_iter=1000, random_state=28)
KElbowVisualizer(model, k=(2,10), metric='silhouette', timings=False).fit(df_scaled).poof();

# K-Means Method

# Run K-Means Algorithm
k_means = KMeans(init='k-means++', n_clusters=5, n_init=10, random_state=28)
k_means.fit(df_scaled)

# Print K-Means Cluster Numbers
k_means.labels_

# Print K-Means Cluster Centres
k_means.cluster_centers_

# Review K-Means Performance Metrics

# WCSS == Inertia
k_means.inertia_

# Silhouette Score
silhouette_score(df_scaled, k_means.labels_)

# Plot Clusters
plt.style.use('default');

plt.figure(figsize=(16, 10));
plt.grid(True);

sc = plt.scatter(df_unscaled['Age'], df_unscaled['Income'], s=500, c="black");
plt.title("K-Means (K=5)", fontsize=20);
plt.xlabel('Age', fontsize=22);
plt.ylabel('Total Spend', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);

# Plot Snake Plot
dat = df_scaled.copy()

dat['Cluster'] = k_means.labels_

datamart_melt = pd.melt(dat.reset_index(),
id_vars=['Cluster'],
value_vars=['Age', 'Income', 'SpendingScore', 'Savings'],
var_name='Feature',
value_name='Value')

plt.title('Snake Plot, K-Means, K=5')
sns.lineplot(x="Feature", y="Value", hue='Cluster', data=datamart_melt)
plt.savefig('/Users/williamchan/Desktop/plots/a1-jewl-kmeans-snake.png', transparent=False);

# Plot Relative Importance of Features
cluster_avg = dat.groupby(['Cluster']).mean()
population_avg = dat.drop(['Cluster'], axis=1).mean()

relative_imp = cluster_avg - population_avg

plt.figure(figsize=(15, 4));
plt.title('Relative Importance of Features');
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn');
plt.savefig('/Users/williamchan/Desktop/a1-jewl-kmeans-importance.png', transparent=False);

# Experiment with Lots of Ks

def do_kmeans(df_scaled, k):
    k_means = KMeans(init='k-means++', n_clusters=k, n_init=10, random_state=28)
    k_means.fit(df_scaled)
    wcss = k_means.inertia_
    sil = silhouette_score(df_scaled, k_means.labels_)
    
    plt.style.use('default');

    sample_silhouette_values = silhouette_samples(df_scaled, k_means.labels_)
    sizes = 200*sample_silhouette_values

    plt.figure(figsize=(16, 10));
    plt.grid(True);

    plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], s=sizes, c=k_means.labels_)
    plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], marker='x', s=500, c="black")

    plt.title("K-Means (K={}, WCSS={:.2f}, Sil={:.2f})".format(k, wcss, sil), fontsize=20);
    plt.xlabel('Annual Income (K)', fontsize=22);
    plt.ylabel('Spending Score', fontsize=22);
    plt.xticks(fontsize=18);
    plt.yticks(fontsize=18);
    plt.savefig('/Users/williamchan/Desktop/plots/mall-kmeans-auto-{}-silhouette-size.png'.format(k));
    plt.show()
    
    
    visualizer = SilhouetteVisualizer(k_means)
    visualizer.fit(df_scaled)
    visualizer.poof()
    fig = visualizer.ax.get_figure()
    fig.savefig('/Users/williamchan/Desktop/plots/a1-jewl-kmeans-auto-{}-silhouette-plot.png'.format(k), transparent=False);
    
    print("K={}, WCSS={:.2f}, Sil={:.2f}".format(k, wcss, sil))

for k in range(2, 10):
    do_kmeans(df_scaled, k)

# Answer to Question [1], Part [c]

# Unscale clusters to derive meaning
df_unscaled = scaler.inverse_transform(k_means.cluster_centers_)
df_unscaled = pd.DataFrame (data=df_unscaled, columns=['Age',
 'Income',
 'SpendingScore',
 'Savings',])
df_unscaled.head(5)
