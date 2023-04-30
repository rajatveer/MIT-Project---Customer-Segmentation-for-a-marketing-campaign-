#!/usr/bin/env python
# coding: utf-8

# # **Customer Segmentation**

# # **Milestone 2**

# **Note:** This is in continuation to the data preprocessing we did in Milestone 1. Results might differ if you have followed different steps in data preprocessing. 

# ## Preparing Data for Segmentation

# ### Dropping columns that we will not use for segmentation

# The decision about which variables to use for clustering is a critically important decision that will have a big impact on the clustering solution. So we need to think carefully about the variables we will choose for clustering. Clearly, this is a step where a lot of contextual knowledge, creativity, and experimentation/iterations are needed.
# 
# Moreover, we often use only a few of the data attributes for segmentation (the segmentation attributes) and use some of the remaining ones (the profiling attributes) only to profile the clusters. For example, in market research and market segmentation, we can use behavioral data for segmentation (to segment the customers based on their behavior like amount spent, units bought, etc.), and then use both demographic as well as behavioral data for profiling the segments found.
# 
# Here, we will use the behavioral attributes for segmentation and drop the demographic attributes like Income, Age, and Family_Size. In addition to this, we need to drop some other columns which are mentioned below.
# 
# * `Dt_Customer`: We have created the `Engaged_in_days` variable using the Dt_Customer variable. Hence, we can drop this variable as it will not help with segmentation.
# * `Complain`: About 95% of the customers didn't complain and have the same value for this column. This variable will not have a major impact on segmentation. Hence, we can drop this variable. 
# * `day`:  We have created the `Engaged_in_days` variable using the 'day' variable. Hence, we can drop this variable as it will not help with segmentation.
# * `Status`: This column was created just to get the `Family_Size` variable that contains the information about the Status. Hence, we can drop this variable.
# * We also need to drop categorical variables like `Education` and `Marital_Status`, `Kids`, `Kidhome`, and `Teenhome` as distance-based algorithms cannot use the default distance like Euclidean to find the distance between categorical and numerical variables.
# * We can also drop categorical variables like `AcceptedCmp1`, `AcceptedCmp2`, `AcceptedCmp3`, `AcceptedCmp4`, `AcceptedCmp5`, and `Response` for which we have create the variable `TotalAcceptedCmp` which is the aggregate of all these variables.

# In[2]:


# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# To scale the data using z-score
from sklearn.preprocessing import StandardScaler

# To compute distances
from scipy.spatial.distance import cdist

# To perform K-means clustering and compute Silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# To visualize the elbow curve and Silhouette scores
# from yellowbrick.cluster import SilhouetteVisualizer

# Importing PCA
from sklearn.decomposition import PCA

# To encode the variable
from sklearn.preprocessing import LabelEncoder

# Importing TSNE
from sklearn.manifold import TSNE

# To perform hierarchical clustering, compute cophenetic correlation, and create dendrograms
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

# To compute distances
from scipy.spatial.distance import pdist

# To import K-Medoids
from sklearn_extra.cluster import KMedoids

# To import Gaussian Mixture
from sklearn.mixture import GaussianMixture

# To supress warnings
import warnings

# To import dbscan
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")


# In[3]:


data = pd.read_csv('unsuper_updated_data')
data.head()


# In[4]:


data.info()


# In[5]:


# Dropping all the irrelevant columns and storing in data_model
data_model = data.drop(
    columns=[
        "Year_Birth",
        "Dt_Customer",
        "day",
        "Complain",
        "Response",
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "Marital_Status",
        "Status",
        "Kids",
        'Education',
        'Kidhome',
        'Teenhome', 'Income','Age', 'Family_Size'
    ],
    axis=1,
)


# In[6]:


data_model.shape


# In[7]:


data_model.info()


# In[8]:


# Check first five rows of new data
data_model.head()


# **Let's plot the correlation plot after we've removed the irrelevant variables.**

# In[9]:


# Plot the correlation plot for new data
plt.figure(figsize=(12,12))
sns.heatmap(data_model.corr(), annot=True)


# **Observations and Insights:**

# In[10]:


# Library to split data
from sklearn.model_selection import train_test_split

# Import libraries for building linear regression model
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Importing libraries for scaling the data
from sklearn.preprocessing import MinMaxScaler

# To ignore warnings
import warnings
warnings.filterwarnings('ignore')


# ### Scaling the Data

# **What is feature scaling?**
# 
# Feature scaling is a class of statistical techniques that, as the name implies, scales the features of our data so that they all have a similar range. You'll understand better if we look at an example:
# 
# If you have multiple independent variables like Age, Income, and Amount related variables, with their range as (18–100 Years), (25K–75K), and (100–200), respectively, feature scaling would help them all to be in the same range.
# 
# **Why feature scaling is important in Unsupervised Learning?**
# 
# Feature scaling is especially relevant in machine learning models that compute some sort of distance metric as we do in most clustering algorithms, for example, K-Means. 
# 
# So, scaling should be done to avoid the problem of one feature dominating over others because the unsupervised learning algorithm uses distance to find the similarity between data points.

# **Let's scale the data**
# 
# **Standard Scaler**: StandardScaler standardizes a feature by subtracting the mean and then scaling to unit variance. Unit variance means dividing all the values by the standard deviation.
# 
# ![SC.png](attachment:SC.png)
# 
# 1. Data standardization is the process of rescaling the attributes so that they have a mean of 0 and a variance of 1.
# 2. The ultimate goal to perform standardization is to bring down all the features to a common scale without distorting the differences in the range of the values.
# 3. In sklearn.preprocessing.StandardScaler(), centering and scaling happen independently on each feature.

# In[11]:


# Applying standard scaler on new data
scaler = StandardScaler()                                                   # Initialize the Standard Scaler

df_scaled = scaler.fit_transform(data_model)                                # fit_transform the scaler function on new data

df_scaled = pd.DataFrame(df_scaled, columns=data_model.columns)      # Converting the embeddings to a dataframe

df_scaled.head()


# ## **Applying T-SNE and PCA to the data to visualize the data distributed in 2 dimensions**

# ### **Applying T-SNE**

# In[12]:


# Fitting T-SNE with number of components equal to 2 to visualize how data is distributed

tsne = TSNE(n_components=2, perplexity=35, random_state=1)        # Initializing T-SNE with number of component equal to 2, random_state=1, and perplexity=35

data_air_pol_tsne = tsne.fit_transform(data_model)                            # fit_transform T-SNE on new data

data_air_pol_tsne = pd.DataFrame(data_air_pol_tsne, columns=[0, 1])           # Converting the embeddings to a dataframe

plt.figure(figsize=(7, 7))                                                    # Scatter plot for two components

sns.scatterplot(x=0, y=1, data=data_air_pol_tsne)                             # Plotting T-SNE


# **Observation and Insights: There is no clear segregation as such shown in figure, we might be facing multicolinearity issue, there could be more outliers in the dataset. 

# ### **Applying PCA**

# **Think about it:**
# - Should we apply clustering algorithms on the current data or should we apply PCA on the data before applying clustering algorithms? How would this help?

# When the variables used in clustering are highly correlated, it causes multicollinearity, which affects the clustering method and results in poor cluster profiling (or biased toward a few variables). PCA can be used to reduce the multicollinearity between the variables. 

# In[13]:


n = data_model.shape[1]     
n


# In[14]:


# Defining the number of principal components to generate
n = data_model.shape[1]                                        # Storing the number of variables in the data

pca = PCA(n_components= n, random_state=1)                     # Initialize PCA with n_components = n and random_state=1

data_pca = pd.DataFrame(pca.fit_transform(df_scaled))         # fit_transform PCA on the scaled data

# The percentage of variance explained by each principal component is stored
exp_var = pca.explained_variance_ratio_                     


# **Let's plot the first two components and see how the data points are distributed.**

# In[15]:


# Scatter plot for two components using the dataframe data_pca
sns.scatterplot(x=data_pca[0] , y=data_pca[1])

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()


# **Let's apply clustering algorithms on the data generated after applying PCA**

# ## **K-Means** 

# In[16]:


distortions = []                                                  # Create an empty list

K = range(2, 10)                                                  # Setting the K range from 2 to 10

for k in K:
    kmeanModel = KMeans(n_clusters=k,random_state=4)              # Initialize K-Means
    kmeanModel.fit(data_pca)                                      # Fit K-Means on the data
    distortions.append(kmeanModel.inertia_)                       # Append distortion values to the empty list created above


# In[17]:


# Plotting the elbow plot
plt.figure(figsize=(16, 8))                                            # Setting the plot size

plt.plot(K, distortions, "bx-")                                        # Plotting the K on X-axis and distortions on y-axis

plt.xlabel("k")                                                        # Title of x-axis

plt.ylabel("Distortion")                                               # Title of y-axis

plt.title("The Elbow Method showing the optimal k")                    # Title of the plot
plt.show()


# **In the above plot, the elbow is seen for K=3 and K=5 as there is some drop in distortion at K=3 and K=5.**

# **Think About It:**
# 
# - How do we determine the optimal K value when the elbows are observed at 2 or more K values from the elbow curve?
# - Which metric can be used to determine the final K value?

# **We can use the silhouette score as a metric for different K values to make a better decision about picking the number of clusters(K).**

# ### **What is the silhouette score?**
# 
# Silhouette score is one of the methods for evaluating the quality of clusters created using clustering algorithms such as K-Means. The silhouette score is a measure of how similar an object is to its cluster (cohesion) compared to other clusters (separation). Silhouette score has a range of [-1, 1].
# 
# * Silhouette coefficients near +1 indicate that the clusters are dense and well separated, which is good.
# * Silhouette score near -1 indicates that those samples might have been assigned to the wrong cluster.

# **Finding silhouette score for each value of K**

# In[146]:


"""sil_score = []                                                             # Creating empty list
cluster_list = range(3, 7)                                                 # Creating a range from 3 to 7
for n_clusters in cluster_list:
    
    # Initialize K-Means with number of clusters equal to n_clusters and random_state=1
    clusterer = KMeans(n_clusters=cluster_list, random_state=1)
    
    # Fit and predict on the pca data
    preds = clusterer.fit(pca)
    
    # Calculate silhouette score - Hint: Use silhouette_score() function
    score = preds.silhouette_score()
    
    # Append silhouette score to empty list created above
    sil_score.append
    
    # Print the silhouette score
    print( "For n_clusters = {}, the silhouette score is {})".format(n_clusters, score)) """


# **From the above silhouette scores, 3 appears to be a good value of K. So, let's build K-Means using K=3.**

# ### **Applying K-Means on data_pca**

# In[18]:


kmeans = KMeans(n_clusters=3, random_state=1)         # Initialize the K-Means algorithm with 3 clusters and random_state=1

kmeans.fit_transform(data_pca)                        # Fitting on the data_pca


# In[19]:


data_pca["K_means_segments_3"] = kmeans.labels_                    # Adding K-Means cluster labels to the data_pca data

data["K_means_segments_3"] = kmeans.labels_                        # Adding K-Means cluster labels to the whole data

data_model["K_means_segments_3"] = kmeans.labels_                  # Adding K-Means cluster labels to data_model


# In[20]:


kmeans.labels_


# In[21]:


# Let's check the distribution
data_model["K_means_segments_3"].value_counts()


# **Let's visualize the clusters using PCA**

# In[22]:


# Function to visualize PCA data with clusters formed
def PCA_PLOT(X, Y, PCA, cluster):
    sns.scatterplot(x=X, y=1, data=PCA, hue=cluster)


# In[23]:


PCA_PLOT(0, 1, data_pca, "K_means_segments_3")


# **Observations and Insights:**

# In[24]:


data_model.info()


# ### **Cluster Profiling**

# In[25]:


# Taking the cluster-wise mean of all the variables. 
# Hint: First groupby 'data' by 'K_means_segments_3' and then find mean
cluster_profile_KMeans_3 = data.groupby(data['K_means_segments_3']).mean()
cluster_profile_KMeans_3.T


# In[26]:


# Highlighting the maximum average value among all the clusters for each of the variables
cluster_profile_KMeans_3.style.highlight_max(color="lightgreen", axis=0)


# **Observations and Insights:**
# Cluster 0 Includes the group of younger customers, with the families with most teenagers at home and are the ones that buy more, deals, and via web.
# 
# Cluster 1 Is the one with older customers, which have more kids at home and do more web visits. They also complaint more than the other clusters.
# 
# Cluster 2 Includes the customers with higher incomes, with larger recency and also are the ones that purchase the most amount of Products. They do this by purchasing via catalogs and in store and are the most receptive to respond to campaigns. They spend the largest by spending the most per purchase.

# **Let us create a boxplot for each of the variables**

# In[27]:


# Columns to use in boxplot
col_for_box = ['Income','Kidhome','Teenhome','Recency','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','Complain','Age','Family_Size','Expenses','NumTotalPurchases','Engaged_in_days','TotalAcceptedCmp','AmountPerPurchase']


# In[28]:


# Creating boxplot for each of the variables
all_col = col_for_box

plt.figure(figsize = (30, 50))

for i, variable in enumerate(all_col):
    plt.subplot(6, 4, i + 1)
    
    sns.boxplot(y=data[variable], x=data['K_means_segments_3'],showmeans=True)
    
    plt.tight_layout()
    
    plt.title(variable)

plt.show()


# **Think About It:**
# - Are the K-Means profiles with K=3 providing any deep insights into customer purchasing behavior or which channels they are using?
# - What is the next step to get more meaningful insights? 

# We can see from the above profiles that K=3 segments the customers into High, Medium and Low-income customers, and we are not getting deep insights into different types of customers. So, let's try to build K=5 (which has another elbow in the Elbow curve) and see if we can get better cluster profiles.

# In[29]:


data_pca.info()


# In[30]:


# Dropping labels we got from K=3 since we will be using PCA data for prediction
# Drop K_means_segments_3. 
data_pca.drop(columns=['K_means_segments_3'], inplace=True)
data_model.drop(columns=['K_means_segments_3'], inplace=True)
data.drop(columns=['K_means_segments_3'], inplace=True)


# In[31]:


data_pca.info()


# **Let's build K-Means using K=5**

# In[32]:


# Fit the K-Means algorithm using number of cluster as 5 and random_state=0 on data_pca
k_means = KMeans(n_clusters=5, random_state=0)
k_means.fit_transform(data_pca)


# In[33]:


# Add K-Means cluster labels to data_pca
data_pca['5_clusters']= k_means.labels_
# Add K-Means cluster labels to whole data
data['5_clusters']= k_means.labels_
# Add K-Means cluster labels to data_model
data_model['5_clusters']= k_means.labels_


# In[34]:


# Let's check the distribution
data_model['5_clusters'].value_counts()


# In[35]:


data_pca.info()


# **Let's visualize the clusters using PCA**

# In[36]:


# Hint: Use PCA_PLOT function created above
sns.scatterplot(x=0, y=1, data= data_pca, hue='5_clusters', palette='colorblind')

#PCA_PLOT(0,1,data_pca, '5_clusters')


# ### **Cluster Profiling**

# In[37]:


# Take the cluster-wise mean of all the variables. Hint: First groupby 'data' by cluster labels column and then find mean
cluster_profile_5 = data.groupby(data['5_clusters']).mean()
cluster_profile_5.T


# In[38]:


# Highlight the maximum average value among all the clusters for each of the variables
cluster_profile_5.style.highlight_max(color='yellow', axis=0)


# **Let's plot the boxplot**

# In[39]:


# Create boxplot for each of the variables
plt.figure(figsize=(30,50))
for i, variable in enumerate (col_for_box):
    plt.subplot(6, 4, i+1)
    sns.boxplot(x= data['5_clusters'], y= data[variable], showmeans= True)
    plt.tight_layout()
    plt.title(variable)
    
plt.show()


# ### **Characteristics of each cluster**

# Cluster 0 has the biggest family size and medium income. It consumes the least wine amount among the other medium and top income clusters. This cluster is a deals seeker.
# 
# Cluster 2 has medium income, medium family size, and medium spending. This cluster does the most web purchases,
# 
# Cluster 3 has the lowest income, medium family size and spends the least across all product types.
# 
# Cluster 4 and 1 represent the smallest size families and the top income earners, but cluster 4 includes a younger group.
# 
# Cluster 4 spends the most by total and item while cluster 1 is the second top spender. Their preferred purchased channel seem to be through catalog.
# 
# Cluster 4 also has the highest wine consumption average while cluster 1 the highest average consumption for fruits, but cluster 4 and 1 have highest consumption of meat. Cluster 4 consumes the most fish and sweets among other groups.

# In[40]:


# Dropping labels we got from K-Means since we will be using PCA data for prediction
# Hint: Use axis=1 and inplace=True
data_pca.drop(columns= ['5_clusters'], inplace= True)
data.drop(columns= ['5_clusters'], inplace= True)
data_model.drop(columns= ['5_clusters'], inplace= True)


# From the above profiles, K=5 provides more interesting insights about customer's purchasing behavior and preferred channels for purchasing products. We can also see that the High, Medium and Low income groups have different age groups and preferences, which was not evident in K=3. So, **we can choose K=5.**

# ## **K-Medoids**

# In[163]:


# To import K-Medoids
get_ipython().system(' pip install scikit-learn-extra')


# **Let's find the silhouette score for K=5 in K-Medoids**

# In[48]:


kmedo = KMedoids(n_clusters=5, random_state=1)  # Initializing K-Medoids with number of clusters as 5 and random_state=1
kmedo.fit_predict(data_pca)                     # fit using data_pca

score= silhouette_score(data_pca, kmedo.labels_)   # calculate the silhouette score
print(score)                                     # print the score


# In[54]:


# add kmedo cluster to data_pca, whole data and data_model
data['kmedo']= kmedo.labels_

data_model['kmedo']= kmedo.labels_

data_pca['kmedo']= kmedo.labels_


# In[53]:


# Let's check the distribution
data_model['kmedo'].value_counts()


# **Let's visualize the clusters using PCA**

# In[57]:


# Hint: Use PCA_PLOT function created above
plt.figure(figsize=(8,8))
sns.scatterplot(x=0, y=1, data=data_pca, hue='kmedo', palette='colorblind')


# ### **Cluster Profiling**

# In[58]:


# Take the cluster-wise mean of all the variables. Hint: First group 'data' by cluster labels column and then find mean
cluster_profile_kmedo = data.groupby(data['kmedo']).mean()
cluster_profile_kmedo


# In[60]:


# Highlight the maximum average value among all the clusters for each of the variables
cluster_profile_kmedo.style.highlight_max(color='yellow', axis=0)


# **Let's plot the boxplot**

# In[64]:


# Create boxplot for each of the variables
plt.figure(figsize=(30,50))
for i, variable in enumerate(col_for_box):
    plt.subplot(6,4, i+1)
    
    sns.boxplot(x=data['kmedo'], y=data[variable], showmeans= True)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# ### **Characteristics of each cluster**

# Cluster 0 has the most income and fewer family members. Clusters 0 and 1 have the highest income and are correlated all purchases, which does not provide many insights. Clusters 2,3 and 4 medium and lower income, and as their spending is relative similar and does not provide additional insights.

# In[65]:


# Dropping labels we got from K-Medoids since we will be using PCA data for prediction
# Hint: Use axis=1 and inplace=True
data_pca.drop(columns=['kmedo'], inplace=True)
data.drop(columns=['kmedo'], inplace=True)
data_model.drop(columns=['kmedo'], inplace=True)


# In[66]:


data_model.info()


# ## **Hierarchical Clustering**

# Let's find the Cophenetic correlation for different distances with different linkage methods.

# ### **What is a Cophenetic correlation?**
# 
# The cophenetic correlation coefficient is a correlation coefficient between the cophenetic distances(Dendrogramic distance) obtained from the tree, and the original distances used to construct the tree. It is a measure of how faithfully a dendrogram preserves the pairwise distances between the original unmodeled data points. 
# 
# The cophenetic distance between two observations is represented in a dendrogram by the height of the link at which those two observations are first joined. That height is the distance between the two subclusters that are merged by that link.
# 
# Cophenetic correlation is the way to compare two or more dendrograms. 

# **Let's calculate Cophenetic correlation for each of the distance metrics with each of the linkage methods**

# In[67]:


# list of distance metrics
distance_metrics = ["euclidean", "chebyshev", "mahalanobis", "cityblock"]

# list of linkage methods
linkage_methods = ["single", "complete", "average"]

high_cophenet_corr = 0                                                 # Creating a variable by assigning 0 to it
high_dm_lm = [0, 0]                                                    # Creating a list by assigning 0's to it

for dm in distance_metrics:
    for lm in linkage_methods:
        Z = linkage(data_pca, metric=dm, method=lm)                    # Applying different linkages with different distance on data_pca
        c, coph_dists = cophenet(Z, pdist(data_pca))                   # Calculating cophenetic correlation
        print(
            "Cophenetic correlation for {} distance and {} linkage is {}.".format(
                dm.capitalize(), lm, c
            )
        )
        if high_cophenet_corr < c:                                     # Checking if cophenetic correlation is higher than previous score
            high_cophenet_corr = c                                     # Appending to high_cophenet_corr list if it is higher
            high_dm_lm[0] = dm                                         # Appending its corresponding distance
            high_dm_lm[1] = lm                                         # Appending its corresponding method or linkage


# In[68]:


# Printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} distance and {} linkage.".format(
        high_cophenet_corr, high_dm_lm[0].capitalize(), high_dm_lm[1]
    )
)


# **Let's have a look at the dendrograms for different linkages with `Cityblock distance`**

# In[69]:


# List of linkage methods
linkage_methods = ["single", "complete", "average"]

# Lists to save results of cophenetic correlation calculation
compare_cols = ["Linkage", "Cophenetic Coefficient"]

# To create a subplot image
fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))            # Setting the plot size

# We will enumerate through the list of linkage methods above
# For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
for i, method in enumerate(linkage_methods):
    Z = linkage(data_pca, metric="Cityblock", method=method)                  # Measures the distances between two clusters

    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")           # Title of dendrogram

    coph_corr, coph_dist = cophenet(Z, pdist(data_pca))                       # Finding cophenetic correlation for different linkages with city block distance
    axs[i].annotate(
        f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
        (0.80, 0.80),
        xycoords="axes fraction",
    )


# **Observations and Insights: We don't see an optimal number of clusters are they are all in one big one

# **Think about it:**
# 
# - Can we clearly decide the number of clusters based on where to cut the dendrogram horizontally?
# - What is the next step in obtaining number of clusters based on the dendrogram?

# **Let's have a look at the dendrograms for different linkages with `Chebyshev distance`**

# In[70]:


# Plot the dendrogram for Chebyshev distance with linkages single, complete and average. 
# Hint: Use Chebyshev distance as the metric in the linkage() function 
# List of linkage methods
linkage_methods = ["single", "complete", "average"]

# Lists to save results of cophenetic correlation calculation
compare_cols = ["Linkage", "Cophenetic Coefficient"]

# To create a subplot image
fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))            # Setting the plot size

# We will enumerate through the list of linkage methods above
# For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
for i, method in enumerate(linkage_methods):
    Z = linkage(data_pca, metric="chebyshev", method=method)                  # Measures the distances between two clusters

    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")           # Title of dendrogram

    coph_corr, coph_dist = cophenet(Z, pdist(data_pca))                       # Finding cophenetic correlation for different linkages with city block distance
    axs[i].annotate(
        f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
        (0.80, 0.80),
        xycoords="axes fraction",
    )


# **Observations and Insights:** We can identify 3 clusters

# **Let's have a look at the dendrograms for different linkages with Mahalanobis distance**

# In[71]:


# Plot the dendrogram for Mahalanobis distance with linkages single, complete and average. 
# Hint: Use Mahalanobis distance as the metric in the linkage() function 
linkage_methods = ["single", "complete", "average"]

# Lists to save results of cophenetic correlation calculation
compare_cols = ["Linkage", "Cophenetic Coefficient"]

# To create a subplot image
fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))            # Setting the plot size

# We will enumerate through the list of linkage methods above
# For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
for i, method in enumerate(linkage_methods):
    Z = linkage(data_pca, metric="mahalanobis", method=method)                  # Measures the distances between two clusters

    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")           # Title of dendrogram

    coph_corr, coph_dist = cophenet(Z, pdist(data_pca))                       # Finding cophenetic correlation for different linkages with city block distance
    axs[i].annotate(
        f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
        (0.80, 0.80),
        xycoords="axes fraction",
    )


# **Observations and Insights:** The optimal number of cluster cannot be confirmed as there is a lot of overlap.

# **Let's have a look at the dendrograms for different linkages with Euclidean distance**

# In[72]:


# Plot the dendrogram for Euclidean distance with linkages single, complete, average and ward. 
# Hint: Use Euclidean distance as the metric in the linkage() function 
linkage_methods = ["single", "complete", "average"]

# Lists to save results of cophenetic correlation calculation
compare_cols = ["Linkage", "Cophenetic Coefficient"]

# To create a subplot image
fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))            # Setting the plot size

# We will enumerate through the list of linkage methods above
# For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
for i, method in enumerate(linkage_methods):
    Z = linkage(data_pca, metric="euclidean", method=method)                  # Measures the distances between two clusters

    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")           # Title of dendrogram

    coph_corr, coph_dist = cophenet(Z, pdist(data_pca))                       # Finding cophenetic correlation for different linkages with city block distance
    axs[i].annotate(
        f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
        (0.80, 0.80),
        xycoords="axes fraction",
    )


# **Think about it:**
# 
# - Are there any distinct clusters in any of the dendrograms?

# **Observations and Insights:**

# In[73]:


# Initialize Agglomerative Clustering with affinity (distance) as Euclidean, linkage as 'Ward' with clusters=3
HCmodel = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward',) 

# Fit on data_pca
HCmodel.fit(data_pca)                         


# In[75]:


# Add Agglomerative Clustering cluster labels to data_pca
data_pca['HCmodel'] = HCmodel.labels_

# Add Agglomerative Clustering cluster labels to the whole data
data['HCmodel'] = HCmodel.labels_

# Add Agglomerative Clustering cluster labels to data_model
data_model['HCmodel'] = HCmodel.labels_


# In[79]:


# Let's check the distribution
data['HCmodel'].value_counts()


# **Let's visualize the clusters using PCA.**

# In[82]:


# Hint: Use PCA_PLOT function created above
plt.figure(figsize=(8,8))
sns.scatterplot(x=0, y=1, data=data_pca, hue='HCmodel', palette='colorblind')


# ### **Cluster Profiling**

# In[85]:


# Take the cluster-wise mean of all the variables. Hint: First group 'data' by cluster labels column and then find mean
cluster_profile_hcmodel = data.groupby(data['HCmodel']).mean()
cluster_profile_hcmodel


# In[87]:


# Highlight the maximum average value among all the clusters for each of the variables
cluster_profile_hcmodel.style.highlight_max(color='yellow', axis=0)


# **Let's plot the boxplot**

# In[89]:


# Create boxplot for each of the variables
plt.figure(figsize=(40,50))
for i, variable in enumerate(col_for_box):
    plt.subplot(6,4,i+1)
    sns.boxplot(x=data['HCmodel'], y=data[variable])
    plt.tight_layout()
    plt.title(variable)
    
plt.show()


# ### **Characteristics of each cluster**

# This visualization doesn't provide enough insights to split into different types of customers.
# 
# Cluster 0:
# 
# Most income and smallest family size.
# 
# Cluster 1:
# 
# Outliers in most of the features.
# 
# Cluster 2:
# 
# Outliers in most of the features.

# **Observations and Insights:**

# In[98]:


# Dropping labels we got from Agglomerative Clustering since we will be using PCA data for prediction
# Hint: Use axis=1 and inplace=True
data_pca.drop(columns=['HCmodel'], inplace=True)
data.drop(columns=['HCmodel'], inplace=True)
data_model.drop(columns=['HCmodel'], inplace=True)


# ## **DBSCAN**

# DBSCAN is a very powerful algorithm for finding high-density clusters, but the problem is determining the best set of hyperparameters to use with it. It includes two hyperparameters, `eps`, and `min samples`.
# 
# Since it is an unsupervised algorithm, you have no control over it, unlike a supervised learning algorithm, which allows you to test your algorithm on a validation set. The approach we can follow is basically trying out a bunch of different combinations of values and finding the silhouette score for each of them.

# In[102]:


# Initializing lists
eps_value = [2,3]                       # Taking random eps value
min_sample_values = [6,20]              # Taking random min_sample value

# Creating a dictionary for each of the values in eps_value with min_sample_values
res = {eps_value[i]: min_sample_values for i in range(len(eps_value))}  


# In[103]:


# Finding the silhouette_score for each of the combinations
high_silhouette_avg = 0                                               # Assigning 0 to the high_silhouette_avg variable
high_i_j = [0, 0]                                                     # Assigning 0's to the high_i_j list
key = res.keys()                                                      # Assigning dictionary keys to a variable called key
for i in key:
    z = res[i]                                                        # Assigning dictionary values of each i to z
    for j in z:
        db = DBSCAN(eps=i, min_samples=j).fit(data_pca)               # Applying DBSCAN to each of the combination in dictionary
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        silhouette_avg = silhouette_score(data_pca, labels)           # Finding silhouette score 
        print( 
            "For eps value =" + str(i),
            "For min sample =" + str(j),
            "The average silhoutte_score is :",
            silhouette_avg,                                          # Printing the silhouette score for each of the combinations
        )
        if high_silhouette_avg < silhouette_avg:                     # If the silhouette score is greater than 0 or the previous score, it will get appended to the high_silhouette_avg list with its combination of i and j              
            high_i_j[0] = i
            high_i_j[1] = j


# In[104]:


# Printing the highest silhouette score
print("Highest_silhoutte_avg is {} for eps = {} and min sample = {}".format(high_silhouette_avg, high_i_j[0], high_i_j[1]))


# **Now, let's apply DBSCAN using the hyperparameter values we have received above.**

# In[114]:


# Apply DBSCAN using the above hyperparameter values
dbs = DBSCAN(eps=2, min_samples=30)
dbs.fit_predict(data_pca)


# In[115]:


# fit_predict on data_pca and add DBSCAN cluster labels to the data_pca
data_pca['dbscan']=dbs.labels_
# fit_predict on data_pca and add DBSCAN cluster labels to data_model
data_model['dbscan'] = dbs.labels_
# fit_predict on data_pca and add DBSCAN cluster labels to whole data
data['dbscan']= dbs.labels_


# In[116]:


# Let's check the distribution
data['dbscan'].value_counts()


# **Let's visualize the clusters using PCA.**

# In[117]:


# Hint: Use PCA_PLOT function created above
plt.figure(figsize=(8,8))
sns.scatterplot(x=0, y=1, data=data_pca, hue='dbscan', palette='colorblind')


# **Observations and Insights:** There are 2 clusters and no informaation can be seen from this clusters.

# **Think about it:**
# 
# - Changing the eps and min sample values will result in different DBSCAN results? Can we try more value for eps and min_sample?

# **Note:** You can experiment with different eps and min_sample values to see if DBSCAN produces good distribution and cluster profiles.

# In[121]:


# Dropping labels we got from DBSCAN since we will be using PCA data for prediction
# Hint: Use axis=1 and inplace=True
data_pca.drop(columns=['dbscan'], inplace=True)
data.drop(columns=['dbscan'], inplace=True)
data_model.drop(columns=['dbscan'], inplace=True)


# ## **Gaussian Mixture Model**

# **Let's find the silhouette score for K=5 in Gaussian Mixture**

# In[131]:


gmm = GaussianMixture(n_components=5, random_state=1) # Initialize Gaussian Mixture Model with number of clusters as 5 and random_state=1

preds = gmm.fit_predict(data_pca, preds)            # Fit and predict Gaussian Mixture Model using data_pca

score = silhouette_score(data_pca, preds)     # Calculate the silhouette score

print(score)                   # Print the score


# **Observations and Insights:**

# In[132]:


# Predicting on data_pca and add Gaussian Mixture Model cluster labels to the whole data
data['gaumixmod']= preds

# Predicting on data_pca and add Gaussian Mixture Model cluster labels to data_model
data_pca['gaumixmod']= preds

# Predicting on data_pca and add Gaussian Mixture Model cluster labels to data_pca
data_model['gaumixmod']= preds


# In[133]:


# Let's check the distribution
data['gaumixmod'].value_counts()


# **Let's visualize the clusters using PCA.**

# In[135]:


# Hint: Use PCA_PLOT function created above
plt.figure(figsize=(8,8))

sns.scatterplot(x=0, y=1, data=data_pca, hue='gaumixmod', palette='colorblind')


# ### **Cluster Profiling**

# In[137]:


# Take the cluster-wise mean of all the variables. Hint: First group 'data' by cluster labels column and then find mean
cluster_profile_gmm = data.groupby(data['gaumixmod']).mean()
cluster_profile_gmm


# In[139]:


# Highlight the maximum average value among all the clusters for each of the variables
cluster_profile_gmm.style.highlight_max(color='lightgreen', axis=0)


# **Let's plot the boxplot**

# In[144]:


# Create boxplot for each of the variables
plt.figure(figsize=(30,50))
for i, variable in enumerate (col_for_box):
    plt.subplot(6,4, i+1)
    sns.boxplot(x=data['gaumixmod'], y=data[variable])
    plt.tight_layout()
    plt.title(variable)
    
plt.show()


# ### **Characteristics of each cluster**

# Cluster 4 and 3 have outliers accross most features.
# 
# Cluster 2 covers a very large group of family sizes from 1 to 3; which does not help to identify multiple customer types.
# 
# Cluster 0 and 1 have outliers on income; which hinders tha ability to identify customer types by their purchasing power.

# ## **Conclusion and Recommendations**

# - **Refined Insights:** What are the most meaningful insights from the data relevant to the problem?
# 
# - **Comparison of various techniques and their relative performance:** How do different techniques perform? Which one is performing relatively better? Is there scope to improve the performance further?
# 
# - **Proposal for the final solution design:** What model do you propose to be adopted? Why is this the best solution to adopt?
