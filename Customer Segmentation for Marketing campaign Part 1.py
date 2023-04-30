#!/usr/bin/env python
# coding: utf-8

# ------------------------------
# ## **Data Dictionary**
# ------------------------------
# 
# The dataset contains the following features:
# 
# 1. ID: Unique ID of each customer
# 2. Year_Birth: Customer’s year of birth
# 3. Education: Customer's level of education
# 4. Marital_Status: Customer's marital status
# 5. Kidhome: Number of small children in customer's household
# 6. Teenhome: Number of teenagers in customer's household
# 7. Income: Customer's yearly household income in USD
# 8. Recency: Number of days since the last purchase
# 9. Dt_Customer: Date of customer's enrollment with the company
# 10. MntFishProducts: The amount spent on fish products in the last 2 years
# 11. MntMeatProducts: The amount spent on meat products in the last 2 years
# 12. MntFruits: The amount spent on fruits products in the last 2 years
# 13. MntSweetProducts: Amount spent on sweet products in the last 2 years
# 14. MntWines: The amount spent on wine products in the last 2 years
# 15. MntGoldProds: The amount spent on gold products in the last 2 years
# 16. NumDealsPurchases: Number of purchases made with discount
# 17. NumCatalogPurchases: Number of purchases made using a catalog (buying goods to be shipped through the mail)
# 18. NumStorePurchases: Number of purchases made directly in stores
# 19. NumWebPurchases: Number of purchases made through the company's website
# 20. NumWebVisitsMonth: Number of visits to the company's website in the last month
# 21. AcceptedCmp1: 1 if customer accepted the offer in the first campaign, 0 otherwise
# 22. AcceptedCmp2: 1 if customer accepted the offer in the second campaign, 0 otherwise
# 23. AcceptedCmp3: 1 if customer accepted the offer in the third campaign, 0 otherwise
# 24. AcceptedCmp4: 1 if customer accepted the offer in the fourth campaign, 0 otherwise
# 25. AcceptedCmp5: 1 if customer accepted the offer in the fifth campaign, 0 otherwise
# 26. Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# 27. Complain: 1 If the customer complained in the last 2 years, 0 otherwise
# 
# **Note:** You can assume that the data is collected in the year 2016.

# # **Milestone 1** 

# ### **Loading Libraries**

# In[143]:


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
# from yellowbrick.cluster import KElbowVisualizer

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
# from sklearn_extra.cluster import KMedoids

# To import Gaussian Mixture
# from sklearn.mixture import GaussianMixture

# To supress warnings
import warnings

warnings.filterwarnings("ignore")


# ### **Let us load the data**

# In[144]:


# loading the dataset
data = pd.read_csv("marketing_campaign+%284%29.csv")
data


# ### **Check the shape of the data**

# In[145]:


# Print the shape of the data
data.shape


# #### **Observations and Insights: Given dataset has 2240 rows in 27 columns and the columns are float, interger and object type

# ### **Understand the data by observing a few rows**

# In[146]:


# View first 5 rows
data.head()


# In[147]:


# View last 5 rows Hint: Use tail() methodd
data.tail()


# ### **Let us check the data types and and missing values of each column** 

# In[148]:


# Check the datatypes of each column. Hint: Use info() method
data.info()


# In[149]:


# Find the percentage of missing values in each column of the data
(data.isnull().sum()/len(data))*100


# #### **Observations and Insights: Except Income column we don't have any null column in the dataset with 1% data missing. We can impute with mean/median or mode. 

# We can observe that `ID` has no null values. Also the number of unique values are equal to the number of observations. So, `ID` looks like an index for the data entry and such a column would not be useful in providing any predictive power for our analysis. Hence, it can be dropped.

# **Dropping the ID column**

# In[150]:


# Remove ID column from data. Hint: Use inplace = True
data.drop(columns=['ID'], inplace=True)


# In[151]:


# check if it's successfully dropped
data.head()


# ## **Exploratory Data Analysis**

# ### **Let us now explore the summary statistics of numerical variables**

# In[152]:


# Explore basic summary statistics of numeric variables. Hint: Use describe() method.
data.describe().T


# In[153]:


pd.DataFrame(data['Income'].unique())


# *Observations and Insights: *
# 
# Year of birth range form 1893 to 1996. 
# 
# 75% income lies between 1.7k to 68k, there could be outliers in Income column 
# 
# There are people who spent more on wine, fish, meat product, fruits, gold then normal.
# 
# There are more number of people who accepted the offer in last 2 year than the people who did not.
# 
# Most people don't have kids at home as the average is 55.4%, It's more common to have a teenager at home.
# 
# The second campaing is the least accepted of all, with an average of 1%

# ### **Let us also explore the summary statistics of all categorical variables and the number of unique observations in each category**

# In[154]:


# List of the categorical columns in the data
cols = ["Education", "Marital_Status", "Kidhome", "Teenhome", "Complain"]


# **Number of unique observations in each category**

# In[155]:


for column in cols:
    print("Unique values in", column, "are :")
    print(data[column].value_counts())
    print("*" * 50)


# Observations and Insights:
# No one has more than 2 kids or teenagers at home There are some values of the marital_status that should be disregarded: Absurd and Yolo, also single should be merged with Alone, and 2nd Cycle should be merged with Master

# **Think About It:**
# 
# - We could observe from the summary statistics of categorical variables that the Education variable has 5 categories. Are all categories different from each other or can we combine some categories? Is 2n Cycle different from Master? 
# - Similarly, there are 8 categories in Marital_Status with some categories having very low count of less than 5. Can we combine these categories with other categories? 

# ### **Let us replace  the "2n Cycle" category with "Master" in Education and "Alone", "Absurd, and "YOLO" with "Single" in Marital_Status**

# In[156]:


# Replace the category "2n Cycle" with the category "Master"
# Hint: Use the replace() method and inplace=True

data["Education"].replace(to_replace=['2n Cycle'], value='Master', inplace=True)


# In[157]:


# Replace the categories "Alone", "Abusrd", "YOLO" with the category "Single"
# Hint: Use the replace() method and inplace=True

data["Marital_Status"].replace(to_replace=['Alone', 'Absurd', 'YOLO'], value='Single', inplace=True) 


# In[158]:


print(data[cols].value_counts())


# ## **Univariate Analysis**
# Univariate analysis is used to explore each variable in a data set, separately. It looks at the range of values, as well as the central tendency of the values. It can be done for both numerical and categorical variables.

# ## **1. Univariate Analysis - Numerical Data**
# Histograms help to visualize and describe numerical data. We can also use other plots like box plot to analyze the numerical columns.

# #### Let us plot histogram for the feature 'Income' to understand the distribution and outliers, if any.

# In[159]:


# Create histogram for the Income feature

plt.figure(figsize=(15, 7))
sns.histplot(x='Income', data=data)
plt.show()


# **We could observe some extreme value on the right side of the distribution of the 'Income' feature. Let's use a box plot as it is more suitable to identify extreme values in the data.** 

# In[160]:


# Plot the boxplot
sns.boxplot(data=data, x='Income', showmeans=True, color="violet")


# #### **Observations and Insights: Given column has some outliers in the dataset.
# 

# **Think About It**
# 
# - The histogram and the box plot are showing some extreme value on the right side of the distribution of the 'Income' feature. Can we consider them as outliers and remove or should we analyze these extreme values?

# In[161]:


# Calculating the upper whisker for the Income variable

Q1 = data.quantile(q=0.25)                          # Finding the first quartile

Q3 = data.quantile(q=0.75)                          # Finding the third quartile

IQR = Q3 - Q1                                       # Finding the Inter Quartile Range

upper_whisker = (Q3 + 1.5 * IQR)['Income']          # Calculating the Upper Whisker for the Income variable

print(upper_whisker)                                # Printing Upper Whisker


# In[162]:


# Let's check the observations with extreme value for the Income variable
data[data.Income > upper_whisker]


# **Think About It:**
# 
# - We observed that there are only a few rows with extreme values for the Income variable. Is that enough information to treat (or not to treat) them? Do we know at what percentile the upper whisker lies? 

# In[163]:


# Check the 99.5% percentile value for the Income variable
data.quantile(q=0.99)['Income']


# #### **Observations and Insights: We have 8 observations that are above 99 percentile meaning outliers which we can remove duirng further analysis.

# In[164]:


# Dropping observations identified as outliers 
# Pass the indices of the observations (separated by a comma) to drop them
data.drop(index=[164, 617, 655, 687, 1300, 1653, 2132, 2233], inplace=True)


# In[165]:


data.shape


# **Now, let's check the distribution of the Income variable after dropping outliers.**

# In[166]:


# Plot histogram and 'Income'
sns.histplot(x='Income', data=data)


# In[167]:


# Plot the histogram for 'MntWines'
sns.histplot(x='MntWines', data=data)


# In[168]:


sns.boxplot(x='MntWines', data=data, showmeans=True)


# In[169]:


Q1 = data.quantile(q=0.25)                          # Finding the first quartile

Q3 = data.quantile(q=0.75)                          # Finding the third quartile

IQR = Q3 - Q1                                       # Finding the Inter Quartile Range

upper_whisker = (Q3 + 1.5 * IQR)['MntWines']          # Calculating the Upper Whisker for the Income variable

print(upper_whisker)   


# In[170]:


df1 = data[data.MntWines > upper_whisker]
df1


# In[171]:


# let's find out 99 percentile of MntWines
percen_MntWines = data.quantile(q=0.99)['MntWines']
percen_MntWines


# In[172]:


(data['MntWines']>percen_MntWines).value_counts()


# In[173]:


data1= data.copy()


# In[174]:


data1[data1.MntWines>percen_MntWines]


# In[175]:


data1.drop(index=[111, 161, 497, 515, 523, 543, 559, 824, 870, 917, 987, 990, 1052, 1191, 1458, 1488, 1641, 1749, 1922, 1961, 2098, 2127], inplace=True)


# In[176]:


(data1['MntWines']>percen_MntWines).value_counts()


# In[177]:


plt.figure(figsize=(8,8))
sns.histplot(x=data1['MntWines'], data=data1)
plt.show()


# In[178]:


# get the lower whisker 
lower_whisker = (Q1 -1.5*IQR)['MntWines']
print(lower_whisker)

# calculate outliers
data1[data1.MntWines<lower_whisker]

# show the outliers


# ##### here we have 22 observations that are greater than 99 percentile.

# In[179]:


# Plot the histogram for 'MntFruits'
sns.histplot(x='MntFruits', data=data)


# In[180]:


sns.boxplot(x='MntFruits', data=data, showmeans=True)


# In[181]:


#get IQR
Q1 = data.quantile(q=0.25)
Q3= data.quantile(q=0.75)
IQR=Q3-Q1
upper_whisker_Fruits= (Q3 + 1.5*Q1)['MntFruits']
print(upper_whisker_Fruits)


# In[182]:


data[data.MntFruits > upper_whisker_Fruits]


# In[183]:


# Plot the histogram for 'MntMeatProducts' 
sns.histplot(x='MntMeatProducts', data=data)


# In[184]:


sns.boxplot(x='MntMeatProducts', data=data, showmeans= True)


# In[185]:


#IQR for MntMeatProducts
Q1= data.quantile(q=0.25)
Q3= data.quantile(q=0.75)
IQR = Q3 -Q1
upper_whisker_MeatProduct = (Q3 + 1.5 * Q1)['MntMeatProducts']
print(upper_whisker_MeatProduct)


# In[186]:


df2= data[data.MntMeatProducts>upper_whisker_MeatProduct]
df2


# In[187]:


# Plot the histogram for 'MntFishProduct'
sns.histplot(x='MntFishProducts', data=data)


# In[188]:


sns.boxplot(x='MntFishProducts', data=data, showmeans=True)


# In[189]:


# get IQR
Q1= data.quantile(q=0.25)
Q3= data.quantile(q=0.75)
IQR = Q3 -Q1
upper_whisker_FishProducts = (Q3 + 1.5 *Q1)['MntFishProducts']
print(upper_whisker_FishProducts)


# In[190]:


df3= data[data.MntFishProducts>upper_whisker_FishProducts]
df3


# In[191]:


# Plot the histogram for 'MntSweetProducts'
sns.histplot(x='MntSweetProducts', data=data)


# In[192]:


# Plot the histogram for 'MntGoldProducts'
sns.histplot(x='MntGoldProds', data=data)


# #### **Note:** Try plotting histogram for different numerical features and understand how the data looks like.

# #### **Observations and Insights for all the plots: we have 35 for wine, 508 for fruits, 509 for meats, 528 for Fish outliers, we will first keep these values into consideration and depending on result we will decide to remove them or not.

# ## **2. Univariate analysis - Categorical Data**

# Let us write a function that will help us create bar plots that indicate the percentage for each category. This function takes the categorical column as the input and returns the bar plot for the variable.

# In[193]:


def perc_on_bar(z):
    '''
    plot
    feature: categorical feature
    the function won't work if a column is passed in hue parameter
    '''

    total = len(data[z])                                          # Length of the column
    plt.figure(figsize=(15,5))
    ax = sns.countplot(data[z],palette='Paired',order = data[z].value_counts().index)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total) # Percentage of each class of the category
        x = p.get_x() + p.get_width() / 2 - 0.05                  # Width of the plot
        y = p.get_y() + p.get_height()                            # Height of the plot
        
        ax.annotate(percentage, (x, y), size = 12)                # Annotate the percentage 
    
    plt.show()                                                    # Show the plot


# #### Let us plot barplot for the variable Marital_Status.

# In[194]:


import warnings
warnings.filterwarnings('ignore')


# In[195]:


# Bar plot for 'Marital_Status'
perc_on_bar('Marital_Status')


# In[196]:


perc_on_bar('Education')


# In[197]:


perc_on_bar('Kidhome')


# In[198]:


perc_on_bar('Teenhome')


# In[199]:


perc_on_bar('Complain')


# #### **Note:** Explore for other categorical variables like Education, Kidhome, Teenhome, Complain.

# **Observations and Insights from all plots:**
# Most of the people are married or together.
# 
# 50 percent of the population has minimum of Graduate degree and around 47 percent population has wither master’s or PHD.
# 
# Majority of the customers don’t have any Teen-home or Kidhome.
# 
# Only 1 percent customer complaint are raised during last 2 years.

# ## **Bivariate Analysis**

# We have analyzed different categorical and numerical variables. Now, let's check how different variables are related to each other.

# ### **Correlation Heat map**
# Heat map can show a 2D correlation matrix between numerical features.

# In[200]:


plt.figure(figsize=(15, 7))                                                        # Setting the plot size
sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")  # Plotting the correlation plot
plt.show()


# Observations and Insights:
# The purchases os wines, meats, and number of catalogs store purchases are directly related to the higher income, while inversely proportional with having kids at home.
# 
# People with higher income don't tend to visit the web often.
# 
# The catalog purchases have big influence on the fish purchases.

# **The above correlation heatmap only shows the relationship between numerical variables. Let's check the relationship of numerical variables with categorical variables.**

# ### **Education Vs Income**

# In[201]:


sns.barplot(x='Education', y='Income', data=data)


# #### **Observations and Insights: Higher the degree higher is the income of the induvidual. Master and Graduate professional seems somewhat similar income, pHD being the highest.

# ### **Marital Status Vs Income**

# In[202]:


# Plot the bar plot for Marital_Status and Income
sns.barplot('Marital_Status', 'Income', data=data)


# #### **Observations and Insights: Houhold income of Widow category is the highest. Rest are pretty similar to one another.

# ### **Kidhome Vs Income**

# In[203]:


# Plot the bar plot for Kidhome and Income
sns.barplot('Kidhome', 'Income', data=data)


# #### **Observations and Insights: Number of kids grtealy affects the income of household.

# **We can also visualize the relationship between two categorical variables.**

# ### **Marital_Status Vs Kidhome**

# In[204]:


pd.crosstab(data['Marital_Status'], data['Kidhome'])


# In[205]:


# Plot the bar plot for Marital_Status and Kidhome
pd.crosstab(data['Marital_Status'], data['Kidhome']).plot(kind='bar',stacked=False)


# #### **Observations and Insights: Most of kids and teen at home found in married household followed by together and single respectively, only 18 kids can be found in widow category.  

# ## **Feature Engineering and Data Processing**
# 
# In this section, we will first prepare our dataset for analysis.
# - Creating new columns
# - Imputing missing values

# **Think About It:**
# 
# - The Year_Birth column in the current format might not be very useful in our analysis. The Year_Birth column contains the information about Day, Month, and year. Can we extract the age of each customer?
# - Are there other columns which can be used to create new features?

# In[206]:


data.head()


# ### **Age** 

# In[207]:


# Extract only the year from the Year_Birth variable and subtracting it from 2016 will give us the age of the customer at the time of data collection in 2016

data["Age"] = 2016 - pd.to_datetime(data['Year_Birth'], format="%Y").apply(lambda x: x.year) 

# Sorting the values in ascending order 
data["Age"].sort_values()                                         


# In[208]:


data.head()


# In[209]:


data[data['Age']>=115]


# #### **Observations and Insights: 3 observation of people older than 115 years might show error in data collection or extreme outliers

# **Think About It:**
# 
# - We could observe from the above output that there are customers with an age greater than 115. Can this be true or a data anomaly? Can we drop these observations?

# In[210]:


# Drop the observations with age > 115
# Hint: Use drop() method with inplace=True
data.drop(index=[192, 239, 339], inplace=True)


# In[211]:


# Plot histogram to check the distribution of age
sns.histplot(x='Age', data=data)


# In[212]:


data[data['Age']>=115]


# **Now, let's check the distribution of age in the data.**

# In[213]:


# Plot histogram to check the distribution of age
sns.histplot(x='Age', data=data)


# In[214]:


data["Age"].describe()


# #### **Observations and Insights: Age seems normallly distributed with mean age of 47.

# ### **Kids** 
# * Let's create feature "Kids" indicating the total kids and teens in the home.

# In[215]:


# Add Kidhome and Teenhome variables to create the new feature called "Kids"
data["Kids"] = data['Kidhome']+data['Teenhome']
data.head()


# ### **Family Size**
# * Let's create a new variable called 'Family Size' to find out how many members each family has.
# * For this, we need to have a look at the Marital_Status variable, and see what are the categories.

# In[216]:


# Check the unique categories in Marial_Status
data['Marital_Status'].unique()


# 
# 
# * We can combine the sub-categories Single, Divorced, Widow as "Single" and we can combine the sub-categories Married and Together as "Relationship" 
# * Then we can create a new variable called "Status" and assign values 1 and 2 to categories Single and Relationship, respectively.
# * Then, we can use the Kids (calculated above) and the Status column to find the family size.

# In[217]:


# Replace "Married" and "Together" with "Relationship"
data['Marital_Status'].replace(['Together', 'Married'], 'Relationship', inplace=True)


# In[218]:


# Replace "Divorced" and "Widow" with "Single"
data['Marital_Status'].replace(['Divorced', 'Widow'], 'Single', inplace=True)


# In[219]:


data['Marital_Status'].unique()


# In[220]:


# Create a new feature called "Status" by replacing "Single" with 1 and "Relationship" with 2 in Marital_Status
data["Status"] = data['Marital_Status'].replace({"Single": 1, "Relationship": 2})


# In[221]:


data['Status']


# In[222]:


# Add two variables Status and Kids to get the total number of persons in each family
data["Family_Size"] = data['Status'] + data['Kids']
data.head()


# ### **Expenses** 
# * Let's create a new feature called "Expenses", indicating the total amount spent by the customers in various products over the span of two years.

# In[223]:


data.info()


# In[224]:


# Create a new feature
# Add the amount spent on each of product 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
data["Expenses"] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']


# ### **Total Purchases**
# * Let's create a new feature called "NumTotalPurchases", indicating the total number of products purchased by the customers.

# In[225]:


# Create a new feature
# Add the number of purchases from each channel 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases'
data["NumTotalPurchases"] = data['NumDealsPurchases']+ data['NumWebPurchases']+ data['NumCatalogPurchases']+ data['NumStorePurchases']


# ### **Engaged in Days**
# * Let's create a new feature called "Engaged in days", indicating how long the customer has been with the company.

# In[226]:


data["Dt_Customer"].head()


# In[227]:


# Converting Dt_customer variable to Python date time object
data["Dt_Customer"] = pd.to_datetime(data['Dt_Customer'])


# **Let's check the max and min of the date.**

# In[228]:


# Check the minimum of the date
# Hint: Use the min() method
data['Dt_Customer'].min()


# In[229]:


# Check the maximum of the date
# Hint: Use the max() method
data['Dt_Customer'].max()


# **Think About It:**
# - From the above output from the max function, we observed that the last customer enrollment date is December 6th, 2014. Can we extract the number of days a customer has been with the company using some date as the threshold? Can January 1st, 2015 be that threshold?

# In[230]:


# Assigning date to the day variable
data["day"] = "01-01-2015"                         

# Converting the variable day to Python datetime object
data["day"] = pd.to_datetime(data.day)              


# In[231]:


data["Engaged_in_days"] = (data["day"] - data["Dt_Customer"]).dt.days     
data['Engaged_in_days']


# ### **TotalAcceptedCmp**
# * Let's create a new feature called "TotalAcceptedCmp" that shows how many offers customers have accepted.

# In[232]:


# Add all the campaign related variables to get the total number of accepted campaigns by a customer
# "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"
data["TotalAcceptedCmp"] = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + data['AcceptedCmp4'] + data['AcceptedCmp5'] + data['Response']


# In[233]:


data.head()


# ### **AmountPerPurchase**
# * Let's create a new feature called "AmountPerPurchase" indicating the amount spent per purchase.

# In[234]:


# Divide the "Expenses" by "NumTotalPurchases" to create the new feature AmountPerPurchase 
data['AmountPerPurchase'] = data['Expenses']/ data['NumTotalPurchases']
data['AmountPerPurchase'].head()


# **Now, let's check the maximum value of the AmountPerPurchase.**

# In[235]:


# Check the max value
# Hint: Use max() function
data['AmountPerPurchase'].max()


# In[236]:


data['AmountPerPurchase'].head()


# In[237]:


round(data['AmountPerPurchase'],1).max()


# **Think About It:**
# 
# - Is the maximum value in the above output valid? What could be the potential reason for such output?
# - How many such values are there? Can we drop such observations?

# In[238]:


# Find how many observations have NumTotalPurchases equal to 0
data[data['NumTotalPurchases'] == 0]


# In[239]:


data.drop(index=[981, 1524], inplace=True)


# In[240]:


data[data['NumTotalPurchases']==0]


# **Now, let's check the distribution of values in AmountPerPurchase column.**

# In[241]:


# Check the summary statistics of the AmountPerPurchase variable
data['AmountPerPurchase'].describe()


# In[242]:


# Plot the histogram for the AmountPerPurchas variable
plt.figure(figsize=(8,10))
sns.histplot(x=data['AmountPerPurchase'], data=data)


# #### **Observations and Insights: 
# Amount per purchase is significantly righ-skewed.
# The max value is 1,679 which is an outlier.

# ### **Imputing Missing Values**

# In[243]:


# Impute the missing values for the Income variable with the median
data['Income'].fillna(data['Income'].median(), inplace= True)


# In[244]:


data['Income'].isnull().sum()


# **Now that we are done with data preprocessing, let's visualize new features against the new income variable we have after imputing missing values.**

# ### **Income Vs Expenses**

# In[245]:


# Plot the scatter plot with Expenses on Y-axis and Income on X-axis  

plt.figure(figsize=(20, 10))                                    # Setting the plot size

sns.scatterplot(x='Income', y='Expenses', data=data)             

plt.xticks(fontsize=16)                                         # Font size of X-label

plt.yticks(fontsize=16)                                         # Font size of Y-label

plt.xlabel("Income", fontsize=20, labelpad=20)                  # Title of X-axis

plt.ylabel("Expenses", fontsize=20, labelpad=20)                # Title of Y-axis


# #### **Observations and Insights: Expenses and Income are directly proportional.

# ### **Family Size Vs Income**

# In[246]:


# Plot the bar plot for Family Size on X-axis and Income on Y-axis
plt.figure(figsize=(8,8))

sns.barplot(y='Income', x='Family_Size', data=data)


# In[247]:


data.to_csv('unsuper_updated_data', index=False)


# #### **Observations and Insights: As family size increases income deceases, single people earning more considering the fact that they can focus on work task as they don't have family issues.

# ## **Proposed approach**
# 
# - **Potential techniques -** What different techniques should be explored?
# - **Overall solution design -** What is the potential solution design?
# - **Measures of success -** What are the key measures of success?
