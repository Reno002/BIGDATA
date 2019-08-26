#!/usr/bin/env python
# coding: utf-8

# # Final Project

# <i>Disusun Oleh: Reno Penggalih Surya Wardhani </i>

# <b>Analisis pada Data Harga Rumah di USA</b><br>Pada Final Project ini, data yang digunakan adalah data harga rumah di USA yang terdiri dari 5000 data dengan 7 feature/variabel. Tujuh feature tersebut, antara lain:
# * Average Area Income (X1)
# * Average Area House Age (X2)
# * Average Area Number of Rooms (X3)
# * Average Area Number of Bedrooms (X4)
# * Area Population (X5)
# * Address (X6)
# * Price (Y)<br>
# 
# <b>USA Housing, sumber : www.kaggle.com .
# 

# In[3]:


# import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[108]:


# import dataset dari S3 bucket Amazon Web Service

import s3fs
import pandas as pd

s3=s3fs.S3FileSystem()

with s3.open('dts-bda-p3/USA_Housing.csv', 'rb') as f:
    data=pd.read_csv(f)
    
data.head()


# In[5]:


# pre-processing , mengecek ada atau tidaknya data null per feature
# data tidak memiliki nilai null sehingga dapat dilanjutkan ke langkah selanjutnya

def num_missing(x):
  return sum(x.isnull())
print ("Missing values per column:")
print (data.apply(num_missing, axis=0))


# <b>1. Scatter Plot untuk masing masing feature adalah sebagai berikut. Digunakan untuk melihat pola persebaran data antar feature.

# In[6]:


sns.pairplot(data)


# In[7]:


plt.figure(figsize=(10,7))
sns.heatmap(data.corr(),annot=True,linewidths=2)


# <b>2. Dari Scatter Plot dapat dilihat bahwa antar feature memiliki pola persebaran data yang mengelompok, sedangkan feature Avg. Area Income dan Price memiliki pola persebaran data yang mengikuti garis lurus (linear). Sehingga diduga feature Avg. Area Income dan Price memiliki korelasi yang kuat. Dapat dilihat pula pada figure heatmap, nilai korelasi feature Avg. Area Income dan Price merupakan nilai korelasi yang tertinggi, yaitu sebesar sebesar 0,64 .

# <b>3. Statistik Deskriptif dari feature. Digunakan untuk melihat ukuran persebaran datanya, yaitu mean, standar deviasi, Quantil 1, Quantil 2, Quantil 3, minimum, maksimum. Juga dapat dilihat beberapa grafik datanya: scatter plot, histogram, boxplot.

# In[9]:


# Statistik deskriptif Avg. Area Income
data['Avg. Area Income'].describe()


# In[10]:


# Statistik deskriptif Price
data['Price'].describe()


# In[12]:


# Menampilkan Scatter Plot Avg. Area Income vs Price

plt.scatter(data['Avg. Area Income'], data['Price'], color='g')
plt.xlabel("Price")
plt.ylabel("Avg. Area Income")
plt.title("Scatter Plot Avg. Area Income vs Price")
plt.show()


# In[13]:


# Menampilkan Histogram

plt.hist(data['Avg. Area Income'], 25)
plt.title('Histogram Avg. Area Income')
plt.show()

plt.hist(data['Price'], 25)
plt.title('Histogram Price')
plt.show()


# In[63]:


# Menampilkan Box Plot

plt.boxplot(data['Avg. Area Income'])
plt.title("Box Plot Avg. Area Income",fontsize = 15)
plt.show()

plt.boxplot(data['Price'])
plt.title("Box Plot Price",fontsize = 15)
plt.show()


# <b>4. Analisis Model Regresi
# * Diduga Avg. Area Income dan Price memiliki korelasi, sehingga akan dilakukan analisis Regresi Linear Sederhana untuk mengetahui pengaruh Avg. Area Income terhadap Price.

# In[74]:


# Menentukan data training dan data testing

msk = np.random.rand(len(data)) < 0.9
train = data[msk]
test = data[~msk]


# In[75]:


# Identify Regresi Linear
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Avg. Area Income']])
train_y = np.asanyarray(train[['Price']])
regr.fit (train_x, train_y)

# Parameter Model
print ('Intercept: ',regr.intercept_)
print ('Coefficients: ', regr.coef_)


# In[76]:


# MAE, MSE, R2

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['Avg. Area Income']])
test_y = np.asanyarray(test[['Price']])
y_hat = regr.predict(test_x)

print("Mean Absolute Error (MAE): %.2f" % np.mean(np.absolute(y_hat-test_y)))
print("Residual Sum of Squares (MSE): %.2f" % np.mean((y_hat-test_y) ** 2))
print("R2-Score: %.2f" % r2_score(test_y, y_hat) )


# * <b>Dari hasil diatas, didapat model Regresi Linear Sederhana y = - 218506,6035 + 21,1709(X1) dengan nilai MSE model = 69991789094,44. Nilai R2 = 0,42 dapat diartikan bahwa sebesar 42% variabel Average Area Income mempengaruhi Price.

# In[97]:


# Scatter Plot 
plt.figure(figsize=(10,7))
plt.plot(test_x, y_hat,  'r')
plt.plot(test_x,test_y, 'bo') 
plt.ylabel('Predicted house prices', fontsize=15)
plt.xlabel('Actual test set house prices', fontsize=15)
plt.title('Actual vs Predicted House Price', fontsize=20)
plt.show()


# * <b>Selain itu, dengan menggunakan data USA Housing dapat dilakukan Analisis Regresi Linear Berganda untuk mengetahui pengaruh variabel Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, dan Area Population terhadap Price.

# In[79]:


# Making a list out of column names
l_column = list(data.columns)
len_feature = len(l_column)

# Mendefinisikan X dan Y baru untuk Regresi Linear Berganda
X = data[l_column[0:len_feature-2]]
Y = data[l_column[len_feature-2]]


# In[80]:


#Import packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Menentukan data training dan data testing secara random
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)


# In[81]:


# Identify Regresi Linear
lm = LinearRegression()
lm.fit(X_train,Y_train)

# Parameter Model
print("Intercept:", lm.intercept_)
print("Coefficients:", lm.coef_)


# In[82]:


cdf = pd.DataFrame(data=lm.coef_, index=X_train.columns, columns=["Coefficients"])
cdf


# In[87]:


train_pred=lm.predict(X_train)
predictions = lm.predict(X_test)

print("Mean absolute error (MAE):", metrics.mean_absolute_error(Y_test,predictions))
print("Mean square error (MSE):", metrics.mean_squared_error(Y_test,predictions))
print("R-squared value of predictions:",round(metrics.r2_score(Y_test,predictions),2))


# * <b>Dari hasil diatas, didapat model Regresi Linear Berganda y = - 2631028,9017 + 21,5976(X1) + 165201,1049(X2) + 119061,4639(X3) + 3212,5856(X4) + 15,2281(X5) dengan nilai MSE model = 10489638335,8055. Nilai R2 = 0,92 dapat diartikan bahwa sebesar 92% variabel Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, dan Area Population secara bersama-sama mempengaruhi Price.

# In[93]:


# Scatter Plot
plt.figure(figsize=(10,7))
plt.title("Actual vs. Predicted House Prices",fontsize=20)
plt.xlabel("Actual test set house prices",fontsize=15)
plt.ylabel("Predicted house prices", fontsize=15)
plt.scatter(x=Y_test,y=predictions)


# <b>5. Analisis Clustering
# * KMeans Clustering<br> Data akan dikelompokkan menjadi 3 cluster, yaitu Cluster 1, Cluster 2, dan Cluster 3.

# In[109]:


data1 = data.drop('Address', axis=1)
data2 = data1.drop('Avg. Area Number of Bedrooms', axis=1)
data3 = data2.drop('Avg. Area Number of Rooms', axis=1)
data4 = data3.drop('Avg. Area House Age', axis=1)
data5 = data4.drop('Area Population', axis=1)
data5.head()


# In[110]:


X_value=np.array(data['Avg. Area Income'])
Y_value=np.array(data['Price'])


# In[111]:


X_value


# In[112]:


# Menentukan centroid secara random
c1 = (83695.272383 , 630943)
c2 = (63345.240046, 1568701)
c3 = (59323.792100,  1577018)


# In[113]:


def calculate_distance(centroid, X, Y):
    distances = []
        
    # Unpack the x and y coordinates of the centroid
    c_x, c_y = centroid
        
    # Iterate over the data points and calculate the distance using the           # given formula
    for x, y in list(zip(X, Y)):
        root_diff_x = (x - c_x) ** 2
        root_diff_y = (y - c_y) ** 2
        distance = np.sqrt(root_diff_x + root_diff_y)
        distances.append(distance)
        
    return distances


# In[119]:


data5['C1_Distance'] = calculate_distance(c1, X_value, Y_value)
data5['C2_Distance'] = calculate_distance(c2, X_value, Y_value)
data5['C3_Distance'] = calculate_distance(c3, X_value, Y_value)


# In[121]:


# Preview the data
print(data5.head())


# In[123]:


# Get the minimum distance centroids
data5['Cluster'] = data5[['C1_Distance', 'C2_Distance', 'C3_Distance']].apply(np.argmin, axis =1)


# In[124]:


# Map the centroids accordingly and rename them
data5['Cluster'] = data5['Cluster'].map({'C1_Distance': 'C1', 'C2_Distance': 'C2', 'C3_Distance': 'C3'})


# In[125]:


# Preview the data
print(data5)


# In[129]:


# Using scikit-learn to perform K-Means clustering
from sklearn.cluster import KMeans
    
# Specify the number of clusters (3) and fit the data X
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Get the cluster centroids
print(kmeans.cluster_centers_)


# In[130]:


# Get the cluster labels
print(kmeans.labels_)


# In[133]:


# Calculate silhouette_score
from sklearn.metrics import silhouette_score

print(silhouette_score(X, kmeans.labels_))


# In[134]:


# Using scikit-learn to perform K-Means clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Specify the number of clusters (3) and fit the data X
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print('silhoutte score untuk cluster = 2', silhouette_score(X, kmeans.labels_))

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print('silhoutte score untuk cluster = 3', silhouette_score(X, kmeans.labels_))

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
print('silhoutte score untuk cluster = 4', silhouette_score(X, kmeans.labels_))


# * <b>Dari hasil diatas, diperoleh silhoutte score untuk cluster = 2 adalah 0,3177 , untuk cluster = 3 adalah 0,3285 , dan untuk cluster = 4 adalah 0,3101. Sehingga dapat disimpulkan bahwa banyak cluster optimal yang dapat dibentuk menggunakan KMeans Clustering adalah 3 cluster, karena menghasilkan nilai silhoutte tertinggi.

# In[ ]:




