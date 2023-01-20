
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

"""This code imports several libraries:

numpy is a library for working with arrays and mathematical operations.
pandas is a library for working with data in a tabular format, similar to a spreadsheet.
matplotlib is a library for creating visualizations, such as plots and charts.
seaborn is a library built on top of matplotlib that provides additional functionality for creating visualizations.
warnings is a built-in Python library for issuing warnings. The line warnings.filterwarnings("ignore") tells Python to ignore any warnings that are generated.
"""

data = pd.read_csv('E:/office/17.01.2023/data.csv')
print(plt.style.available) 
plt.style.use('ggplot')
data = data.replace('..', np.nan)
data.isnull().sum().any()
data = data.dropna()

"""This code does the following:

data = pd.read_csv('c06cfb6f-438d-4eab-a999-8b8f83f8b065_Data.csv') reads a CSV file and stores it in a variable called data using the pd.read_csv() function from the pandas library.

print(plt.style.available) prints a list of available styles that can be used to customize the appearance of plots created with matplotlib.

plt.style.use('ggplot') sets the current style for plots to 'ggplot'.

data = data.replace('..', np.nan) replaces all instances of '..' in the dataframe with np.nan which stands for 'Not a Number'

data.isnull().sum().any() returns True if there are any missing values in the dataset, False otherwise.

data = data.dropna() removes any rows that contain missing values (rows with np.nan) from the dataframe and overwrites the original dataframe with the new one, which no longer contains any missing values.
"""

data.head()

"""data.head() will display the first 5 rows of the data dataframe. It allows to check the top entries of the dataframe which could include column names, values and types. This is useful for quickly getting a sense of what the data looks like and how it is structured."""

data.isnull().sum().any()

"""The code you provided checks if there are any missing values in the dataframe by using the isnull() method to check for null values and then summing the result along each column using sum(), and then checking if any of the sums are non-zero using any(). This will return True if there are any missing values in the dataframe and False otherwise.

It should be noted that this code will only work if the dataframe is called data. If the dataframe has a different name, you should replace data with the correct name.

It's also important to note that this code will only check for missing values represented by np.nan or pd.NaT or pd.NA and not for empty strings, or other forms of missing values, if you have any of those, you should check for them separately, or use more robust libraries that can handle missing values like missingno or sweetviz.
"""

data = data.dropna()

"""The code you provided will remove all rows that contain missing values (represented by np.nan, pd.NaT or pd.NA) from the dataframe. The dropna() method removes all rows with one or more missing values by default.

This code will modify the original dataframe, and any missing values in the dataframe will be permanently removed, so you should be careful when using this method and make sure that you have a backup of the original data or that you are able to reload the data in case you need it.

You could also use the argument thresh in the dropna() method to specify a minimum number of non-missing values required to keep a row. For example, data.dropna(thresh=3) will remove all rows that have less than 3 non-missing values.

You could also use fillna() method to fill the missing values with a given value or using interpolation methods like ffill or bfill.

It's also important to note that if the dataframe is big, dropping missing values can lead to a significant loss of information and should be used with caution.
"""

data.head()

data.info()

data = data.dropna()
print("The total number of data-points after removing the rows with missing values are:", len(data))

"""The first line of code you provided removes all rows that contain missing values (represented by np.nan, pd.NaT or pd.NA) from the dataframe using the dropna() method.

The second line of code uses the len() function to print the total number of data points remaining in the dataframe after the missing values have been removed. By using the len() function on the dataframe, it returns the number of rows in the dataframe. The message "The total number of data-points after removing the rows with missing values are:" will be printed, followed by the number of rows remaining in the dataframe.

It's important to note that this code will only remove the missing values represented by np.nan, pd.NaT or pd.NA, if you have any other forms of missing values, you should check for them separately and drop them if you want to.

Also, if the dataframe is big, dropping missing values can lead to a significant loss of information and should be used with caution. You should consider if it's appropriate to drop the missing values or if it's better to fill them with interpolated values or other methods.
"""

X = data.drop(['Series Name', 'Series Code', 'Country Name', 'Country Code'], axis=1)
y = data["Series Name"]

"""The first line of code you provided is used to extract the input features (X) and target variable (y) from the dataframe. The drop() method is used to remove specified columns from the dataframe. In this case, the columns 'Series Name', 'Series Code', 'Country Name', 'Country Code' are being removed from the dataframe using the axis=1 argument, which indicates that the operation should be applied to columns, not rows. The resulting dataframe is then assigned to the variable X.

The second line of code assigns the column 'Series Name' from the dataframe to the variable y. This column represents the target variable that we want to predict using the input features in X.

It's important to note that the column names used in the drop() method should match the column names in the dataframe exactly. Also, this code assumes that the dataframe contains columns named 'Series Name', 'Series Code', 'Country Name', 'Country Code' and that the dataframe is named 'data'.

In addition, it's important to check if the values of the target variable(y) are numerical or categorical, if they are categorical you should use one-hot encoding or other encoding methods to convert them into numerical values.
"""

X

cols = X.columns

from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X = ms.fit_transform(X)

"""The code you provided uses the MinMaxScaler class from the sklearn.preprocessing module to scale the input feature data (X).

The MinMaxScaler is a preprocessing method that scales the features in a given range, usually between 0 and 1. This scaling is done by subtracting the minimum value of each feature from the values and then dividing by the range (max - min).

The MinMaxScaler() instantiates an object of the class, and the fit_transform() method is applied on the input feature data (X), which computes the minimum and maximum to be used for later scaling, and returns the transformed data.

It's important to note that the scaling should be done only on the input features and not on the target variable. Also, it's important to keep the same scaler object that was fit on the training data to use on the test data.

Also, if you have any categorical values, you should first convert them to numerical values using one-hot encoding or other encoding methods before applying the MinMaxScaler.
"""

X = pd.DataFrame(X, columns=[cols])

"""The code you provided is used to convert the scaled input features (X) into a DataFrame and assign column names to the DataFrame.

The pd.DataFrame() function creates a new DataFrame from the input data (X), and the columns parameter is used to assign column names to the DataFrame. In this case, the column names are passed as the list [cols], where cols is a variable that contains the original column names of the input features.

It's important to keep the same column names after scaling to keep the dataframe understandable and interpretable. This will make it easier to compare the original and scaled data, and also to identify which feature corresponds to which column in the dataframe.

It's also important to note that this code assumes that the input feature data (X) has been scaled using the MinMaxScaler and that the original column names of the input features are stored in the variable cols.
"""

X.head()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0) 

kmeans.fit(X)

"""The code you provided is used to perform k-means clustering on the input feature data (X) using the KMeans class from the sklearn.cluster module.

The KMeans() function instantiates an object of the class with the specified parameters. In this case, the number of clusters is set to 2 using the n_clusters parameter and a random seed is set using the random_state parameter. This will ensure that the clustering results are reproducible.

The fit() method is then applied on the input feature data (X) to train the model. The model will group the input feature data into 2 clusters based on the similarity of the features.

It's important to note that the number of clusters should be chosen based on the specific problem and the nature of the data. You could use the elbow method to find the optimal number of clusters or you could use the silhouette score to evaluate the quality of the clustering.

Also, it's important to keep in mind that k-means clustering is sensitive to the initial centroid locations, and it might not find the global optimum, so it's important to run the clustering multiple times and choose the best clustering results.
"""

kmeans.cluster_centers_

kmeans.inertia_

labels = kmeans.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

"""The first line of code you provided assigns the cluster labels generated by the KMeans model to the variable "labels". The labels_ attribute of the KMeans object contains the cluster assignments for each sample in the input feature data (X).

The next lines of code are used to evaluate the quality of the clustering by comparing the cluster assignments (labels) to the true labels (y) of the input feature data. The code compares the two arrays element by element using the == operator and then summing the number of times they match. The result of this comparison is then stored in the variable "correct_labels"

The last line of code prints the result of the evaluation in a human-readable format. It gives the number of samples that were correctly labeled and the total number of samples.

It's important to note that k-means clustering is an unsupervised learning method, so the true labels of the input feature data may not be known. This evaluation is only possible if true labels are available, and it's not always the case. In this case, you should use other evaluation metrics like silhouette score, or calculate the inertia of the clusters.

Also, it's important to note that this code assumes that the input feature data (X) has been clustered using the KMeans model and that the true labels of the input feature data are stored in the variable 'y'.
"""

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))

from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()

"""The code you provided is used to perform the Elbow method for determining the optimal number of clusters for K-Means clustering on the input feature data (X).

The Elbow method is used to determine the number of clusters that gives the best trade-off between the clustering quality (inertia) and the number of clusters. Inertia is a measure of the sum of the squared distances between the data points and the cluster center.

The code uses a for loop to iterate over a range of possible number of clusters (1 to 10), and for each number of clusters, it creates an instance of the KMeans class with the given number of clusters, using 'k-means++' initialization, max_iter = 300, n_init = 10 and random_state = 0. Then it fits the KMeans model to the input feature data (X) and appends the inertia of the model to the list cs.

Then the code creates a plot of the inertia values against the number of clusters using the Matplotlib library. The plot helps to identify the point at which the inertia starts to decrease at a slower rate, which is often considered as the optimal number of clusters.

It's important to note that the optimal number of clusters can vary depending on the dataset and the problem, the Elbow method is a heuristic method and the results should be validated with other metrics. Also, the initialization method, max_iter, n_init, and random_state parameters can have an impact on the final clustering result, and it's important to try different values to find the best results.
"""

data1 = data.replace({'[^0-9.]+': ''}, regex=True)
data = data1.replace('..', np.nan)
data.isnull().sum().any()
data1 = data1.dropna()
data1

data

data = data.apply(pd.to_numeric, errors='coerce')

data = data.dropna()

data = data.replace('12.28629796619571.4438030241194195429821.86137729962901431470022.078',np.NaN)
data = data.dropna()

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
year = ['2017 [YR2017]',	'2018 [YR2018]',	'2019 [YR2019]',	'2020 [YR2020]', '2021 [YR2021]']
GDP = [43.1279998000218,	43.0081024392632,	42.9040289361208,	42.8201985231957, 42.7588032477727]
fig1, ax1 = plt.subplots()
ax1.pie(GDP, labels=year, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  

plt.show()

"""The code you provided creates a pie chart that shows the proportion of GDP in different years. The chart will be plotted counter-clockwise.

The first two lines of code define the year and GDP variables, which contain the labels and values for the pie chart, respectively. The values in GDP are the percentage of GDP in each year.

Then, the code creates a figure and an axis using the plt.subplots() function from the Matplotlib library. The pie() function is used to create the pie chart and several parameters are passed to customize the chart. The labels are passed using the labels parameter, and the GDP values are passed using the GDP parameter. The autopct parameter is used to display the percentage value of each slice, the shadow parameter is set to True to give a 3D effect to the chart, and the startangle parameter is set to 90 degrees to rotate the chart counter-clockwise.

The axis('equal') function is used to ensure that the pie chart is drawn as a circle, and not as an ellipse.

The last line of code plt.show() is used to display the chart.
It's important to note that this code assumes that the GDP data is in the form of a list and that the year data is in the form of a list. Also, the number of elements in the GDP list and the year list should be the same.
"""

sns.barplot(GDP, year)
plt.show()

"""The code you provided creates a bar chart that shows the GDP for different years.

The first line of code uses the barplot() function from the seaborn library to create the bar chart. The GDP values are passed as the first argument, and the year labels are passed as the second argument. The barplot() function creates a bar chart with the GDP values on the y-axis and the year labels on the x-axis.

The second line of code plt.show() is used to display the chart.

It's important to note that this code assumes that the GDP data is in the form of a list and that the year data is in the form of a list. Also, the number of elements in the GDP list and the year list should be the same. Also, the GDP data should be in the form of a list of numerical values and the year data should be in the form of a list of strings.

If you want to add the label to y-axis you can use plt.ylabel('GDP') and for x-axis you can use plt.xlabel('Year')
"""

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
year = ['2017 [YR2017]',	'2018 [YR2018]',	'2019 [YR2019]',	'2020 [YR2020]', '2021 [YR2021]']
GDP = [43.1279998000218,	43.0081024392632,	42.9040289361208,	42.8201985231957, 42.7588032477727]
ax.bar(GDP,year)
plt.show()

"""The code you provided creates a bar chart that shows the GDP for different years.

The first line imports the matplotlib library, the second line creates a new figure, the third line creates an axis and adds it to the figure. The following two lines define the year and GDP variables, which contain the labels and values for the bar chart, respectively.

The last line uses the bar() function to create the bar chart. The bar() function takes two arguments. The first argument should be the x-axis values and the second argument should be the y-axis values. In this case, the x-axis values are passed as GDP and y-axis values are passed as year, which will cause the chart to be plotted incorrectly. It should be ax.bar(year, GDP) instead of ax.bar(GDP, year).

The last line of code plt.show() is used to display the chart.

It's important to note that this code assumes that the GDP data is in the form of a list of numerical values and the year data is in the form of a list of strings. Also, the number of elements in the GDP list and the year list should be the same.

If you want to add the label to y-axis you can use ax.set_ylabel('GDP') and for x-axis you can use ax.set_xlabel('Year')
"""

data

df = pd.read_csv('c06cfb6f-438d-4eab-a999-8b8f83f8b065_Data.csv')
df = df.replace('..', np.nan)
df = df.dropna()
df

from sklearn.linear_model import LinearRegression
# create the independent and dependent variables
X = df[['1990 [YR1990]', '2000 [YR2000]','2020 [YR2020]', '2013 [YR2013]', '2014 [YR2014]', '2015 [YR2015]', '2016 [YR2016]', '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]']].values
y = df['2012 [YR2012]'].values

# fit the linear regression model
reg = LinearRegression().fit(X, y)

# predict the value for 2021
predicted_2021 = reg.predict([[df['1990 [YR1990]'].values[0], df['2000 [YR2000]'].values[0], df['2012 [YR2012]'].values[0], df['2013 [YR2013]'].values[0], df['2014 [YR2014]'].values[0], df['2015 [YR2015]'].values[0], df['2016 [YR2016]'].values[0], df['2017 [YR2017]'].values[0], df['2018 [YR2018]'].values[0], df['2019 [YR2019]'].values[0]]])

# print the predicted value
print("Predicted value for 2021:", predicted_2021[0])

"""The code you provided is used to create a linear regression model and use it to predict the GDP for 2021.

The first line imports the LinearRegression class from the sklearn.linear_model module, which is used to create the linear regression model.

The next lines of code create the independent and dependent variables for the model. The independent variables are the GDP values for the years 1990, 2000, 2020, 2013, 2014, 2015, 2016, 2017, 2018, and 2019, and they are stored in the variable X. The dependent variable is the GDP value for the year 2012, and it is stored in the variable y.

The LinearRegression().fit(X, y) line is used to fit the linear regression model using the independent and dependent variables.

The next line of code uses the predict() method of the LinearRegression object to predict the GDP value for 2021, using the GDP values for the years 1990, 2000, 2020, 2013, 2014, 2015, 2016, 2017, 2018, and 2019 as input.

The last line of code prints the predicted GDP value for 2021.

It's important to note that this code assumes that the input dataframe df contains all the columns specified in the code and that the data is in the form of numerical values. Also, the predictions made by a linear regression model are based on the assumption that the relationship between the independent and dependent variables is linear. If the relationship is not linear, then the predictions may not be accurate.
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit the linear regression model
reg = LinearRegression().fit(X_train, y_train)

# calculate the accuracy score
score = reg.score(X_test, y_test)

# print the accuracy score
print("Accuracy score:", score)

"""The code you provided is used to evaluate the accuracy of a linear regression model.

The first line of code imports the train_test_split function from the sklearn.model_selection module, which is used to split the input data into a training set and a test set. The test_size parameter is set to 0.2, which means that 20% of the data will be used for testing, and 80% of the data will be used for training. The X and y variables from the previous example are passed to the train_test_split function, which returns the training and test sets for the independent and dependent variables.

The next line of code uses the LinearRegression().fit(X_train, y_train) line to fit the linear regression model using the training data.

The following line uses the score() method of the LinearRegression object to calculate the accuracy score of the model, using the test data. The score method returns the coefficient of determination R^2 of the prediction. The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().

The last line of code prints the accuracy score of the model.

It's important to note that this code assumes that the input data is in the form of numerical values and that the data is split into two sets: training and test sets, where the model is trained on the training set and then evaluated on the test set. The accuracy score ranges between 0 and 1, where a score of 1 indicates that the model is a perfect fit, and a score of 0 indicates that the model is not a good fit.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X = df[[ '2000 [YR2000]','2012 [YR2012]', '2013 [YR2013]', '2014 [YR2014]', '2015 [YR2015]', '2016 [YR2016]', '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]','2020 [YR2020]']].values
y = df['1990 [YR1990]'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# make predictions on the test set
y_pred = clf.predict(X_test)

# calculate the accuracy score
score = accuracy_score(y_test, y_pred)

# print the accuracy score
print("Accuracy score:", score)

"""The code you provided is used to train a logistic regression model and evaluate its accuracy.

The first two lines of code import the LogisticRegression class and the accuracy_score function from the sklearn.linear_model and sklearn.metrics modules respectively.

The next lines of code create the independent and dependent variables for the model. The independent variables are the GDP values for the years 2000, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, and 2020, and they are stored in the variable X. The dependent variable is the GDP value for the year 1990, and it is stored in the variable y.

The train_test_split(X, y, test_size=0.4) function is used to split the data into a training set and a test set, with 40% of the data being used for testing and 60% of the data being used for training.

The next line creates an instance of the LogisticRegression class and assigns it to the variable clf.

The following line uses the fit() method of the LogisticRegression object to train the model on the training data.

The next line uses the predict() method of the LogisticRegression object to make predictions on the test set.

The following line uses the accuracy_score() function to calculate the accuracy of the model, using the test set and the predictions made on the test set as inputs. The accuracy score ranges between 0 and 1, where a score of 1 indicates that the model is a perfect fit, and a score of 0 indicates that the model is not a good fit.

The last line of code prints the accuracy score of the model.

It's important to note that this code assumes that the input dataframe df contains all the columns specified in the code and that the data is in the form of numerical values. Also, logistic regression is a supervised learning algorithm that is used for binary classification tasks. If the output variable is not binary it should be transformed in order to use logistic regression.
"""

