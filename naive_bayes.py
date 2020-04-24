
# Reference : https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
# Creating a Naive-Bayes Classifier from scikit-learn



# step 1 : defining dataset

# Assigning features and label variables

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']

temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# step 2: Encoding Features

# Import LabelEncoder

from sklearn import preprocessing

#creating labelEncoder

le = preprocessing.LabelEncoder()
# Converting string labels into numbers.

wheather_encoded=le.fit_transform(wheather)

print wheather_encoded

# Converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)
print "Temp:",temp_encoded
print "Play:",label

#Combinig weather and temp into single listof tuples
features=zip(weather_encoded,temp_encoded)
print features


# Generating Model :
# Generate a model using naive bayes classifier in the following steps:
# 1.Create naive bayes classifier
# 2.Fit the dataset on classifier
# 3.Perform prediction

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features,label)

#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print "Predicted Value:", predicted

#---------------------------------------------------------------------------

# Naive-Bayes with Multi-Labels:

# loading data:
#Import scikit-learn dataset library
from sklearn import datasets

#Load dataset
wine = datasets.load_wine()

# Exploring data:

# print the names of the 13 features
print "Features: ", wine.feature_names

# print the label type of wine(class_0, class_1, class_2)
print "Labels: ", wine.target_names

# Import train_test_split function
from sklearn.cross_validation import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109) # 70% training and 30% test

# Model Generation
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

# Evaluating the Model

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



