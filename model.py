import pandas as pd
""" data= pd.read_csv("employeeChurn.csv") """


import os
import sklearn as sklearn
import pickle

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "employeeChurn.csv"))

# Import LabelEncoder 
from sklearn import preprocessing
#creating labelEncoder 
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
data['salary']=le.fit_transform(data['salary'])
data['Departments']=le.fit_transform(data['Departments'])




X=data[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'Departments', 'salary']]
y=data['left']


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% test



#Import Gradient Boosting Classifier model
from sklearn.ensemble import GradientBoostingClassifier

#Create Gradient Boosting Classifier
gb = GradientBoostingClassifier()

#Train the model using the training sets 
gb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gb.predict(X_test)




#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


pickle.dump(gb,open('model.pkl','wb')) 
