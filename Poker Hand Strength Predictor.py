#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# In[9]:


# Load the data
data = pd.read_csv("C:\\Users\\Shaurya\\Downloads\\poker-hand-testing.data", header=None)


# In[10]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)


# In[11]:


# Scale the input data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[12]:


# Create a Logistic Regression model
model = LogisticRegression(max_iter=10000)


# In[13]:


# Train the model on the training data
model.fit(X_train_scaled, y_train)


# In[14]:


# Make predictions on the test data
y_pred = model.predict(X_test_scaled)


# In[15]:


# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:




