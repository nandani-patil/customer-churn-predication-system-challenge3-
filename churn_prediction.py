import numpy as np
from sklearn.tree import DecisionTreeClassifier

# training data
# [years_with_company , monthly_bill]
X = np.array([
    [1,500],
    [2,600],
    [5,1000],
    [7,1200],
    [1,400],
    [6,1100]
])

# 1 = customer leaves , 0 = stays
y = np.array([1,1,0,0,1,0])

# model
model = DecisionTreeClassifier()

# training
model.fit(X,y)

# user input
years = int(input("Enter years with company: "))
bill = int(input("Enter monthly bill: "))

prediction = model.predict([[years,bill]])

if prediction[0] == 1:
    print("Customer Will Leave")
else:
    print("Customer Will Stay")
