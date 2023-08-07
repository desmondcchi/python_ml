import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

label = "G3"  # predict also means label in ML

x = np.array(data.drop(labels=[label], axis=1)) # 5 dimensions
y = np.array(data[label])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1
)  # test_size means that 10% of the data will go in the test dataset and 90% will go into the training dataset

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print("Coefficients: ", linear.coef_) # 5 coefficients because of 5 dimensions
print("Intercept", linear.intercept_)

# Using the model.
predictions = linear.predict(x_test)

# Print the predicted values and the expected values.
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])