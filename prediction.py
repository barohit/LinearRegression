import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection, preprocessing
from sklearn import linear_model

def main():
    data = pd.read_csv('data.csv')
    predicted_category = "years"

    label_encoder = preprocessing.LabelEncoder()
    brand = label_encoder.fit_transform(list(data["brand"]))


    x = list(zip(np.array(data["maintenance-quality"]), np.array(data["yearly-mileage"]), brand))
    y = np.array(data[predicted_category])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

    linear_regression_model = linear_model.LinearRegression()
    linear_regression_model.fit(x_train, y_train)
    acc = linear_regression_model.score
    predicted_values = linear_regression_model.predict(x_test)

    for i in range(len(predicted_values)):
        print(str(y_test[i]) + ": " + str(predicted_values[i]))
    print(acc)

main()
