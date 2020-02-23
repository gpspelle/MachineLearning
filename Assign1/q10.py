import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def get_rates():
    with open('amazon_reviews_us_Gift_Card_v1_00.tsv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        rates = np.zeros(5)
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                print(f'Column names are {", ".join(row)}')
            else:
                v = int(row[7])
                rates[v - 1] += 1

        print(rates)
        plt.plot([1, 2, 3, 4, 5], rates, 'o')
        plt.show()
        plt.savefig('rates.png')


def get_column(num):
    v = []
    with open("amazon_reviews_us_Gift_Card_v1_00.tsv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for lines in csv_reader:
            v.append(lines[num])

    v.pop(0)
    return v

stars = get_column(7)
verified = get_column(11)
reviews_head = get_column(12)
reviews_body = get_column(13)

stars = list(map(int, stars))
verified_bin = [1 if y == 'Y' else -1 for y in verified]

reviews = [((i + j).replace(',', '')).replace('.', '') for i, j in zip(reviews_head, reviews_body)]
len_reviews = [len(i) for i in reviews]

X = [[i, j] for i, j in zip(verified_bin, len_reviews)]

for i in range(2, 10):
    poly = PolynomialFeatures(i)
    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, stars, test_size=0.1, shuffle=False)
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, y_train)  # perform linear regression

    y_test_pred = linear_regressor.predict(X_test) 
    y_train_pred = linear_regressor.predict(X_train) 
    print("Squared error test for", i, mean_squared_error(y_test_pred, y_test))
    print("Squared error train for", i, mean_squared_error(y_train_pred, y_train))
    #print("Accuracy test for", i, r2_score(y_test, y_test_pred))
    #print("Accuracy test for", i, r2_score(y_test, y_test_pred))
    print("Score test", i, linear_regressor.score(X_train, y_train))
    print("Score train", i, linear_regressor.score(X_test, y_test))

    print("Slope for", i, linear_regressor.intercept_)
    print("Coeffs for", i, linear_regressor.coef_)
    print("***********************************************")
