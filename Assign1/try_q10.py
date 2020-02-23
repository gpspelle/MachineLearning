import csv
import numpy as np
from matplotlib import pyplot as plt

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

def cost(theta, truth, verified_bin, len_reviews):

    tam = len(verified_bin)
    J = 0
    for i in range(tam):
        J += (hyp(theta, verified_bin[i], len_reviews[i]) - truth[i]) ** 2 

    J *= (1/(2*tam))

    return J


def dev_cost(theta, truth, X):

    tam = len(truth)
    dev = np.empty(tam)
    h = np.empty(tam)
    for i in range(tam):
        h[i] = hyp(theta, X[i][1], X[i][2])

    return (np.dot(np.transpose(X), ( h - truth ) ))/tam 


def hyp(theta, verified_bin, len_reviews):
    return theta[0] + theta[1]*verified_bin + theta[2]*len_reviews


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

tam = len(stars)
it = 0
while it < 5:
    it += 1
    print(it)
    y_pred = np.empty(tam)
    for i in range(tam):
        y_pred[i] = hyp(theta, X[i][1], X[i][2])

    diff = stars - y_pred
    grad = dev_cost(theta, stars, X)

    theta = theta - (grad) * alpha

print(theta)
