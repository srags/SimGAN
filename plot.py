import csv

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

with open('data0.csv') as csvfile:
    so = csv.reader(csvfile, delimiter=',', quotechar='"')
    # so_data = []
    # so.next()
    for row in so:
        # print(row)
        plt.plot(row)
        plt.ylabel('ylabel')
    plt.show()

with open('data4.csv') as csvfile:
    so = csv.reader(csvfile, delimiter=',', quotechar='"')
    # so_data = []
    # so.next()
    for row in so:
        print(len(row))
        plt.plot(row)
    plt.show()

with open('data8.csv') as csvfile:
    so = csv.reader(csvfile, delimiter=',', quotechar='"')
    # so_data = []
    # so.next()
    for row in so:
        print(len(row))
        plt.plot(row)
    plt.show()

with open('data12.csv') as csvfile:
    so = csv.reader(csvfile, delimiter=',', quotechar='"')
    # so_data = []
    # so.next()
    for row in so:
        print(len(row))
        plt.plot(row)
    plt.show()

with open('data16.csv') as csvfile:
    so = csv.reader(csvfile, delimiter=',', quotechar='"')
    # so_data = []
    # so.next()
    for row in so:
        print(len(row))
        plt.plot(row)
    plt.show()