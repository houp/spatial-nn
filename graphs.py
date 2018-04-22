import numpy as np
import matplotlib.pyplot as plt


def make_plot(errors, rule):
    plt.plot(errors.keys(), errors.values(), 'ro')
    plt.xlabel('size of the training set')
    plt.ylabel('square error')
    plt.savefig('graphs/' + str(rule) + '.pdf')
    plt.close()


def make_graphs_for_rules(rules):
    results = open('results/1', 'r')
    errors = eval(results.read())
    for r in rules:
        make_plot(errors[r], r)


rules = [r for r in range(50)]

make_graphs_for_rules(rules)
