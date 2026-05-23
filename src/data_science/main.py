import sys

import pandas as pd

from data_science import pipeline as classif
from data_science import unsupervised as unsup


def report(source, dataframe, task):

    print(dataframe)
    """task = task[:-1]

    dataframe.insert(755, 'class', 0)
    i = 0
    for value in dataframe['class\r']:
        new_value = int(value[:-1])
        dataframe.loc[i, 'class'] = new_value
        i+=1
    del dataframe['class\r']"""

    if task == "classification":
        classif.classification(dataframe, source)
    elif task == "unsupervised":
        unsup.run(source, dataframe)
    else:
        # pr.preprocessing(dataframe)
        pass


def main():
    """A: read arguments"""
    args = sys.stdin.readline().rstrip("\n").split(" ")
    n, source, task = int(args[0]), args[1], args[2]

    """B: read dataset"""
    data, header = [], sys.stdin.readline().rstrip("\n").split(",")
    for i in range(n - 1):
        data.append(sys.stdin.readline().rstrip("\n").split(","))
    dataframe = pd.DataFrame(data, columns=header)

    """C: output results"""
    report(source, dataframe, task)


if __name__ == "__main__":
    main()
