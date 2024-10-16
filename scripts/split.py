from sklearn.model_selection import train_test_split
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename")
args = parser.parse_args()
filename = args.filename
directory = os.getcwd()

with open(filename, 'r') as f:
    f = f.readlines()

train, test = train_test_split(f, test_size=0.2)
val, test = train_test_split(test, test_size=0.5)

with open(os.path.join(directory, 'train.json'), 'w') as a, open(os.path.join(directory, 'test.json'), 'w') as b, open(os.path.join(directory, 'val.json'), 'w') as c:
    for i in train:
        a.write(i)

    for i in test:
        b.write(i)

    for i in val:
        c.write(i)
