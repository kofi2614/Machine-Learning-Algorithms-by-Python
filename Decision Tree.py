import csv
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import random


#read data from csv file
data = []
with open('training_set.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',', quotechar = '|')
	for row in reader:
		data.append(row)
#label dataset
#for 2-class variable such as gender and target, the label is binary (0,1)
#for other variables such as contract lenth, the label starts at 1 instead of 0
#variable such as ID and age stay unchanged
binary = [1, 3, 5, 7, 8]

for i in range(1, len(data[1])-1):
	if i != 2:
		d = {}
		if i in binary:
			count = 0
		else:
			count = 1
		for j in range(0,len(data)):
			if not(data[j][i] in d):
				d[data[j][i]] = count
				count = count+1
		for k in range(len(data)):
			data[k][i] = d[data[k][i]]

for i in range(len(data)):
	if data[i][9] == "Late":
		data[i][9] = 1
	else: 
		data[i][9] = 0

### set training and test data
trainx = []
trainy = []
testx = []
testy = []

for i in range(len(data)):
	if i % 10 == 0:
		testx.append(data[i][1:8])
		testy.append(data[i][9])
	else:
		trainx.append(data[i][1:8])
		trainy.append(data[i][9])

###build the tree
depth = [None, 2, 4, 8, 16]
node = [2, 4, 8, 16, 32, 64, 128, 256] 

for i in range(5):
	accuracy = []
	correct = 0
	incorrect = 0
	prediction = []
	for j in range(8):
		clf = tree.DecisionTreeClassifier(max_depth = depth[i], max_leaf_nodes = node[j])
		clf.fit(trainx, trainy)
		prediction = clf.predict(testx)
		for k in range(len(testy)):
			if prediction[j] == testy[j]:
				correct += 1
			else:
				incorrect += 1
		accuracy.append(float(correct)/(correct+incorrect))
	plt.scatter(node, accuracy)
	plt.title('Unbalanced dataset, Max Depth = %s' % depth[i])
	plt.savefig('Unbalanced dataset, Max Depth = %s' % depth[i])
	plt.clf()
#now try to get 50:50 dataset 
#from the total dataset, there are 1104 late adopter and 3557 non-late adopter
#so we need to throw out 2453 datapoints that are not late adopters
#randomly generate 2453 integers out of 3557, these are datapoints that we are going to throw out
throw = random.sample(range(3557),2453)
count = 1
data2 = []
for i in range(len(data)):
	if data[i][9] == 0:
		if not (count in throw):
			data2.append(data[i])
		count = count +1
	else:
		data2.append(data[i])
         

testx2 = []
testy2 = []
trainx2 = []
trainy2 = []
for i in range(len(data2)):
	if i % 10 == 0:
		testx2.append(data2[i][1:8])
		testy2.append(data2[i][9])
	else:
		trainx2.append(data2[i][1:8])
		trainy2.append(data2[i][9])

for i in range(5):
	accuracy = []
	correct = 0
	incorrect = 0
	prediction = []
	for j in range(8):
		clf = tree.DecisionTreeClassifier(max_depth = depth[i], max_leaf_nodes = node[j])
		clf.fit(trainx2, trainy2)
		prediction = clf.predict(testx2)
		for k in range(len(testy2)):
			if prediction[j] == testy2[j]:
				correct += 1
			else:
				incorrect += 1
		accuracy.append(float(correct)/(correct+incorrect))
	plt.scatter(node, accuracy)
	plt.title('Balanced dataset, Max Depth = %s' % depth[i])
	plt.savefig('Balanced dataset, Max Depth = %s' % depth[i])
	plt.clf()








