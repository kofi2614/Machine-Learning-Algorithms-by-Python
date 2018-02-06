import json
import numpy as np
from sklearn import cluster, datasets
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import normalize
import seaborn as sns; sns.set()
import random
from copy import deepcopy
import pandas as pd
f = open('titanic.json', 'r')
d = json.load(f)
f.close()
data = []
target = []
for i in range(0, len(d)):
	data.append([d[i]['Age'],d[i]['Fare'], d[i]['SiblingsAndSpouses'],d[i]['ParentsAndChildren'],d[i]['Embarked'], d[i]['Sex']])
	target.append(int(d[i]['Survived']))
count = [0, 0, 0, 0]
total = [0, 0, 0, 0]
for i in range(0, len(data)):
	for j in range(0,4):
		if data[i][j] == u'':
			data[i][j] = 'miss'
			count[j] = count[j] +1
		else: 
			data[i][j] = float(data[i][j])
			total[j] = total[j] + data[i][j]
	for k in range(4,6):
		if data[i][k] == u'':
			data[i][k] = data[i-1][k]
		else:
			data[i][k] = str(data[i][k])
for i in range(0, len(data)):
	for j in range(0, 4):
		if data[i][j] == 'miss':
			data[i][j] = total[j]/(len(data)-count[j])           
data1 = []
for i in range(0, len(data)):
	data1.append([data[i][0], data[i][1], data[i][2]+data[i][3], data[i][4], data[i][5]])

for i in range(len(data1)):
	if data1[i][3] == 'S':
		data1[i][3] = 1
	elif data1[i][3] == 'Q':
		data1[i][3]=  2
	else:
		data1[i][3] = 3
	if data1[i][4] == 'male':
		data1[i][4] = 0
	else:
		data1[i][4] = 1  
data1 = normalize(data1)
Z = linkage(data1, method='ward', metric='euclidean')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.axhline(y = 5, c = 'k')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.savefig('Hierarchical Clustering Dendrogram')
plt.close()

k = 3
def dist(a, b, ax=1):
	return np.linalg.norm(a - b, axis=ax)
def check(clusters):
	count = np.zeros(k)
	for i in range(len(clusters)):
		count[int(clusters[i])] = count[int(clusters[i])]+1
	zero = 0
	for i in range(len(count)):
		if count[i] == 0:
			zero = zero + 1
	return zero

def initial(k, data1):
	zero = 1
	while zero != 0:
		c = []
		for i in range(k):
			row = []
			for j in range(5):
				row.append(random.uniform(0,1))
			c.append(row)
		c = np.array(c)
	
		for i in range(len(data1)):
			distances=  dist(data1[i], c)
			cluster = np.argmin(distances)
			clusters[i] = cluster
		zero = check(clusters)	
	return(c)
clusters = np.zeros(len(data1))
c = initial(k, data1)
c_old = np.zeros(c.shape)

error = dist(c, c_old, None)
feature = ['Age', 'Fare', 'Companions Count', 'Embarked Location', 'Sex']
iteration = 0 
while (error != 0) and (iteration < 11) :
	for i in range(len(data1)):
		distances=  dist(data1[i], c)
		cluster = np.argmin(distances)
		clusters[i] = cluster
	c_old = deepcopy(c)
	for i in range(k):
		points = [data1[j] for j in range(len(data1)) if clusters[j] == i]
		c[i] = np.mean(points, axis=0)
	error = dist(c, c_old, None)
	if (iteration in [0, 5, 10]) or (error == 0):
		for h in range(len(data1[1])-1):
			for l in range(h+1, len(data1[1])):
				plt.rcParams['figure.figsize'] = (16,9)
				plt.style.use('ggplot')
				fig, ax = plt.subplots()
				colors = ['r', 'g', 'b', 'y', 'c', 'm']
				mlist = ['x', '+', '.']
				plt.title('K-Means Cluster Scatter Plot, Iteration = <%i>' % iteration)
				plt.xlabel('<%s>' % feature[h])
				plt.ylabel('<%s>' % feature[l])
				for i in range(k):
					#for j in range(len(data1)):
						#if (clusters[j] == i) and (target[j] == 1):
					#		points = np.array[data1[j]
					#	elif (clusters[j] == i) and (target[j] == 0):
					#		pointu = np.array[data1[j]
					points = np.array([data1[j] for j in range(len(data1)) if (clusters[j] == i) and (target[j] == 1)])
					pointu = np.array([data1[j] for j in range(len(data1)) if (clusters[j] == i) and (target[j] == 0)])
					ax.scatter(pointu[:,h], pointu[:,l], s=50, c=colors[i], marker = '+')
					ax.scatter(points[:,h], points[:,l], s=7, c=colors[i])
				ax.scatter(c[:,h], c[:,l], marker='*', s=200, c='#050505')
				plt.savefig('X=<%s>, Y=<%s>, Iteration = <%i>.png' % (feature[h], feature[l], iteration))
				plt.close()
	iteration = iteration + 1
countc = np.zeros(k)
ccount = np.zeros(k)
for i in range(len(clusters)):
	countc[int(clusters[i])] = countc[int(clusters[i])]+1
	if target[i] ==1:
		ccount[int(clusters[i])] = ccount[int(clusters[i])]+1
print ccount
print countc
print c

