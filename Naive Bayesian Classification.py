import csv
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

#read data from csv file
data = []
with open('Flying_Fitness.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',', quotechar = '|')
	for row in reader:
		data.append(row)
freq=[]
for i in range(6):
	freq.append({})
for i in range(1,7):
	for j in range(1,len(data)):
		if not(data[j][i] in freq[i-1]):
			freq[i-1][data[j][i]]=1
		else:
			freq[i-1][data[j][i]] += 1

freq_1=[]
freq_0=[]
for i in range(5):
	freq_1.append({})
	freq_0.append({})
for i in range(2,7):
	for j in range(1,len(data)):
		if not(data[j][i] in freq_1[i-2]):
			freq_1[i-2][data[j][i]]=0
		if data[j][1]=='1':
			freq_1[i-2][data[j][i]] += 1
		if not(data[j][i] in freq_0[i-2]):       
			freq_0[i-2][data[j][i]]=0
		if data[j][1]=='0':
			freq_0[i-2][data[j][i]] += 1
for i in range(5):
	for j in range(4):
		if str(j) in freq_1[i]:
			freq_1[i][str(j)] +=1
			freq[i+1][str(j)] +=1
			freq[0]['1'] +=1
		if str(j) in freq_0[i]:
			freq_0[i][str(j)] +=1
			freq[i+1][str(j)] +=1
			freq[0]['0'] +=1

scores = []
for i in range(1,len(data)):
	score = 1
	for j in range(2,7):
		score = score * freq_1[j-2][data[i][j]]/float(freq[0]['1'])
	scores.append(score)
spe = []
sen = []
for i in range(len(scores)):
	t = scores[i]
	prediction = []
	TP=0
	TN=0
	for j in range(1, len(data)):
		if scores[j-1]>=t:
			prediction.append('1')
		else:
			prediction.append('0')
	for k in range(len(prediction)):
		if data[k+1][1]=='1' and prediction[k] == '1':
			TP +=1
		elif data[k+1][1]=='0' and prediction[k] == '0':
			TN +=1
	sen.append(float(TP)/freq[0]['1'])
	spe.append(1-float(TN)/freq[0]['0'])

plt.xlabel('1-Specity')  
plt.ylabel('Sensitivity') 
plt.plot(sen,spe,'bo',markersize = 5)
plt.savefig('ROC_a') 




