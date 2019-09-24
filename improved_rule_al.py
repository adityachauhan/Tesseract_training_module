import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats.mstats import pearsonr
from scipy.stats import mode
import csv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from collections import Counter
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
import warnings

warnings.filterwarnings("ignore")

# implementation of PCA for projection to a 2D-subspace
def compute_PCA(data):
	mean_vec = []
	for i in range(data.shape[0]):
		mean_vec.append(np.mean(data[i,:]))
	
	cov_mat = np.cov(data)

	# eigenvectors and eigenvalues computation from the covariance matrix
	eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
	eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
	total_variance = 0

	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs.sort()
	eig_pairs.reverse()

	# computation of the fraction of variance explained by the selected principal components
	for i in eig_pairs:
		total_variance += i[0]
	
	matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))
	data = data.transpose()
	Y = data.dot(matrix_w)
	return Y

# querying labels from a human expert
def ask_expert(data, indices):
	target_learnt = data[:,4]
	print(indices)
	for idx in indices:
		response = raw_input("Querying label for Customer Record: " + str(data[idx,:-1]) + " according to the Score Card.....")
 		target_learnt[idx] = int(response)
	return target_learnt

# visualisation function
def create_plot(data, labels, algo_name, qsamples):
	min_max_sc = preprocessing.MinMaxScaler()
	pca_X = min_max_sc.fit_transform(data)
	pca_X = pca_X.transpose()
	projected_data = compute_PCA(pca_X)
	colors = ['cyan','yellow','magenta']
	
	qs = list(qsamples)
	diff_labels = np.zeros(data.shape[0])

	plt.ion()
	plt.figure()

	if qsamples:
		for i in range(data.shape[0]):
			if i not in qsamples:
				diff_labels[i] = labels[i]
			else:
				diff_labels[i] = 2
		colors = ['cyan','yellow','magenta']
		scatter_plot = plt.scatter(projected_data[:,1], projected_data[:,0], c=diff_labels, cmap=matplotlib.colors.ListedColormap(colors))
		p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc=colors[0])
		p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc=colors[1])
		p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc=colors[2])
		plt.legend((p1, p2, p3), ('Defaulter','Non-Defaulter','Re-labelled Samples'), loc='best')
	else:
		colors = ['cyan','yellow']
		scatter_plot = plt.scatter(projected_data[:,1], projected_data[:,0], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
		p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc=colors[0])
		p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc=colors[1])
		plt.legend((p1, p2), ('Defaulter','Non-Defaulter'), loc='best')


	plt.xlabel('Feature 2')
	plt.ylabel('Feature 1')
	plt.title('Distribution for the two classes for '+algo_name)
	

	# plt.show()

def select_samples_for_expert(sm, new_labels):
	interesting_points = set()
	for i in range(len(new_labels)):
		indexes = np.where(sm[i,:] > 0.7)[0]
		disagree_ind = []
		count = np.zeros((2,1))
		class0 = np.where(new_labels[indexes] == 0)[0]
		class1 = np.where(new_labels[indexes] == 1)[0]

		count[0] = len(class0)
		count[1] = len(class1)
		
		# selecting points satisfying the minority threshold
		if count[0] >= count[1]:
			if count[0] > 0:
				mt_ratio = count[1]/count[0]
				if float(mt_ratio) < 0.04:
					disagree_ind = [n for n in indexes if (X[n,4]==1)]
		else:
			if count[1] > 0:
				mt_ratio = count[0]/count[1]
				if float(mt_ratio) < 0.04:
					disagree_ind = [n for n in indexes if (X[n,4]==0)]		
		if disagree_ind:
			for idx in disagree_ind:
				interesting_points.add(idx)
	return interesting_points

# reading data from the file
table = []
with open('/Users/megha/Downloads/MOCK_DATA.csv', 'rb') as csvfile:
	customerdata = csv.reader(csvfile, delimiter = ',')
	i = 0
	for row in customerdata:
		table.append(row)

# subsampling the dataset to get a uniform distribution of samples
dataset = np.zeros((500,4))
dataset = np.asarray(table[1:501])
dataX = []
count_0 = 0
count_1 = 0
for i in range(0,500):
	if count_1 <= 30:
		if int(dataset[i,0]) >= 600 and float(dataset[i,1]) <= 0.6 and float(dataset[i,2]) <= 0.4 and int(dataset[i,3]) >= 50:
			dataX.append(dataset[i,:])
			count_1 = count_1+1
		else:
			if count_0 <= 170:
				dataX.append(dataset[i,:])
				count_0 = count_0 + 1

dataX = np.asarray(dataX)
X = np.zeros((dataX.shape[0],5))
X[:,:-1] = dataX

# build the rule-based system to define a decision hyperplane (base learner)
for i in range(X.shape[0]):
	if X[i,0] >= 600: # only keeping the CIBIL score
		X[i,4] = 1
	else:
		X[i,4] = 0


scaled_X = np.zeros((X.shape[0],5))
min_max_sc = preprocessing.MinMaxScaler()
scaled_X[:,:-1] = min_max_sc.fit_transform(X[:,:-1])
scaled_X[:,4] = X[:,4]

#labeling the data according to the rule system defined - learning the true labels for the data
true_X = np.zeros((X.shape[0],5))
true_X[:,:-1] = np.asarray(X[:,:-1])
for i in range(X.shape[0]):
	if true_X[i,0] >= 600 and true_X[i,1] <= 0.6 and true_X[i,2] <= 0.4 and true_X[i,3] >= 50:
		true_X[i,4] = 1
	else:
		true_X[i,4] = 0

fpr_init, tpr_init, thresholds = roc_curve(X[:,4], true_X[:,4])
auc_data = auc(fpr_init,tpr_init)
create_plot(scaled_X[:,:-1], X[:,4], "the Base Rule Classifier with AUC : %.4f" % auc_data, set())

fpr_init, tpr_init, thresholds = roc_curve(X[:,4], true_X[:,4])
auc_data = auc(fpr_init,tpr_init)
print("AUC for Rule-based Learner: %.4f" % auc_data)

# construct a similarity matrix with cosine similarity metric
sim_matrix = np.zeros((X.shape[0],X.shape[0]))
for i in range(sim_matrix.shape[0]):
	x = scaled_X[i,:]
	for j in range(sim_matrix.shape[1]):
		if j is not i:
			y = scaled_X[j,:]
			sim_matrix[i][j] = cosine_similarity(x,y)

max_iterations = 20

improved_labels = X[:,4]
total_samples_queried = 0
samples_blk_iter = 0
samples_queried = set()

auc_data = 0.0
auc_rule = []
prev_auc = 0.0
while(round(auc_data,2) < 0.85 or max_iterations >= 0):
	prev_auc = auc_data
	sel_samples = select_samples_for_expert(sim_matrix, improved_labels)
	print(len(sel_samples))
	total_samples_queried += len(sel_samples)
	samples_blk_iter += len(sel_samples)
	
	queried_samples = []

	for smp in sel_samples:
		if smp not in samples_queried:
			queried_samples.append(smp)
		samples_queried.add(smp)

	improved_labels = ask_expert(X,sel_samples)
	# print(improved_labels)

	fpr, tpr, thresholds = roc_curve(improved_labels, true_X[:,4])
	auc_data = auc(fpr,tpr)
	
	print("AUC Improvement in the Base Rule Learner : %.4f" % auc_data)
	
	if max_iterations % 2 == 0:
		create_plot(X[:,:-1], improved_labels, "Improvement by Crowd Wisdom with AUC : %.4f" % auc_data + "\nSamples Queried Upto this point : "+str(samples_blk_iter), samples_queried)
		auc_rule.append(auc_data)
		samples_blk_iter = 0
	max_iterations -= 1

print("Total number of samples queried by Human Expert - "+str(total_samples_queried))
print(auc_rule)

