from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import csv
import operator
import logging
import numpy as np
import warnings


warnings.filterwarnings('ignore')

default_args = {
	'owner' : 'megha',
	'start_date' : datetime(2016, 07, 28)-timedelta(days=1),
}

ml_dag = DAG('al_test', default_args = default_args, schedule_interval = '@once')

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
		# print(i[0])
		total_variance += i[0]
	# ans = eig_pairs[0][0]/total_variance
	# print("Variance Explained by first principal component : %.4f" % ans)
	# ans = eig_pairs[1][0]/total_variance
	# print("Variance Explained by second principal component : %.4f" % ans)

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
def create_plot(data, labels, algo_name):
	min_max_sc = preprocessing.MinMaxScaler()
	pca_X = min_max_sc.fit_transform(data)
	pca_X = pca_X.transpose()
	projected_data = compute_PCA(pca_X)
	colors = ['blue','yellow']
	plt.ion()
	plt.figure()
	scatter_plot = plt.scatter(projected_data[:,1], projected_data[:,0], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
	plt.xlabel('Feature 2')
	plt.ylabel('Feature 1')
	plt.title('Distribution for the two classes for '+algo_name)
	# plt.title(algo_name)
	p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc=colors[0])
	p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc=colors[1])
	plt.legend((p1, p2), ('Defaulter','Non-Defaulter'), loc='best')
	# plt.show()

def generate_true_labels(filename):
	# reading data from the file
	table = []
	with open('/Users/megha/Downloads/'+filename+'.csv', 'rb') as csvfile:
		customerdata = csv.reader(csvfile, delimiter = ',')
		for row in customerdata:
			table.append(row)
	logging.info("Successfully loaded the feature file")

	# subsampling the dataset to get a uniform distribution of samples
	true_data = np.zeros((500,5))
	true_data[:,:-1] = np.asarray(table[1:501])
	for i in range(true_data.shape[0]):
		# heuristic rules to generate the data labels
		if true_data[i,0] >= 600 and true_data[i,1] <= 0.6 and true_data[i,2] <= 0.4 and true_data[i,3] >= 50:
			true_data[i,4] = 1
		else:
			true_data[i,4] = 0
	logging.info("Generated labels based on heuristics for the unlabelled data")
	return true_data

def base_rule_learner():
	loan_data = generate_true_labels('MOCK_DATA')
	for i in range(loan_data.shape[0]):
		if loan_data[i,0] >= 600:
			loan_data[i,4] = 1
		else:
			loan_data[i,4] = 0
	return loan_data

def subsample_for_sample_selection():
	sampled_data = []
	true_data = generate_true_labels("MOCK_DATA")
	base_data = base_rule_learner()
	logging.info("Successfully built the base-rule model")

	fpr, tpr, thresh = roc_curve(base_data[:,4], true_data[:,4])
	auc_data = auc(fpr,tpr)
	logging.info("Base-Rule Classifier with AUC: %.4f" % auc_data)

	sorted_data = sorted(base_data, key=operator.itemgetter(0))
	sorted_data = np.asarray(sorted_data)

	scaled_data = np.zeros((sorted_data.shape[0],5))
	min_max_sc = preprocessing.MinMaxScaler()
	scaled_data[:,:-1] = min_max_sc.fit_transform(sorted_data[:,:-1])
	scaled_data[:,4] = sorted_data[:,4]
	
	for i in range(scaled_data.shape[0]):
		if i%4 == 0:
			sampled_data.append(scaled_data[i,:])

	sampled_data = np.asarray(sampled_data)
	sim_matrix = np.zeros((sampled_data.shape[0],sampled_data.shape[0]))
	for i in range(sim_matrix.shape[0]):
		x = sampled_data[i,:-1]
		for j in range(sim_matrix.shape[1]):
			if j is not i:
				y = sampled_data[j,:-1]
				sim_matrix[i][j] = cosine_similarity(x,y)

	interesting_points = set()
	logging.info("The size of the sampled data: "+str(sampled_data.shape[0]))

	for i in range(sampled_data.shape[0]):
 		indexes = np.where(sim_matrix[i,:] > 0.98)[0]
 		disagree_ind = []
		count = np.zeros((2,1))
 		for ind in indexes:
 			if sampled_data[ind,4] == 0:
 				count[0] += 1
 			else:
 				count[1] += 1
 	
 		# selecting points satisfying the minority threshold
 		if count[0] >= count[1]:
 			if count[0] > 0:
 				mt_ratio = count[1]/count[0]
 				if float(mt_ratio) < 0.4: 
 					disagree_ind = [n for n in indexes if (sampled_data[n,4]==1)]
 		else:
 			if count[1] > 0:
 				mt_ratio = count[0]/count[1]
 				if float(mt_ratio) < 0.4:
 					disagree_ind = [n for n in indexes if (sampled_data[n,4]==0)]
 	
		if disagree_ind:
			for idx in disagree_ind:
				interesting_points.add(idx*4)

	logging.info("The number of points of uncertainty chosen by the cosine similarity metric are :" + str(len(interesting_points)))
	return sorted_data, scaled_data, interesting_points

def active_learning_iterations():
	true_data = generate_true_labels('MOCK_DATA') # required for computing the ROC
	original_data = sorted(true_data, key=operator.itemgetter(0))
	original_data = np.asarray(original_data)

	(data, sc_data, points_to_relabel) = subsample_for_sample_selection() # points of interest, original data needed to query from the crowd

	improved_labels = ask_expert(data,points_to_relabel)
	fpr, tpr, thresh = roc_curve(improved_labels, original_data[:,4])
	auc_data = auc(fpr,tpr)
	logging.info("Improving the Base Rule with Crowd Wisdom with AUC: %.4f" % auc_data)

	auc_rf = []
	auc_al = []
	max_iter = 20
	total_samples_queried = set()
	auc_al = []
	auc_rf = []
	num_samples_per5_iter = 0

	while(round(auc_data,2) <= 0.9 and max_iter >= 0):
		samples_to_label = set()
		new_data = np.zeros((sc_data.shape[0],sc_data.shape[1]))
		new_data[:,:-1] = sc_data[:,:-1]
		new_data[:,4] = improved_labels
		data[:,4] = improved_labels
	
		model = RandomForestClassifier()
		param_grid = {"n_estimators": [30, 50, 70, 100],
			  "max_depth": [2,3,4],
              "max_features": [2, 3, 4]
             }

		gs_model = GridSearchCV(model, param_grid=param_grid)
		gs_model.fit(new_data[:,:-1], improved_labels)
		desc_scores = gs_model.predict_proba(new_data[:,:-1])

	
		improved_labels = gs_model.predict(new_data[:,:-1])
		fpr1, tpr1, thresholds = roc_curve(improved_labels, original_data[:,4])
		auc_data = auc(fpr1,tpr1)

		if max_iter % 5 == 0:
			auc_data = round(auc_data, 4)
			# create_plot(new_X[:,:-1],improved_labels, "RDF with AUC : " + str(auc_data))
			auc_rf.append(auc_data)
		logging.info("AUC with RF Classifier: %.4f" % auc_data)

		desc_scores = np.asarray(desc_scores)
		for i in range(desc_scores.shape[0]):
			if(len(samples_to_label) <= 10 and i not in total_samples_queried):
				if abs(desc_scores[i,0]-desc_scores[i,1]) <= 0.7:
					samples_to_label.add(i)
					total_samples_queried.add(i)
		logging.info("The new samples selected to label: %d", len(samples_to_label))
		num_samples_per5_iter += len(samples_to_label)

		improved_labels = ask_expert(data,samples_to_label)
		fpr1, tpr1, thresholds = roc_curve(improved_labels, original_data[:,4])
		auc_data = auc(fpr1,tpr1)
	
		if max_iter % 5 == 0:
			auc_data = round(auc_data, 4)
			# create_plot(new_X[:,:-1], improved_labels, "ALS with AUC : " + str(auc_data)+"\n(Samples Selected for relabelling - "+str(num_samples_per5_iter)+")")
			num_samples_per5_iter = 0 # resetting value of number of samples queried
			auc_al.append(auc_data)
		logging.info("AUC with Crowd Wisdom: %.4f" % auc_data)

		max_iter = max_iter - 1 
	logging.info(auc_al)
	logging.info(auc_rf)

# task instantiation
t0 = BashOperator(
	task_id = 'start',
	bash_command = 'echo "Started" ',
	dag = ml_dag
	)
t1 = PythonOperator(
	task_id =  'base_learner',
	python_callable = base_rule_learner,
	dag = ml_dag
	)

t2 = PythonOperator(
	task_id =  'sub_sampler',
	python_callable = subsample_for_sample_selection,
	dag = ml_dag
	)

t3 = PythonOperator(
 	task_id =  'ml_algo',
 	python_callable = active_learning_iterations,
 	dag = los_dag
 	)


t0.set_downstream(t1)
t1.set_downstream(t2)
t2.set_downstream(t3)

