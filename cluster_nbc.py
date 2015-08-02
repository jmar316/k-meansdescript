import csv, string, sys, re, random, itertools, numpy, operator
from collections import Counter
from collections import defaultdict
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
from pylab import plot,show

# Jason Martins - jmar316
# More Exploration of Naive Bayes Classifiers using Word Clusters

#####################################
def csv2LISTDICT(file_arg):
	#	Local Variables
	table_list = []				# our result of a list of dictionaries
	temp_row = {}  				# hold our temp rows
	
	#	Begin
	file = open(file_arg)
	data = csv.reader(file)
	
	for row in data:
		if data.line_num == 1: 			# Extract header line to be used in zip later
			file_header = row
		else:
			temp_row = dict(zip(file_header, row))
			
			# Remove punctuation (incl. \n) &  mark lower case
			temp_row["text"] = string.replace(temp_row["text"],'.',' ')
			temp_row["text"] = temp_row["text"].translate(None, string.punctuation)
			temp_row["text"] = temp_row["text"].lower()
			temp_row["text"] = string.replace(temp_row["text"],'\n',' ')
			table_list.append(temp_row)
	
	return table_list
#####################################
def freqCOUNT_kmean(table):

	#	Local Variables
	bag_of_words = ''
	locate_word  = {}
	grp_result_p = defaultdict(list)
	grp_result_np = defaultdict(list)
	j = p = np = 0
	clusters = 50
	
	#	Begin - create one giant string
	for i in range(len(table)):
		bag_of_words = ''.join([table[i]["text"],bag_of_words])
	
	# Create our big bag of words (only grab 201-2200)
	counter_result = Counter(re.findall(r'\w+',bag_of_words)).most_common()[200:2200]

	W_p = numpy.zeros(shape=(2000,2500))
	W_np = numpy.zeros(shape=(2000,2500))

	# Will help us determine what name is what row
	for entry in counter_result:
			locate_word[j] = entry[0]
			j += 1
	
	for i in range(len(table)): # Iterate through all reviews
		#print i, table[i]["stars"],table[i]["text"]		
		if table[i]["stars"] == '5': 				# isPositive = stars = 5
			review_split = grepString(table[i]["text"])
			for key in locate_word:
				#print 'W_p',key,'Count:',review_split.count(key), 'in ',i,'th review'
				W_p[key,p] = review_split.count(locate_word[key])
			p += 1
		else:
			review_split = grepString(table[i]["text"])
			for key in locate_word:
				#print 'W_np',key,'Count:',review_split.count(key), 'in ',i,'th review'
				W_np[key,np] = review_split.count(locate_word[key])
			np += 1
	
	#K-Means 
	centroids_p,variance_p = kmeans(W_p, clusters)
	centroids_np,variance_np = kmeans(W_np, clusters)
	groupings_p,distance_p = vq(W_p,centroids_p)
	groupings_np,distance_np = vq(W_np,centroids_np)
	
	#VQ  Returns:	
	#groupings : ndarray
	#A length M array holding the code book index for each observation.
	#distance : ndarray
	#The distortion (distance) between the observation and its nearest code.
	
	positive_score = numpy.ndarray.sum(distance_p)
	not_positive_score = numpy.ndarray.sum(distance_np)
	print 'Number of clusters:',clusters
	print 'W_p K-Means Score:', round(positive_score,4)
	print 'W_np K-Means Score:', round(not_positive_score,4)
	
	# Create a dictionary of word clusters (key = cluster number i.e 0-49)
	for i in range(0,clusters):
		for j in range(len(groupings_p)):
			if groupings_p[j] == i:
				grp_result_p[i].append(locate_word[j])
			if groupings_np[j] == i:
				grp_result_np[i].append(locate_word[j])
				
	return W_p,W_np,grp_result_p,grp_result_np
	
#####################################
def freqCOUNT_sphericalKmean(table):

	#	Local Variables
	bag_of_words = ''
	locate_word  = {}
	grp_result_p = defaultdict(list)
	grp_result_np = defaultdict(list)
	j = p = np = 0
	clusters = 50
	
	#	Begin - create one giant string
	for i in range(len(table)):
		bag_of_words = ''.join([table[i]["text"],bag_of_words])
	
	# Create our big bag of words (only grab 201-2200)
	counter_result = Counter(re.findall(r'\w+',bag_of_words)).most_common()[200:2200]

	W_p = numpy.zeros(shape=(2000,2500))
	W_np = numpy.zeros(shape=(2000,2500))
	cluster_mtrx_p = numpy.zeros(shape=(clusters,2500))
	cluster_mtrx_np = numpy.zeros(shape=(clusters,2500))

	# Will help us determine what name is what row
	for entry in counter_result:
			locate_word[j] = entry[0]
			j += 1

	
	for i in range(len(table)): # Iterate through all reviews
		#print i, table[i]["stars"],table[i]["text"]		
		if table[i]["stars"] == '5': 				# isPositive = stars = 5
			review_split = grepString(table[i]["text"])
			for key in locate_word:
				#print 'W_p',key,'Count:',review_split.count(key), 'in ',i,'th review'
				W_p[key,p] = review_split.count(locate_word[key])
			p += 1
		else:
			review_split = grepString(table[i]["text"])
			for key in locate_word:
				#print 'W_np',key,'Count:',review_split.count(key), 'in ',i,'th review'
				W_np[key,np] = review_split.count(locate_word[key])
			np += 1
	
	#Step 1: Initialize Clusters
	
	random = numpy.random.randint(2000, size = (clusters,1))	
	
	for i in range(len(random)):
		cluster_mtrx_p[i] = W_p[random[i]]
		cluster_mtrx_np[i] = W_np[random[i]]

	
	#Step 2: Normalize all word data vectors
	
	W_p = normalize(W_p)
	W_np = normalize(W_np)
	
	cluster_mtrx_p = normalize(cluster_mtrx_p)
	cluster_mtrx_np = normalize(cluster_mtrx_np)
	
	cluster_mtrx_p_OLD = numpy.zeros(shape=(clusters,2500))
	cluster_mtrx_np_OLD = numpy.zeros(shape=(clusters,2500))

	groupings_p = numpy.zeros(shape=(1,2000))
	groupings_np = numpy.zeros(shape=(1,2000))
	
	#Step 3: Perform Spherical K-Means
	iterator = 0
	
	while not numpy.array_equal(cluster_mtrx_p,cluster_mtrx_p_OLD):
		iterator += 1
		print iterator
		for i in range(len(W_p)):						#[0, 1, 2, 3, 4, 5, 6, 7]
			result_p = {}
			result_np = {}		 						
			for j in range(len(cluster_mtrx_p)): 		#[0,1,2]
				dot_result_p = numpy.dot(W_p[i],cluster_mtrx_p[j])
				dot_result_np = numpy.dot(W_np[i],cluster_mtrx_np[j])
				result_p[j] = dot_result_p
				result_np[j] = dot_result_np
			# assign that word to cluster
			groupings_p[0][i] = max(result_p.iteritems(), key=operator.itemgetter(1))[0]
			groupings_np[0][i] = max(result_np.iteritems(), key=operator.itemgetter(1))[0]	

		cluster_mtrx_p_OLD = cluster_mtrx_p
		cluster_mtrx_np_OLD = cluster_mtrx_np
		cluster_mtrx_p = numpy.zeros(shape=(clusters,2500))
		cluster_mtrx_np = numpy.zeros(shape=(clusters,2500))
	
		for k in range(len(groupings_p[0])):
			for l in range(clusters):
				if groupings_p[0][k] == l:
					cluster_mtrx_p[l] += W_p[k]
				if groupings_np[0][k] == l:
					cluster_mtrx_np[l] += W_np[k]

		cluster_mtrx_p = normalize(cluster_mtrx_p)
		cluster_mtrx_np = normalize(cluster_mtrx_np)
		#Sum up vectors in that cluster and normalize once again
		if iterator >= 100:
			break
		
	#Step 4: Error
	
	# Iterate through all word vectors and compare similarity to its cluster
	# Summing that value over all words (with their associated cluster centers) 
	# would give you an objective measure of how close the cluster fit to the data.
	
	# Reference index of groupings_p and associate that with the index of W_p
	# Perform dot product (cosine) with cluster_mtrx and W_p (sum across p and np)
	
	positive_score = 0
	not_positive_score = 0
		
	for i in range(len(groupings_p[0])):
		cluster_p = groupings_p[0][i] # index to reference in cluster_mtrx_p
		cluster_np = groupings_np[0][i]
		positive_score +=  numpy.dot(W_p[cluster_p],cluster_mtrx_p[cluster_p])
		not_positive_score += numpy.dot(W_np[cluster_np],cluster_mtrx_np[cluster_np])
	
	print 'Number of clusters:',clusters	
	print 'W_p Spherical K-Means Score:', round(positive_score,4)
	print 'W_np Spherical K-Means Score:', round(not_positive_score,4)


	# Create a dictionary of word clusters (key = cluster number i.e 0-49)
	for i in range(0,clusters):
		for j in range(len(groupings_p[0])):
			if groupings_p[0][j] == i:
				grp_result_p[i].append(locate_word[j])
			if groupings_np[0][j] == i:
				grp_result_np[i].append(locate_word[j])
	
	return W_p,W_np,grp_result_p,grp_result_np
	
#####################################
def normalize(matrix):
	
	matrix += 0.000000001 # Laplace Smoothing 
	
	for i in range(len(matrix)):
		matrix[i] = matrix[i]/(numpy.linalg.norm(matrix[i]))
		#print numpy.linalg.norm(matrix[i])
	
	return matrix

#####################################
				
def assembleBINARY(train_table,train_grpngs_p,train_grpngs_np):
	#	Local Variable
	binFeature_dict  = defaultdict(list)
	cluster = 50
	binfeat_matrix_pos = numpy.zeros(shape=(5000,cluster))
	binfeat_matrix_neg = numpy.zeros(shape=(5000,cluster))
	clusters = 2*cluster

	
	#	Begin - create a dictionary with 2000 keys -> top words
	#   	each key will map to a list to hold the binary features
		
	# Now search across the training table and assemble features
	# T_p -> 0,49 ... T_np -> 50,99
	for i in range(len(train_table)):
		review_vector = grepString(train_table[i]["text"])
		for k in train_grpngs_p.keys():
			if len(set(train_grpngs_p[k]).intersection(review_vector)) > 0 :
				binfeat_matrix_pos[i][k] = 1
		for k in train_grpngs_np.keys():
			if len(set(train_grpngs_np[k]).intersection(review_vector)) > 0 :
				binfeat_matrix_neg[i][k] = 1
	
	result_mtrx = numpy.concatenate((binfeat_matrix_pos,binfeat_matrix_neg),axis=1)

	cluster_string = 'cluster'
	
	#	Convert matrix into dictionary
	#   key = cluster1, cluster2, etc... value = list of binary results
	for j in range(clusters):
		key = "{}{}".format("cluster", j)
		binFeature_dict[key] = numpy.ndarray.tolist(result_mtrx.T[j])	
	
	
	#This will hold key -> word and a value 
	# -> list of whether the word appeared or not in a given review
	return binFeature_dict
	
#####################################	
def grepString(text):
	# Will take a review and return all words in that review, separated
	review_vector = re.findall(r'\w+',text,flags=re.I)
	return review_vector

#####################################
def assembleBINARY_classTask(train_table,feature):
	#	Local Variable
	binFeature_dict  = defaultdict(list)
	
	#	Begin - create a dictionary with 1 key -> isFunny or isPositive
	#   	each key will map to a list to hold the binary features
	if feature == 5:
		binFeature_dict['isFunny'] = [0] * len(train_table)
		
		# Now search across the training table and assemble features
		for i in range(len(train_table)):
			#for k in binFeature_dict.keys():
			if int(train_table[i]['funny']) > 0:
				binFeature_dict['isFunny'][i] = 1
	else:
		binFeature_dict['isPositive'] = [0] * len(train_table)	
		
		# Now search across the training table and assemble features
		for i in range(len(train_table)):
			#for k in binFeature_dict.keys():
			if int(train_table[i]['stars']) == 5:
				binFeature_dict['isPositive'][i] = 1			
	
	return binFeature_dict

#####################################
def calcNBC(testData,learn_classTaskFeature,learn_bag0wordsFeature,test_classTaskFeature,test_bag0wordsFeature):
	#	Local Variable(s)
	classTask_word = defaultdict(dict) 	# - creating a dictionary of dictionaries 
										# - to hold p(word1|isFunny), p(word2|isFunny)
	p_classTask  = defaultdict(dict)   	# - hold p(isFunny) values	
	prediction_classTask = defaultdict(list) # to hold our predicted values
	classTask = ''
	yes_total = no_total = 0
	incorrect_classifications = float(0)
	
	# Begin
	###########################################
	# Prepare p(isFunny or isPositive) counts #
	###########################################
	
	if 'isFunny' in learn_classTaskFeature:	# will be faster and use dict hashing & no linear
		classTask = 'isFunny'
	else:
		classTask = 'isPositive'
	
	prediction_classTask[classTask] = [0] * len(test_classTaskFeature[classTask])
	#prediction_classTask[classTask] = [0] * len(testData)	
	p_classTask[classTask]['yes'] = float(0)
	p_classTask[classTask]['no'] = float(0)
	
	for i in range(len(learn_classTaskFeature[classTask])):
		if learn_classTaskFeature[classTask][i] == 1:
			p_classTask[classTask]['yes'] +=1
			yes_total += 1
		else:
			p_classTask[classTask]['no'] +=1
			
	no_total = len(learn_classTaskFeature[classTask]) - yes_total
	
	p_classTask[classTask]['yes'] /= len(learn_classTaskFeature[classTask])
	p_classTask[classTask]['no']  /= len(learn_classTaskFeature[classTask])
	
	p_classTask[classTask]['countYES'] = yes_total
	p_classTask[classTask]['countNO'] = no_total
	
	
	###########################################
	# p(word1|isFunny)...p(word2|isFunny).... #
	###########################################
	
	# Reference
	# yes_yes = isFunny = 1 & word = 1  
	# yes_no  = isFunny = 1 & word = 0
	# no_yes = isFunny = 0 & word = 1
	# no_no = isFunny = 0 & word = 0 
	
	for word_key in learn_bag0wordsFeature.iterkeys():
		classTask_word[word_key]['yes_yes'] = float(0)
		classTask_word[word_key]['yes_no'] = float(0)
		classTask_word[word_key]['no_yes'] = float(0)
		classTask_word[word_key]['no_no'] = float(0)
		
		for i in range(len(learn_classTaskFeature[classTask])):
			if learn_classTaskFeature[classTask][i] == 1:
				if learn_bag0wordsFeature[word_key][i] == 1:
					classTask_word[word_key]['yes_yes'] += 1
				else:
					classTask_word[word_key]['yes_no'] += 1
			else:
				if learn_bag0wordsFeature[word_key][i] == 1:
					classTask_word[word_key]['no_yes'] += 1
				else:
					classTask_word[word_key]['no_no'] += 1
		
		#Laplace Correction
		
		#Numerator
		classTask_word[word_key]['yes_yes'] += 1
		classTask_word[word_key]['yes_no'] += 1
		classTask_word[word_key]['no_yes'] += 1
		classTask_word[word_key]['no_no'] += 1
		
		#Denominator
		classTask_word[word_key]['yes_yes'] /= (p_classTask[classTask]['countYES']+2)			
		classTask_word[word_key]['yes_no'] /= (p_classTask[classTask]['countYES']+2)		
		classTask_word[word_key]['no_yes'] /= (p_classTask[classTask]['countNO']+2)
		classTask_word[word_key]['no_no'] /= (p_classTask[classTask]['countNO']+2)
	
	for i in range(len(test_classTaskFeature[classTask])):
		rolling_probClassTRUE = rolling_probClassFALSE = float(1)
		
		for key in test_bag0wordsFeature.iterkeys():
			if test_bag0wordsFeature[key][i] == 1:
				rolling_probClassTRUE *= classTask_word[key]['yes_yes']
				rolling_probClassFALSE *= classTask_word[key]['no_yes']
			else:
				rolling_probClassTRUE *= classTask_word[key]['yes_no']
				rolling_probClassFALSE *= classTask_word[key]['no_no']
		
		rolling_probClassTRUE *= p_classTask[classTask]['yes']
		rolling_probClassFALSE *= p_classTask[classTask]['no']

		if rolling_probClassTRUE >= rolling_probClassFALSE:
			prediction_classTask[classTask][i] = 1
		else:	
			prediction_classTask[classTask][i] = 0
	
	for i in range(len(test_classTaskFeature[classTask])):
		if prediction_classTask[classTask][i] != test_classTaskFeature[classTask][i]:
			incorrect_classifications += 1.0 
	
	# Calculating Zero-One Loss (incorrect/ total trials)
	zero_one_loss = incorrect_classifications/float(len(test_classTaskFeature[classTask]))

	return zero_one_loss

######## Main #########################################################################

#### BEGIN ################
# Capture arguments
trainingDataFilename = sys.argv[1]
testDataFilename = sys.argv[2]
classLabelIndex = sys.argv[3]
printTopWords = int(sys.argv[4])

#######################################################################################

# Training Set Processing (going to print out top words for only training set)

train_table_list = csv2LISTDICT(trainingDataFilename)

#	Standard K-Means (output T) ######################################################
if sys.argv[5] == 'k-means':
	print 'Performing K-Means'
	train_Wp,train_Wnp,train_grpngs_p,train_grpngs_np = freqCOUNT_kmean(train_table_list)
	feat_mtrxPOSNEG = assembleBINARY(train_table_list,train_grpngs_p,train_grpngs_np) 
#	Spherical K-Means (output T) ######################################################
else:
	print 'Performing Spherical K-Means'
	train_Wp,train_Wnp,train_grpngs_p,train_grpngs_np = freqCOUNT_sphericalKmean(train_table_list)
	feat_mtrxPOSNEG = assembleBINARY(train_table_list,train_grpngs_p,train_grpngs_np)

#######################################################################################

# Cross Validation with entire dataset 
# incremental 10-fold cross validation

training_classification_feature = assembleBINARY_classTask(train_table_list,7)
tss = [100,250,500,1000,2000]
t = {}

t['range1'] = range(0,500)
t['range2'] = range(500,1000)
t['range3'] = range(1000,1500)
t['range4'] = range(1500,2000)
t['range5'] = range(2000,2500)
t['range6'] = range(2500,3000)
t['range7'] = range(3000,3500)
t['range8'] = range(3500,4000)
t['range9'] = range(4000,4500)
t['range10'] = range(4500,5000)
 
for tss_count in tss:
	print 'tss value:', tss_count
	results = []
	for int_range in t.iterkeys():  # t = [1..5..10]
		#print 'test_set will be set to:', int_range
	
		# Purge and declare
		test_clasification_feature = {}
		test_clasification_feature['isPositive'] = []
		test_cluster_features = defaultdict(list)
	
		train_clasification_feature = {}
		train_clasification_feature['isPositive'] = []
		train_cluster_features = defaultdict(list)

		s_complement_classification = {}
		s_complement_classification['isPositive'] = []
		s_complement_cluster_feature = defaultdict(list)

		round_key = int_range
	
		# Create test_set = St
		for i in t[int_range]:
			test_clasification_feature['isPositive'].append(training_classification_feature['isPositive'][i])
			for key in feat_mtrxPOSNEG.iterkeys():
				test_cluster_features[key].append(feat_mtrxPOSNEG[key][i])
	
		# Create Sc = S - St
		for range_complement in t.iterkeys():
			if range_complement != round_key:
				for i in t[range_complement]:
					s_complement_classification['isPositive'].append(training_classification_feature['isPositive'][i])
					for key in feat_mtrxPOSNEG.iterkeys():
						s_complement_cluster_feature[key].append(feat_mtrxPOSNEG[key][i]) 

		#for i in tss:
		population = range(len(s_complement_cluster_feature['cluster1']))
		random_indexes = random.sample(population,tss_count)
		for j in random_indexes:
			train_clasification_feature['isPositive'].append(s_complement_classification['isPositive'][j])
			for key in s_complement_cluster_feature.iterkeys():
				train_cluster_features[key].append(s_complement_cluster_feature[key][j])
		

		results.append(calcNBC(train_table_list,train_clasification_feature,train_cluster_features,
									test_clasification_feature,test_cluster_features))
	
	
#Report the average performance in Zero One Loss over the ten-fold cross validation 
#trials and the associated standard error. The standard error is the standard deviation 
#divided by the square root of the number of trials
	
	print 'Zero Loss Average:',round(numpy.average(results),4), \
	'Std Dev:',round(numpy.std(results),4), \
	'Standard Error:',round((numpy.std(results))/(numpy.sqrt(10)),4)

######## End of Main #########################################################################






