k-meansdescript
===================

Descriptive Modeling using k-means and spherical k-means methods on yelp.com data
-------------------
Instructions for cluster_nbc.py:

- Expected Input:
	- trainingDataFilename = sys.argv[1] (training file)
	- testDataFilename = sys.argv[2] (data you wish to run algorithm on)
	- classLabelIndex = sys.argv[3] (classification variable - col. number)
	- printTopWords = int(sys.argv[4]) (binary, if you wish to print top 10 words)
	- ClusteringTechnique = sys.argv[5] ("k-means" or "spherical-k-means")

- Code Execution Example
	- python cluster_nbc.py stars_data.csv stars_data.csv 7 0 k-means
	- python cluster_nbc.py stars_data.csv stars_data.csv 7 0 spherical-k-means

