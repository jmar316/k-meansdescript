k-meansdescript
===================

Descriptive Modeling using k-means and spherical k-means methods on yelp.com data
-------------------
Instructions for cluster_nbc.py:
- Code pokes around yelp.com data examining relationships between variables. 
(Includes pre-processing)

- Expected Input:
trainingDataFilename = sys.argv[1]
testDataFilename = sys.argv[2]
classLabelIndex = sys.argv[3]
printTopWords = int(sys.argv[4])
ClusteringTechnique = sys.argv[5]

- Code Execution Example
python cluster_nbc.py stars_data.csv stars_data.csv 7 0 k-means
OR
python cluster_nbc.py stars_data.csv stars_data.csv 7 0 spherical-k-means

