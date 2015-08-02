#Jason Martins - jmar316
#Analysis of cluster_nbc.py results on yelp.com data

#Error bar function
#(Credit: https://bmscblog.wordpress.com/2013/01/23/error-bars-with-r/)
add.error.bars <- function(X,Y,SE,w,col=1){
  X0 = X; Y0 = (Y-SE); X1 =X; Y1 = (Y+SE);
  arrows(X0, Y0, X1, Y1, code=3,angle=90,length=w,col=col);
}

##############################################################################
#Inspects the clustering results, comparing standard kmeans to spherical kmeans
##############################################################################
cluster_size = c(10,20,50,100,200)
standard_k_wp = c(11779.2852635,11683.2107161,11406.6710459,10995.3908437,10254.4337)
standard_k_wnp = c(15435.6311279,15291.6361499,14889.1096816,14360.1424529,13401.2289)

spherical_k_wp = c(424.6356,376.1951,268.5646,238.8279,266.7283)
spherical_k_wnp = c(533.2919,429.8896,362.0934,266.7283,201.183)

#Standard K-Means
plot(cluster_size,standard_k_wp,ylim=range(c(standard_k_wp,standard_k_wnp)),type="b",
     main = "Cluster Size vs. Standard K-Means Score",ylab = "Std. K-Means Score (red=Positive, blue=Not Positive)",
     xlab="Number of Clusters", xlim=range(cluster_size),col = "red")
par(new = TRUE)
plot(cluster_size, standard_k_wnp, ylim=range(c(standard_k_wp,standard_k_wnp)), axes = FALSE, 
     xlab = "", ylab = "",type="b",  xlim=range(cluster_size),col = "blue")

#Spherical K-Means
plot(cluster_size,spherical_k_wp,ylim=range(c(spherical_k_wp,spherical_k_wnp)),type="b",
     main = "Cluster Size vs. Spherical K-Means Score",ylab = "Sph. K-Means Score (red=Positive, blue=Not Positive)",
     xlab="Number of Clusters", xlim=range(cluster_size),col = "red")
par(new = TRUE)
plot(cluster_size, spherical_k_wnp, ylim=range(c(spherical_k_wp,spherical_k_wnp)), axes = FALSE, 
     xlab = "", ylab = "",type="b",  xlim=range(cluster_size),col = "blue")
##############################################################################

##############################################################################
#Does clustering approach improves performance? Examine 0/1 loss across
##############################################################################
tss = c(100,250,500,1000,2000)
spherical_kmeans_zeroLOSS = c(0.4072,0.372,0.365,0.3708,0.3726)
standard_kmeans_zeroLOSS = c(0.4278,0.4392,0.396,0.3752,0.3744)
spherical_kmeans_stdERROR = c(0.0228,0.0188,0.0197,0.019,0.0188)
standard_kmeans_stdERROR = c(0.0371,0.0279,0.0209,0.0198,0.0198)

plot(tss,standard_kmeans_zeroLOSS,ylim=range(c(0.47,0.33)),type="b",
     main = "TSS vs. K-Means 0/1 Loss",ylab = "0/1 Loss (red=standard, blue=spherical)",
     xlab="Value of TSS", xlim=range(c(tss)),col = "red")
par(new = TRUE)
plot(tss,spherical_kmeans_zeroLOSS,ylim=range(c(0.47,0.33)), axes = FALSE, 
     xlab = "", ylab = "",type="b",  xlim=range(c(tss)),col = "blue")

add.error.bars(tss,standard_kmeans_zeroLOSS,standard_kmeans_stdERROR,0.1,col="red")
add.error.bars(tss,spherical_kmeans_zeroLOSS,spherical_kmeans_stdERROR,0.1,col="blue")
##############################################################################

##############################################################################
#Assess whether using the topic features improves performance
##############################################################################
tss = c(100,250,500,1000,2000)
spherical_kmeans_zeroLOSS = c(0.4072,0.372,0.365,0.3708,0.3726)
hw3_zeroLOSS = c(0.4988,0.4968,0.4988,0.498,0.4964)
hw3_stdERROR = c(0.156,0.1555,0.1555,0.1561,0.1562)
spherical_kmeans_stdERROR = c(0.0228,0.0188,0.0197,0.019,0.0188)

plot(tss,hw3_zeroLOSS,ylim=range(c(0.67,0.33)),type="b",
     main = "Sphr. K-Means vs HW3 NBC 0/1 Loss",ylab = "0/1 Loss (red=HW3, blue=spherical)",
     xlab="Value of TSS", xlim=range(c(tss)),col = "red")
par(new = TRUE)
plot(tss,spherical_kmeans_zeroLOSS,ylim=range(c(0.67,0.33)), axes = FALSE, 
     xlab = "", ylab = "",type="b",  xlim=range(c(tss)),col = "blue")

add.error.bars(tss,hw3_zeroLOSS,hw3_stdERROR,0.1,col="red")
add.error.bars(tss,spherical_kmeans_zeroLOSS,spherical_kmeans_stdERROR,0.1,col="blue")
##############################################################################

##############################################################################
#Assess whether the number of features improves performance
##############################################################################
tss = c(100,250,500,1000,2000)
hw3_100_2000_zeroLOSS = c(0.5016,0.5022,0.5006,0.4984,0.4998)
spherical_kmeans_zeroLOSS = c(0.4072,0.372,0.365,0.3708,0.3726)
combination_zeroLOSS = c(0.4596,0.3858,0.3666,0.3626,0.3646)
hw3_100_2000_stdERROR = c(0.1561,0.1559,0.1558,0.1558,0.1559)
spherical_kmeans_stdERROR = c(0.0228,0.0188,0.0197,0.019,0.0188)
combination_stdERROR = c(0.0546,0.0227,0.0189,0.0248,0.0213)


plot(tss,hw3_100_2000_zeroLOSS,ylim=range(c(0.67,0.33)),type="b",
     main = "Comparing # of features",ylab = "0/1 Loss (r=100HW3, b=100Sph, g=combo)",
     xlab="Value of TSS", xlim=range(c(tss)),col = "red")
par(new = TRUE)
plot(tss,spherical_kmeans_zeroLOSS,ylim=range(c(0.67,0.33)), axes = FALSE, 
     xlab = "", ylab = "",type="b",  xlim=range(c(tss)),col = "blue")
par(new = TRUE)
plot(tss,combination_zeroLOSS,ylim=range(c(0.67,0.33)), axes = FALSE, 
     xlab = "", ylab = "",type="b",  xlim=range(c(tss)),col = "green")

add.error.bars(tss,hw3_100_2000_zeroLOSS,hw3_100_2000_stdERROR,0.1,col="red")
add.error.bars(tss,spherical_kmeans_zeroLOSS,spherical_kmeans_stdERROR,0.1,col="blue")
add.error.bars(tss,combination_zeroLOSS,combination_stdERROR,0.1,col="green")

                