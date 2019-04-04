"
Thomas Athey 3/20/19
Runs several different agglomeration methods and saves their results, analagous to python_create_hc.py
"

library(mclust)

setwd('C:/Users/Thomas Athey/Documents/Labs/Labs/jovo/clustering/drosophila_python')
X <- read.csv(file='embedded_right.csv',header=TRUE,sep=',')
c_file <- read.csv(file='classes.csv',header=TRUE,sep=',')
c <- factor(c_file$x)

setwd('C:/Users/Thomas Athey/Documents/Labs/Labs/jovo/clustering/pyclust/compare_bic')

models <- c('EII','VII','EEE','VVV')
ks <- 19:1

for (model in models) {
  print(model)
  mat <- hc(X,modelName=model)
  
  n = nrow(X)
  c_agglom = 1:n
  k_level = n
  
  agglom <- data.frame(V1=1:n)
  
  for (k in ks) {
    print(k)
    while (k_level > k) {
      a = mat[1,n-k_level+1]
      b = mat[2,n-k_level+1]
      c_agglom[c_agglom == b] = a
      k_level = k_level-1
    }
    agglom[,length(ks)-k+1] = c_agglom
  }
  fname = paste('r_hc_',model,'.csv',sep='')
  write.csv(agglom,file=fname,row.names = F,col.names = F,quote=F)
}