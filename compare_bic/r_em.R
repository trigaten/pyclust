"
Reads in agglomeration assignments then proceeds with EM.

Should be only run after r_create_hc.R and python_create_hc.py
"

library(mclust)


"
This function creates the agglomeration tree that can be used to initialize the clustering in mclust
"
partition2mat <- function(part) {
  unqs = unique(part)
  n = length(part)
  counter = 1
  mat = matrix(,nrow=n,ncol=2)
  for (un in unqs) {
    idxs = which(part %in% un)
    if (length(idxs)>1) {
      target = idxs[1]
      idxs = idxs[2:length(idxs)]
      for (idx in idxs) {
        mat[counter,1] = idx
        mat[counter,2] = target
        counter = counter+1
      }
    }
    
  }
  mat <- mat[!rowSums(!is.finite(mat)),]
  return(mat)
}

setwd('C:/Users/Thomas Athey/Documents/Labs/Labs/jovo/clustering/pyclust') #directory where data lies
X <- read.csv(file='embedded_right.csv',header=TRUE,sep=',')
c_file <- read.csv(file='classes.csv',header=TRUE,sep=',')
c <- factor(c_file$x)

setwd('C:/Users/Thomas Athey/Documents/Labs/Labs/jovo/clustering/pyclust/compare_bic') #directory where hierarchical agglomeration results lie

files = c('r_hc_EEE.csv','r_hc_EII.csv','r_hc_VII.csv','r_hc_VVV.csv','python_hc_average.csv','python_hc_complete.csv','python_hc_single.csv','python_hc_ward.csv')
covs = c('VVV','EEE','VVI','VII')
kmax = 19
ks = 1:kmax

for (cov in covs) {
  bics <- matrix(,nrow=length(files),ncol=length(ks))
  fn=1
  for (f in files) {
    agglom_clusters <- read.csv(f,header=T,sep=',')
    for (k in ks) {
      #let's read the agglomeration results then use them to initialize mclust
      agglom = agglom_clusters[,kmax-k+1]
      unqs = unique(agglom)
      label <- 0
      agglom_temp <- agglom
      for (un in unqs) {
        agglom[agglom_temp==un] = label
        label <- label+1
      }
      mat <- partition2mat(agglom)
      init <- list('init'=mat)
      model = Mclust(X,G=k,modelNames=cov,initialization = init,
                     tol = c(1.e-3,sqrt(.Machine$double.eps)),
                     itmax = c(100,.Machine$integer.max),verbose=F)
      if (!is.null(model)) {
        setwd('C:/Users/Thomas Athey/Documents/Labs/Labs/jovo/clustering/pyclust/compare_bic/r_em_params') #*************8
        if (substr(f,1,1)=='r') {
          title = paste('r_',cov,'_',substr(f,6,nchar(f)-4),'_',k,sep='')
        } else {
          title = paste('r_',cov,'_',substr(f,11,nchar(f)-4),'_',k,sep='')
        }
        
        write.csv(model$parameters$pro,file=paste(title,'_weights.csv',sep=''),row.names=F,quote=F)
        write.csv(model$parameters$mean,file=paste(title,'_means.csv',sep=''),row.names=F,quote=F)
        write.csv(model$parameters$variance$sigma,file=paste(title,'_variances.csv',sep=''),row.names=F,quote=F)
        setwd('C:/Users/Thomas Athey/Documents/Labs/Labs/jovo/clustering/pyclust/compare_bic') #*********************
        bics[fn,k] = model$BIC[1]
      }
    }
    fn=fn+1
  }
  fname = paste('r_bic_',cov,'.csv',sep='')
  write.csv(bics,file=fname,row.names = F,col.names = F,quote=F)
}