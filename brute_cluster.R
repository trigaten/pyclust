"
Runs all clustering methods and saves that with the best ari and bic
"

library("mclust")

setwd('C:/Users/Thomas Athey/Documents/Labs/Labs/jovo/clustering/drosophila_python')
X <- read.csv(file='embedded_right.csv',header=TRUE,sep=',')
c_file <- read.csv(file='classes.csv',header=TRUE,sep=',')
c <- factor(c_file$x)

#true labels
plots <- data.frame('x1'=unlist(X[1]),"x2"=unlist(X[2]),"c_true"=c)
#jpeg('true.jpg')
plot(plots$x1,plots$x2,col=c('red','blue','green','yellow')[plots$c_true],pch=19,xlab='feature 1',ylab='feature 2',main='True labels')
#dev.off()

#bicplot
ks <- 6
BIC <- mclustBIC(X,ks,verbose=FALSE)
#jpeg('r_bicplot.jpg')
plot(BIC)
#dev.off()



#find best bic
model <- Mclust(X,ks,verbose=FALSE)
best_combo_bic <- model$modelName
best_g_bic <- model$G
best_bic <- max(model$BIC,na.rm=T)
c_hat_bic <- factor(model$classification)

write.csv(c_hat_bic,file='r_k6_cluster.csv',row.names=F,quote=F)
write.csv(model$parameters$pro,file='r_k6_weights.csv',row.names=F,quote=F)
write.csv(model$parameters$mean,file='r_k6_means.csv',row.names=F,quote=F)
write.csv(model$parameters$variance$sigma,file='r_k6_variances.csv',row.names=F,quote=F)

plots$c_hat_bic = c_hat_bic
best_ari_bic = adjustedRandIndex(plots$c_true,plots$c_hat_bic)
print('Best BIC:')
print(best_combo_bic)
print(best_bic)
title <- paste('mclust Best BIC:',format(best_bic,digits=3),'from:',best_combo_bic,', k=',best_g_bic,'ari=',format(best_ari_bic,digits=3),sep=' ')
#********************filename and colors*******
#jpeg('r_bic_allk.jpg')
plot(plots$x1,plots$x2,col=c('red','blue','green','yellow','brown','black')[plots$c_hat_bic],pch=19,xlab='feature 1',ylab='feature 2',main=title)
#dev.off()



#find best ari
models <- mclust.options('emModelNames')
best_ari <- 0

for(k in ks) {
  for(m in models) {
    model <- Mclust(X,G=k,modelNames=m,verbose=FALSE)
    if (is.null(model)) {
      next
    }
    c_hat <-factor(model$classification)
    ari = adjustedRandIndex(plots$c_true,c_hat)
    if(ari>best_ari) {
      best_ari <- ari
      best_combo_ari <- m
      c_hat_ari <- factor(model$classification)
      best_g_ari <- k
    }
  }
}

plots$c_hat_ari = c_hat_ari
print('Best ARI:')
print(best_combo_ari)
print(best_ari)
title <- paste('mclust Best ARI:',format(best_ari,digits=3),'from:',best_combo_ari,', k=',best_g_ari,sep=' ')
#****************filname and colors********8
#jpeg('r_ari_k6.jpg')
plot(plots$x1,plots$x2,col=c('red','blue','green','yellow','brown','black')[plots$c_hat_ari],pch=19,xlab='feature 1',ylab='feature 2',main=title)
#dev.off()
