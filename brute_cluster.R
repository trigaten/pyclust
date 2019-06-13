"
Runs all clustering methods and saves that with the best ari and bic
"

library("mclust")
setwd("C:/Users/Thomas Athey/Documents/Labs/Labs/jovo/clustering/pyclust")

#0-drosophila, 1-BC, 2-diabetes
dataset = 2
savefigs = NULL

if (dataset==0) {
  X <- read.csv(file='data/embedded_right.csv',header=TRUE,sep=',')
  c_file <- read.csv(file='data/classes.csv',header=TRUE,sep=',')
  c <- factor(c_file$x)
  modelNames=mclust.options("emModelNames")
  ks <- 1:19
} else if (dataset==1) {
  #read mean texture, extreme area, and extreme smoothness
  X <- read.csv(file='data/wdbc.data',header=FALSE,sep=',')[,c(4,26,27)]
  c <- read.csv(file='data/wdbc.data',header=FALSE,sep=',')[,2]
  modelNames=c("VVV","EEE","EEV","VEV")
  ks <- 1:9
} else if (dataset==2) {
  #read glucose area, insulin area, and SSPG
  X <- read.csv(file='data/T36.1',header=FALSE,sep = ',')[,c(7,8,9)]
  c <- read.csv(file='data/T36.1',header=FALSE, sep = ',')[,10]
  modelNames=mclust.options("emModelNames")
  ks <- 1:6
}

colors = c('red','blue','green','yellow','brown','black', 'orange',"coral","cyan","darkolivegreen1","gold2","burlywood","gray64","deeppink")

#true labels******************************
plots <- data.frame('x1'=unlist(X[1]),"x2"=unlist(X[2]),"c_true"=c)
if (! is.null(savefigs)) {
  title = paste(savefigs,'_r_true.jpg')
  jpeg(title)
  plot(plots$x1,plots$x2,col=colors[plots$c_true],pch=19,xlab='feature 1',ylab='feature 2',main='True labels')
  dev.off()
} else {
  plot(plots$x1,plots$x2,col=colors[plots$c_true],pch=19,xlab='feature 1',ylab='feature 2',main='True labels')
}

#bicplot********************************
BIC <- mclustBIC(X,ks,verbose=FALSE, modelNames=modelNames)
if (! is.null(savefigs)) {
  title = paste(savefigs,'_r_bicplot.jpg')
  jpeg(title)
  plot(BIC)
  dev.off()
} else {
  plot(BIC)
}

#bicari******************************
ARI = BIC
ARI_models = BIC
l <- length(ks)
model_names <- colnames(BIC)
for(model_num in 0:(length(BIC)-1)) {
  model_name <- model_names[model_num %/% l +1]
  if(!(is.na(BIC[model_num+1]))) {
    k = model_num %% l+1
    model <- Mclust(X,G=k,modelNames=model_name,verbose=FALSE)
    c_hat <-factor(model$classification)
    ARI[model_num+1] = adjustedRandIndex(plots$c_true,c_hat)
    ARI_models[model_num+1] = model_num %/% l +1
  }
}
stats <- data.frame('bic'=BIC[1:266],'ari'=ARI[1:266],'model'=ARI_models[1:266])
linmod <- lm(stats$ari ~ stats$bic)
header <- paste("Mclust's ARI vs BIC on Drosophila Data with Correlation r^2=",format(summary(linmod)$r.squared,digits=3))
if (! is.null(savefigs)) {
  title <- paste(savefigs,'_r_bicari.jpg') 
  jpeg(title)
  plot(stats$bic,stats$ari,col=colors[stats$model],xlab='bic',ylab='ari',main=header)
  dev.off()
} else {
  plot(stats$bic,stats$ari,col=colors[stats$model],xlab='bic',ylab='ari',main=header)
}

#best bic***************************************
model <- Mclust(X,ks,verbose=FALSE,modelNames=modelNames)
best_combo_bic <- model$modelName
best_g_bic <- model$G
best_bic <- max(model$BIC,na.rm=T)
c_hat_bic <- factor(model$classification)

plots$c_hat_bic = c_hat_bic
best_ari_bic = adjustedRandIndex(plots$c_true,plots$c_hat_bic)
print('Best BIC:')
print(best_combo_bic)
print(best_bic)
header <- paste('mclust Best BIC:',format(best_bic,digits=3),'from:',best_combo_bic,', k=',best_g_bic,'ari=',format(best_ari_bic,digits=3),sep=' ')
if (! is.null(savefigs)) {
  title <- paste(savefigs,'_r_bestbic.jpg') 
  jpeg(title)
  plot(plots$x1,plots$x2,col=colors[plots$c_hat_bic],pch=19,xlab='feature 1',ylab='feature 2',main=header)
  dev.off()
} else {
  plot(plots$x1,plots$x2,col=colors[plots$c_hat_bic],pch=19,xlab='feature 1',ylab='feature 2',main=header)
}


#best ari****************************
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
header <- paste('mclust Best ARI:',format(best_ari,digits=3),'from:',best_combo_ari,', k=',best_g_ari,sep=' ')
if (! is.null(savefigs)) {
  title <- paste(savefigs,'_r_bestari.jpg') 
  jpeg(title)
  plot(plots$x1,plots$x2,col=colors[plots$c_hat_ari],pch=19,xlab='feature 1',ylab='feature 2',main=header)
  dev.off()
} else {
  plot(plots$x1,plots$x2,col=colors[plots$c_hat_ari],pch=19,xlab='feature 1',ylab='feature 2',main=header)
}

