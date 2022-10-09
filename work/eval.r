
load('result_norm.rda')
tm <- true_alpha[1]; tv <- true_alpha[2]   ## ターゲット分布のパラメータ
KLdiv <- function(alpha){ ## KL-div計算(ターゲット vs. 推定結果)
  m <- alpha[1]; v <- alpha[2]
  return((log(v/tv)+ (tv+(m-tm)^2)/v -1)/2)
}

KLmat <- c()
for (ii in 1:10){
  KLtmp <- c()
  for (i in 1:iterate){
    KLtmp  <- c(KLtmp, KLdiv(res[[ii]][i,]))
  }
  KLmat <- rbind(KLmat,KLtmp) 
}

## ## 各学習プロセスでの KL-div の値をプロット
## plot(KLmat[1,],col=1,type='l',ylim=c(0, 0.15))
## for (ii in 2:10){
##   lines(KLmat[ii,],col=ii)
## }

est <- c()
for(ii in 1:10){
  est <- rbind(est,c(res[[ii]][500,1], sqrt(res[[ii]][500,2])))
}
dimnames(est)[[2]] <- c('mean','var')
error <- rowMeans(abs(t(est)-c(tm,tv)))

## 表示
cat('true mean: ',tm, ', true var',tv,'\n')
cat('estimated values\n')
print(est)
cat('\naveraged error of mean[iterate]: ',error[1],'\naveraged error of var[iterate]:  ', error[2],'\n',sep='')


## est <- c()
## for(ii in 1:10){
##   cat('\n ==',ii,'==\n')
##   cat('error of mean[1]: ',abs(res[[ii]][1,1]-true_alpha[1]),',  error of mean[iterate]: ',abs(res[[ii]][iterate,1]-true_alpha[1]),'\n')
##   cat('error of  var[1]: ',abs(res[[ii]][1,2]-true_alpha[2]),',  error of  var[iterate]: ',abs(res[[ii]][iterate,2]-true_alpha[2]),'\n')
##   ##print(abs(res[[ii]][,1]-true_alpha[1])[c(1,500)])
##   ##print(abs(sqrt(res[[ii]][,2])-sqrt(true_alpha[2]))[c(1,500)])
##   est <- rbind(est,c(res[[ii]][500,1], sqrt(res[[ii]][500,2])))
## }

## plot(abs(t(res[[1]])-true_alpha)[1,],col=1,type='l',ylim=c(0, 0.3))
## for (ii in 2:10){
##   lines(abs(t(res[[ii]])-true_alpha)[1,],col=ii)
## }


## ii <- 3
## par(mfrow = c(1,2))
## plot(abs(t(res[[ii]])-true_alpha)[1,])
## plot(abs(t(res[[ii]])-true_alpha)[2,])

