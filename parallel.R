training<-read.csv("pml-training.csv")
testing<-read.csv("pml-testing.csv")
#a<-read.csv("data.csv")
library(parallel)
library(doParallel)
library(caret)
library(e1071)

ntrain <- length(training)
ntest <- length(testing)

#partitioning data
set.seed(100)
inTrain <- createDataPartition(training$classe,p=.7)

#cleaning
trainset <- training[inTrain[[1]],8:160]
testset <- training[-inTrain[[1]],8:160]
testing <- testing[,8:160]
for (i in 152:1){
  if (sum(is.na(trainset[,i])) >= ntrain*0.2 | is.na(mean(trainset[,i],na.rm = TRUE))){
    trainset <- trainset[,-i]
    testset <- testset[,-i]
    testing <- testing[,-i]
  }
}
#preprocessing
summary(prcomp(trainset[,-53]))

train_pre_obj <- preProcess(trainset[,-ncol(trainset)],method = c("center","scale","pca"),pcaComp = 10)
trainset.pre <- predict(train_pre_obj, newdata = trainset[,-ncol(trainset)])
testset.pre <-  predict(train_pre_obj, newdata = testset[ ,-ncol(testset) ])
train.classe <- trainset$classe
test.classe <- testset$classe

#training different models:
nc <- detectCores()
fitControl <- trainControl(method="cv", number=8, allowParallel = TRUE)
#Random Forest:
cluster <- makeCluster(nc)
registerDoParallel(cluster)
m.rf <- train(x=trainset.pre, y=train.classe, method="rf", trControl = fitControl)
stopCluster(cluster)
registerDoSEQ()
#linear discrimant analysis:
m.lda <- train(x=trainset.pre, y=train.classe, method="lda")
#Gradient Boosting Algorithm:
cluster <- makeCluster(nc)
registerDoParallel(cluster)
m.gbm <- train(x=trainset.pre, y=train.classe, method="gbm", trControl = fitControl)
stopCluster(cluster)
registerDoSEQ()
#Support Vector Machine:
m.svm <- svm(x=trainset.pre, y=train.classe, method="svm")


#performance:
rf.predicted <- predict(m.rf,testset.pre)
lda.predicted <- predict(m.lda,testset.pre)
gbm.predicted <- predict(m.gbm,testset.pre)
svm.predicted <- predict(m.svm,testset.pre)



confusionMatrix(test.classe,rf.predicted)$overall
confusionMatrix(test.classe,lda.predicted)$overall
confusionMatrix(test.classe,gbm.predicted)$overall
confusionMatrix(test.classe,svm.predicted)$overall

#stacking predictors
combined<-cbind(rf.predicted,gbm.predicted,svm.predicted)
mc<-train(x=combined,y=test.classe,method="rf", trControl = fitControl)
combined.predicted<-predict(mc,newdata=combined)
confusionMatrix(test.classe,combined.predicted)$overall

#output
newdata<-cbind(predict(m.rf,testset.pre),predict(m.gbm,testset.pre),predict(m.svm,testset.pre))
output<-predict(mc,newdata = newdata)

