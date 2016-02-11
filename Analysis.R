options(warn=-1)
args <- commandArgs(TRUE)
dataURL<-as.character(args[1])
header<-as.logical(args[2])
install.packages("e1071")
install.packages("rpart")
install.packages("class")
install.packages("randomForest")
install.packages("caret")
install.packages("adabag")
install.packages("ggplot2")
install.packages("mlbench")
install.packages("lattice")
library(rpart)
library(e1071)
library(nnet)
library(class)
library(caret)
library(randomForest)
library(adabag)
library(ggplot2)
library(lattice)

d<-read.csv(dataURL,header = header,na.strings = c("?"))
d <- na.omit(d)
# create 10 samples
set.seed(123)

for(i in 1:10) {
cat("Running sample ",i,"\n")

#which one is the class attribute
Classcol<-as.integer(args[3])

colname<-colnames(d)[Classcol]
d[,colname]=factor(as.factor(d[,colname]), labels = c("0","1"))
levels(d[,colname]) <- c(0,1)
form<- as.formula(paste(colname,"~.",sep=""))

sampleInstances<-sample(1:nrow(d),size = 0.9*nrow(d))
trainData<-d[sampleInstances,]
testData<-d[-sampleInstances,]

# Decision Tree
model<- rpart(form,trainData,method="class",minbucket=2)
pred <- predict(model,testData,type="class")
accuracy<-mean(pred == testData[ ,Classcol])*100
# example of how to output
method="DT" 
cat("Method = ", method,", accuracy= ", accuracy,"\n")

# SVM
model <- svm(form, data = trainData)
pred<- predict(model,testData,type="class")
accuracy<-mean(pred == testData[ ,Classcol])*100
# example of how to output
method="SVM" 
cat("Method = ", method,", accuracy= ", accuracy,"\n")


#Naive Bayesian
model = naiveBayes(form, data = trainData)
pred = predict(model, testData,type="class")
accuracy<-mean(pred == testData[ ,Classcol])*100
# example of how to output
method="Naive Bayesian" 
cat("Method = ", method,", accuracy= ", accuracy,"\n")

#KNN
model = knn(trainData[,-Classcol], testData[,-Classcol], cl = trainData[,colname], k = 400)
#model = knn(trainData, testData, cl = trainData[,colname], k = 3)
accuracy<-mean(model == testData[ ,Classcol])*100
# example of how to output
method="KNN" 
cat("Method = ", method,", accuracy= ", accuracy,"\n")

#Logistic Regression
model <- glm(form, data = trainData, family = "binomial")
pred <- predict(model,testData,type="response")
threshold=.55
p<-sapply(pred, FUN=function(x) if (x>threshold) 1 else 0)
accuracy<-mean(p == testData[ ,Classcol])*100
# example of how to output
method="Logistic Regression" 
cat("Method = ", method,", accuracy= ", accuracy,"\n")

#Neural Network
model <- nnet(form, trainData,size=4,maxit=10,decay=0.001)
pred<-predict(model,testData,type="class")
accuracy<-mean(pred == testData[ ,Classcol])*100
# example of how to output
method="Neural Network" 
cat("Method = ", method,", accuracy= ", accuracy,"\n")

#Random Forest
model <- randomForest(form, data=trainData)
pred <- predict(model, testData)
accuracy<-mean(pred == testData[ ,Classcol])*100
# example of how to output
method="Random Forests" 
cat("Method = ", method,", accuracy= ", accuracy,"\n")
 
#Bagging
model <- bagging(form,data=trainData,mfinal=15,control=rpart.control(maxdepth=2))
pred <- predict.bagging(model,newdata=testData)
#pred$class
accuracy<-mean(pred$class == testData[ ,Classcol])*100
# example of how to output
method="Bagging" 
cat("Method = ", method,", accuracy= ", accuracy,"\n")

#Boosting
model <- boosting(form,data=trainData,mfinal=10, coeflearn="Freund",boos=FALSE , control=rpart.control(maxdepth=3))
pred <- predict.boosting(model,newdata=testData)
accuracy<-mean(pred$class == testData[ ,Classcol])*100
# example of how to output
method="Boosting" 
cat("Method = ", method,", accuracy= ", accuracy,"\n")

}

