library(tidyverse)
library(ggplot2)
library(MLTools)
library(caret)
library(GGally)
library(ROCR)

# Load the datasets
fdata = read.table("TrainingSet1001_MIC.dat", sep = ' ', header=TRUE)
validation_data = read.table("ValidationSet999_MIC.dat", sep = ' ', header=TRUE)

#Convert output variable to factor
fdata$Y <- as.factor(fdata$Y)
levels(fdata$Y)

## Exploratory analysis -------------------------------------------------------------------------------------
#For datasets with more than two inputs
ggpairs(fdata,aes(color = Y, alpha = 0.3))
#Function for plotting multiple plots between the output of a data frame
#and the predictors of this output.
PlotDataframe(fdata = fdata, 
              output.name = "Y")


## Divide the data into training and test sets ---------------------------------------------------
set.seed(150) #For replication
#create random 80/20 % split
trainIndex <- createDataPartition(fdata$Y,      #output variable. createDataPartition creates proportional partitions
                                  p = 0.8,      #split probability for training
                                  list = FALSE, #Avoid output as a list
                                  times = 1)    #only one partition
#obtain training and test sets
fTR <- fdata[trainIndex,]
fTS <- fdata[-trainIndex,]
#Create dataset to include model predictions
fTR_eval <- fTR
fTS_eval <- fTS

## Initialize trainControl -----------------------------------------------------------------------
ctrl <- trainControl(method = "cv",                        #k-fold cross-validation
                     number = 10,                          #Number of folds
                     summaryFunction = defaultSummary,     #Performance summary for comparing models in hold-out samples.
                     classProbs = TRUE)                    #Compute class probs in Hold-out samples

#-------------------------------------------------------------------------------------------------
#---------------------------- LOGISTIC REGRESSION MODEL ----------------------------------------------
#-------------------------------------------------------------------------------------------------
## Train model -----------------------------------------------------------------------------------
set.seed(150) #For replication
#Train model using training data
LogReg.fit <- train(form = Y ~ ., #formula for specifying inputs and outputs.
                    data = fTR,               #Training dataset 
                    method = "glm",                   #Train logistic regression
                    preProcess = c("center","scale"), #Center an scale inputs
                    trControl = ctrl,                 #trainControl Object
                    metric = "Accuracy")              #summary metric used for selecting hyperparameters
LogReg.fit          #information about the resampling
summary(LogReg.fit) #detailed information about the fit of the final model
str(LogReg.fit)     #all information stored in the model


## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval$LRprob <- predict(LogReg.fit, type="prob", newdata = fTR) # predict probabilities
fTR_eval$LRpred <- predict(LogReg.fit, type="raw", newdata = fTR) # predict classes 
#test
fTS_eval$LRprob <- predict(LogReg.fit, type="prob", newdata = fTS) # predict probabilities
fTS_eval$LRpred <- predict(LogReg.fit, type="raw", newdata = fTS) # predict classes 

#fichero sin ouput
validation_data_eval <- validation_data
validation_data_eval$LRprob <- predict(LogReg.fit, type="prob", newdata = validation_data) # predict probabilities
validation_data_eval$LRpred <- predict(LogReg.fit, type="raw", newdata = validation_data) # predict classes 


## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$LRpred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$LRpred, 
                fTS_eval$Y, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$Y,       #Real observations
                     fTR_eval$LRprob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$Y,       #Real observations
                     fTS_eval$LRprob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)


#-------------------------------------------------------------------------------------------------
#---------------------------- KNN MODEL  ----------------------------------------------
#-------------------------------------------------------------------------------------------------
set.seed(150) #For replication
#Train knn model model.
#Knn contains 1 tuning parameter k (number of neigbors). Three options:
#  - Train with a fixed parameter: tuneGrid = data.frame(k = 5),
#  - Try with a range of values specified in tuneGrid: tuneGrid = data.frame(k = seq(2,120,4)),
#  - Caret chooses 10 values: tuneLength = 10,
knn.fit = train(form = Y ~ ., #formula for specifying inputs and outputs.
                data = fTR,   #Training dataset 
                method = "knn",
                preProcess = c("center","scale"),
                #tuneGrid = data.frame(k = 5),
                tuneGrid = data.frame(k = seq(1,50,1)),
                #tuneLength = 10,
                trControl = ctrl, 
                metric = "Accuracy")
knn.fit #information about the settings
ggplot(knn.fit) #plot the summary metric as a function of the tuning parameter
knn.fit$finalModel #information about final model trained


## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval$knn_prob <- predict(knn.fit, type="prob" , newdata = fTR) # predict probabilities
fTR_eval$knn_pred <- predict(knn.fit, type="raw" , newdata = fTR) # predict classes 
#test
fTS_eval$knn_prob <- predict(knn.fit, type="prob" , newdata = fTS) # predict probabilities
fTS_eval$knn_pred <- predict(knn.fit, type="raw" , newdata = fTS) # predict classes 


#Plot classification in a 2 dimensional space
Plot2DClass(fTR[,1:10], #Input variables of the model
            fTR$Y,     #Output variable
            knn.fit,#Fitted model with caret
            var1 = "X1", var2 = "X2", #variables that define x and y axis
            selClass = "YES")     #Class output to be analyzed 



## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$knn_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$knn_pred, 
                fTS_eval$Y, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$Y,       #Real observations
                     fTR_eval$knn_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$Y,       #Real observations
                     fTS_eval$knn_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)


#-------------------------------------------------------------------------------------------------
#---------------------------- DECISION TREE ------------------------------------------------------
#-------------------------------------------------------------------------------------------------
library(rpart)
library(rpart.plot)
library(partykit)
set.seed(150) #For replication
#Train decision tree
#rpart contains 1 tuning parameter cp (Complexity parameter). Three options:
#  - Train with a fixed parameter: tuneGrid = data.frame(cp = 0.1),
#  - Try with a range of values specified in tuneGrid: tuneGrid = data.frame(cp = seq(0,0.4,0.05))),
#  - Caret chooses 10 values: tuneLength = 10,

#NOTE: Formula method could be used, but it will automatically create dummy variables. 
# Decision trees can work with categorical variables as theey are. Then, x and y arguments are used
tree.fit <- train(x = fTR[,1:10],  #Input variables.
                  y = fTR$Y,   #Output variable
                  method = "rpart",   #Decision tree with cp as tuning parameter
                  control = rpart.control(minsplit = 5,  # Minimum number of obs in node to keep cutting
                                          minbucket = 5), # Minimum number of obs in a terminal node
                  parms = list(split = "gini"),          # impuriry measure
                  # tuneGrid = data.frame(cp = 0.1), # TRY this: tuneGrid = data.frame(cp = 0.25),
                  #tuneLength = 10,
                  tuneGrid = data.frame(cp = seq(0,0.1,0.0005)),
                  trControl = ctrl, 
                  metric = "Accuracy")
tree.fit #information about the resampling settings
ggplot(tree.fit) #plot the summary metric as a function of the tuning parameter
summary(tree.fit)  #information about the model trained
tree.fit$finalModel #Cuts performed and nodes. Also shows the number and percentage of cases in each node.
#Basic plot of the tree:
plot(tree.fit$finalModel, uniform = TRUE, margin = 0.1)
text(tree.fit$finalModel, use.n = TRUE, all = TRUE, cex = .8)
#Advanced plots
rpart.plot(tree.fit$finalModel, type = 2, fallen.leaves = FALSE, box.palette = "Oranges")
tree.fit.party <- as.party(tree.fit$finalModel)
plot(tree.fit.party)

#Measure for variable importance
varImp(tree.fit,scale = FALSE)
plot(varImp(tree.fit,scale = FALSE))

## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval <- fTR
fTR_eval$tree_prob <- predict(tree.fit, type="prob", newdata = fTR) # predict probabilities
fTR_eval$tree_pred <- predict(tree.fit, type="raw", newdata = fTR) # predict classes 
#test
fTS_eval <- fTS
fTS_eval$tree_prob <- predict(tree.fit, type="prob", newdata = fTS) # predict probabilities
fTS_eval$tree_pred <- predict(tree.fit, type="raw", newdata = fTS) # predict classes 


## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$tree_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$tree_pred, 
                fTS_eval$Y, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$Y,       #Real observations
                     fTR_eval$tree_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$Y,       #Real observations
                     fTS_eval$tree_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)


#-------------------------------------------------------------------------------------------------
#---------------------------- SVM LINEAR ------------------------------------------------------
#-------------------------------------------------------------------------------------------------
library(kernlab)
set.seed(150) #For replication
#Train linear  svm
#svm contains 1 tuning parameter C (Cost). Three options:
#  - Train with a fixed parameter: tuneGrid = data.frame(C = 0.1),
#  - Try with a range of values specified in tuneGrid: tuneGrid = data.frame(cp = seq(0.1,10,0.5)),
#  - Caret chooses 10 values: tuneLength = 10,
svm.fit <- train(form = Y ~ ., #formula for specifying inputs and outputs.
                 data = fTR,   #Training dataset 
                 method = "svmLinear",
                 preProcess = c("center","scale"),
                 # 1) try C=0.1
                 #tuneGrid = data.frame(C = 0.1),
                 # 2) try C=10 and compare with C=0.1
                 #tuneGrid = data.frame(C = 10),
                 # 3) find the optimal value of C
                 #tuneGrid = expand.grid(C = c(0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000)),
                 tuneGrid = data.frame(C = seq(0.0001,0.005,0.001)),
                 #tuneLength = 10,
                 trControl = ctrl, 
                 metric = "Accuracy")
svm.fit #information about the resampling settings
ggplot(svm.fit) + scale_x_log10()
svm.fit$finalModel #information about the model trained
#Plot the svm support vectors:
isupvect <- alphaindex(svm.fit$finalModel)[[1]] #indexes for support vectors
#plot support vectors
ggplot() + geom_point(data = fTR[isupvect,],aes(x = X1, y = X2), color = "red") +
  geom_point(data = fTR[-isupvect,], aes(x = X1, y = X2))


## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval <- fTR
fTR_eval$svm_prob <- predict(svm.fit, type="prob", newdata = fTR) # predict probabilities
fTR_eval$svm_pred <- predict(svm.fit, type="raw", newdata = fTR) # predict classes 
#test
fTS_eval <- fTS
fTS_eval$svm_prob <- predict(svm.fit, type="prob", newdata = fTS) # predict probabilities
fTS_eval$svm_pred <- predict(svm.fit, type="raw", newdata = fTS) # predict classes 

## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$svm_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$svm_pred, 
                fTS_eval$Y, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$Y,       #Real observations
                     fTR_eval$svm_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$Y,       #Real observations
                     fTS_eval$svm_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)



#-------------------------------------------------------------------------------------------------
#---------------------------- SVM RADIAL ------------------------------------------------------
#-------------------------------------------------------------------------------------------------
library(kernlab)
set.seed(150) #For replication
#Train model using training data
#Train radial  svm
#svm contains 2 tuning parameter C (Cost) and sigma. Three options:
#  - Train with a fixed parameter: tuneGrid = data.frame( sigma=100, C=1),
#  - Try with a range of values specified in tuneGrid: tuneGrid = expand.grid(C = seq(0.1,100,length.out = 8), sigma=seq(0.01,50,length.out = 4)),
#  - Caret chooses 10 values: tuneLength = 10,
svm.fit = train(form = Y ~ ., #formula for specifying inputs and outputs.
                data = fTR,   #Training dataset 
                method = "svmRadial",
                preProcess = c("center","scale"),
                tuneGrid = expand.grid(C = c(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30), sigma=c(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)),
                #tuneGrid =  data.frame(sigma = 0.01, C = 0.1),  
                #tuneGrid = expand.grid(C = seq(0.1,1000,length.out = 8), sigma=seq(0.01,50,length.out = 4)),
                #tuneLength = 10,
                trControl = ctrl, 
                metric = "Accuracy")
svm.fit #information about the resampling settings
ggplot(svm.fit) + scale_x_log10()
svm.fit$finalModel #information about the model trained
#Plot the svm support vectors:
isupvect <- alphaindex(svm.fit$finalModel)[[1]] #indexes for support vectors
#plot support vectors
ggplot() + geom_point(data = fTR[isupvect,], aes(x = X1, y = X2), color = "red") +
  geom_point(data = fTR[-isupvect,], aes(x = X1, y = X2))
plot(svm.fit$finalModel, data = as.matrix(predict(svm.fit$preProcess,fTR[,1:2])))


## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval <- fTR
fTR_eval$svm_prob <- predict(svm.fit, type="prob", newdata = fTR) # predict probabilities
fTR_eval$svm_pred <- predict(svm.fit, type="raw", newdata = fTR) # predict classes 
#test
fTS_eval <- fTS
fTS_eval$svm_prob <- predict(svm.fit, type="prob", newdata = fTS) # predict probabilities
fTS_eval$svm_pred <- predict(svm.fit, type="raw", newdata = fTS) # predict classes 

##write.table(fTS_eval$svm_pred ,"TeamName.csv", col.names = FALSE, row.names = FALSE)

## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$svm_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$svm_pred, 
                fTS_eval$Y, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$Y,       #Real observations
                     fTR_eval$svm_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$Y,       #Real observations
                     fTS_eval$svm_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)


#-------------------------------------------------------------------------------------------------
#-------------------------------- MLP ------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
library(NeuralNetTools) ##Useful tools for plotting and analyzing neural networks
library(nnet)
#Train MLP
#mlp contains 2 tuning parameters size and decay Three options:
#  - Train with a fixed parameter: tuneGrid = data.frame(size =5, decay = 0),
#  - Try with a range of values specified in tuneGrid: tuneGrid = expand.grid(size = seq(5,25,length.out = 5), decay=c(10^(-9),0.0001,0.001,0.01,0.1,1,10)),
#  - Caret chooses 10 values: tuneLength = 10,
set.seed(150) #For replication
mlp.fit = train(form = Y ~ ., #formula for specifying inputs and outputs.
                data = fTR,   #Training dataset 
                method = "nnet",
                preProcess = c("center","scale"),
                maxit = 250,    # Maximum number of iterations
                #tuneGrid = data.frame(size =5, decay = 0),
                tuneGrid = expand.grid(size = seq(3,10,1),
                                       decay=c(0.0001,0.001,0.01,0.1,1)),
                trControl = ctrl, 
                metric = "Accuracy")

mlp.fit #information about the resampling settings
ggplot(mlp.fit)+scale_x_log10()

mlp.fit$finalModel #information about the model trained
#summary(mlp.fit$finalModel) #information about the network and weights
plotnet(mlp.fit$finalModel) #Plot the network

#Statistical sensitivity analysis
library(NeuralSens)
SensAnalysisMLP(mlp.fit) 



## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval$mlp_prob = predict(mlp.fit, type="prob" , newdata = fTR) # predict probabilities
fTR_eval$mlp_pred = predict(mlp.fit, type="raw" , newdata = fTR) # predict classes 
#test
fTS_eval$mlp_prob = predict(mlp.fit, type="prob" , newdata = fTS) # predict probabilities
fTS_eval$mlp_pred = predict(mlp.fit, type="raw" , newdata = fTS) # predict classes 


## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$mlp_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$mlp_pred, 
                fTS_eval$Y, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$Y,       #Real observations
                     fTR_eval$mlp_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$Y,       #Real observations
                     fTS_eval$mlp_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)



