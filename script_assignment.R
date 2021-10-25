library(tidyverse)
library(ggplot2)
library(MLTools)
library(caret)
library(GGally)

# Step 1 - Import the dataset
diabetes = read.table("Diabetes.csv", sep = ';', header=TRUE)

# Step 2 - check out missing values
summary(diabetes)
# No existen NAs
diabetes$DIABETES = factor(diabetes$DIABETES)

# Step 3 - Plot the data and check for outliers
# showing the data without any processing
ggpairs(diabetes, aes(color=DIABETES))

# create a dataset with no zero values for the columns glusose, bloodpress, skinthickness, insulin & bodymassindex
diabetes_noCeros = filter(diabetes, GLUCOSE != 0, BLOODPRESS != 0, SKINTHICKNESS != 0, INSULIN != 0, BODYMASSINDEX != 0) 

# create two dataframes, one with the diabetics and the other with the no diabetics
diabetes_SI = filter(diabetes_noCeros, DIABETES == 1)
diabetes_NO = filter(diabetes_noCeros, DIABETES == 0)

diabetes_tidy = diabetes

# Now, we are going to replace the zeros with the median of this column depending if the patient is diabetic or no
# zero values treatment for the GLUCOSE column
count(diabetes_tidy, GLUCOSE == 0)
diabetes_tidy$GLUCOSE[diabetes_tidy$GLUCOSE == 0 & diabetes_tidy$DIABETES == 0] = median(diabetes_NO$GLUCOSE)
diabetes_tidy$GLUCOSE[diabetes_tidy$GLUCOSE == 0 & diabetes_tidy$DIABETES == 1] = median(diabetes_SI$GLUCOSE)

# zero values treatment for the BLOODPRESS column
count(diabetes_tidy, BLOODPRESS == 0)
diabetes_tidy$BLOODPRESS[diabetes_tidy$BLOODPRESS == 0 & diabetes_tidy$DIABETES == 0] = median(diabetes_NO$BLOODPRESS)
diabetes_tidy$BLOODPRESS[diabetes_tidy$BLOODPRESS == 0 & diabetes_tidy$DIABETES == 1] = median(diabetes_SI$BLOODPRESS)

# zero values treatment for the SKINTHICKNESS column
count(diabetes_tidy, SKINTHICKNESS == 0)
diabetes_tidy$SKINTHICKNESS[diabetes_tidy$SKINTHICKNESS == 0 & diabetes_tidy$DIABETES == 0] = median(diabetes_NO$SKINTHICKNESS)
diabetes_tidy$SKINTHICKNESS[diabetes_tidy$SKINTHICKNESS == 0 & diabetes_tidy$DIABETES == 1] = median(diabetes_SI$SKINTHICKNESS)

# zero values treatment for the INSULIN column
count(diabetes_tidy, INSULIN == 0)
diabetes_tidy$INSULIN[diabetes_tidy$INSULIN == 0 & diabetes_tidy$DIABETES == 0] = median(diabetes_NO$INSULIN)
diabetes_tidy$INSULIN[diabetes_tidy$INSULIN == 0 & diabetes_tidy$DIABETES == 1] = median(diabetes_SI$INSULIN)

# zero values treatment for the BODYMASSINDEX column
count(diabetes_tidy, BODYMASSINDEX == 0)
diabetes_tidy$BODYMASSINDEX[diabetes_tidy$BODYMASSINDEX == 0 & diabetes_tidy$DIABETES == 0] = median(diabetes_NO$BODYMASSINDEX)
diabetes_tidy$BODYMASSINDEX[diabetes_tidy$BODYMASSINDEX == 0 & diabetes_tidy$DIABETES == 1] = median(diabetes_SI$BODYMASSINDEX)

# plotting the data after the treatment done
ggpairs(diabetes_tidy, aes(color=DIABETES))

# We analyze the 99 value of the SKINTHICKNESS column
# It does not make sense, 99 is a strange value, it could be a measure error
# With that value of SKINTHICKNESS, the patient must also have a high BODYMASSINDEX
# So the decision is to drop that row from the table
diabetes_tidy = filter(diabetes_tidy, SKINTHICKNESS < 90)

# Step 4 - Encode the categorical variables
diabetes_tidy$DIABETES <- as.factor(diabetes_tidy$DIABETES)
levels(diabetes_tidy$DIABETES)

# Step 5 - Analyze the continuous variables: relations between variables
ggpairs(diabetes_tidy, aes(color=DIABETES))
PlotDataframe(diabetes_tidy, output.name = "DIABETES")

# Step 6 - check out for class imbalances
table(diabetes_tidy$DIABETES)
prop.table(table(diabetes_tidy$DIABETES))

# step 7 - data partition
# converting DIABETES column to make the values YES & NO instead of 1 & 0
levels(diabetes_tidy$DIABETES) = c("NO", "YES")

## Divide the data into training and test sets ---------------------------------------------------
set.seed(150) #For replication
#create random 80/20 % split
trainIndex <- createDataPartition(diabetes_tidy$DIABETES,      #output variable. createDataPartition creates proportional partitions
                                  p = 0.8,      #split probability for training
                                  list = FALSE, #Avoid output as a list
                                  times = 1)    #only one partition
#obtain training and test sets
fTR <- diabetes_tidy[trainIndex,]
fTS <- diabetes_tidy[-trainIndex,]
inputs <- 1:(ncol(fTR) - 1)
#Create dataset to include model predictions
fTR_eval <- fTR
fTS_eval <- fTS

## Initialize trainControl -----------------------------------------------------------------------
ctrl <- trainControl(method = "cv",                        #k-fold cross-validation
                     number = 10,                          #Number of folds
                     summaryFunction = defaultSummary,     #Performance summary for comparing models in hold-out samples.
                     classProbs = TRUE)                    #Compute class probs in Hold-out samples


#-------------------------------------------------------------------------------------------------
#---------------------------- LOGISTIC REGRESSION MODEL (ALL VARIABLES) --------------------------
#-------------------------------------------------------------------------------------------------
## Train model -----------------------------------------------------------------------------------
set.seed(150) #For replication
#Train model using training data
LogReg.fit <- train(form = DIABETES ~ ., #formula for specifying inputs and outputs.
                    data = fTR,               #Training dataset 
                    method = "glm",                   #Train logistic regression
                    preProcess = c("center","scale"), #Center an scale inputs
                    trControl = ctrl,                 #trainControl Object
                    metric = "Accuracy")              #summary metric used for selecting hyperparameters
LogReg.fit          #information about the resampling
summary(LogReg.fit) #detailed information about the fit of the final model
str(LogReg.fit)     #all information stored in the model

## Understanding resampling methods -------------------------------------------
str(LogReg.fit$control$index)       #Training indexes
str(LogReg.fit$control$indexOut)    #Test indexes
LogReg.fit$resample                 #Resample test results
boxplot(LogReg.fit$resample$Accuracy, xlab = "Accuracy")
title("Boxplot for summary metrics of test samples")
dotplot(LogReg.fit$resample$Accuracy, xlab = "Accuracy")
title("Dotplot for summary metrics of test samples")

## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval$LRprob <- predict(LogReg.fit, type="prob", newdata = fTR) # predict probabilities
fTR_eval$LRpred <- predict(LogReg.fit, type="raw", newdata = fTR) # predict classes 
#test
fTS_eval$LRprob <- predict(LogReg.fit, type="prob", newdata = fTS) # predict probabilities
fTS_eval$LRpred <- predict(LogReg.fit, type="raw", newdata = fTS) # predict classes 
head(fTR_eval)

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$LRpred, #Predicted classes
                reference = fTR_eval$DIABETES, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$LRpred, 
                fTS_eval$DIABETES, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$DIABETES,       #Real observations
                     fTR_eval$LRprob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$DIABETES,       #Real observations
                     fTS_eval$LRprob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)


#-------------------------------------------------------------------------------------------------
#---------------------------- LOGISTIC REGRESSION MODEL (SELECTED VARIABLES) ---------------------
#-------------------------------------------------------------------------------------------------
## Train model -----------------------------------------------------------------------------------
set.seed(150) #For replication

#Train model using training data
LogReg.fit <- train(form = DIABETES ~ PREGNANT + GLUCOSE + SKINTHICKNESS + INSULIN + BODYMASSINDEX, #formula for specifying inputs and outputs.
                    data = fTR,               #Training dataset 
                    method = "glm",                   #Train logistic regression
                    preProcess = c("center","scale"), #Center an scale inputs
                    trControl = ctrl,                 #trainControl Object
                    metric = "Accuracy")              #summary metric used for selecting hyperparameters
LogReg.fit          #information about the resampling
summary(LogReg.fit) #detailed information about the fit of the final model
str(LogReg.fit)     #all information stored in the model

## Understanding resampling methods -------------------------------------------
str(LogReg.fit$control$index)       #Training indexes
str(LogReg.fit$control$indexOut)    #Test indexes
LogReg.fit$resample                 #Resample test results
boxplot(LogReg.fit$resample$Accuracy, xlab = "Accuracy")
title("Boxplot for summary metrics of test samples")
dotplot(LogReg.fit$resample$Accuracy, xlab = "Accuracy")

## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval$LRprob <- predict(LogReg.fit, type="prob", newdata = fTR) # predict probabilities
fTR_eval$LRpred <- predict(LogReg.fit, type="raw", newdata = fTR) # predict classes 
#test
fTS_eval$LRprob <- predict(LogReg.fit, type="prob", newdata = fTS) # predict probabilities
fTS_eval$LRpred <- predict(LogReg.fit, type="raw", newdata = fTS) # predict classes 
head(fTR_eval)

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$LRpred, #Predicted classes
                reference = fTR_eval$DIABETES, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$LRpred, 
                fTS_eval$DIABETES, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$DIABETES,       #Real observations
                     fTR_eval$LRprob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$DIABETES,       #Real observations
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
knn.fit = train(form = DIABETES ~ ., #formula for specifying inputs and outputs.
                data = fTR,   #Training dataset 
                method = "knn",
                preProcess = c("center","scale"),
                #tuneGrid = data.frame(k = 5),
                tuneGrid = data.frame(k = seq(1,15,1)),
                #tuneLength = 1000,
                trControl = ctrl, 
                metric = "Accuracy")
knn.fit #information about the settings
ggplot(knn.fit) #plot the summary metric as a function of the tuning parameter
knn.fit$finalModel #information about final model trained

### A partir de una k = 13 el modelo pierde abruptamente precisión siendo 12 el valor el valor de k que mejor precisión obtiene.
### Aumentar el número de folds del trainControl de cross-validation aumenta la precisión de las predicciones sobre el training set (0.85 -> 0.87).
### Aumentar el número de folds del trainControl de cross-validation reduce la precisión de las precicciones sobre el test set (0.83 -> 0.75).

## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval$knn_prob <- predict(knn.fit, type="prob" , newdata = fTR) # predict probabilities
fTR_eval$knn_pred <- predict(knn.fit, type="raw" , newdata = fTR) # predict classes 
#test
fTS_eval$knn_prob <- predict(knn.fit, type="prob" , newdata = fTS) # predict probabilities
fTS_eval$knn_pred <- predict(knn.fit, type="raw" , newdata = fTS) # predict classes 

## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$knn_pred, #Predicted classes
                reference = fTR_eval$DIABETES, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$knn_pred, 
                fTS_eval$DIABETES, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$DIABETES,       #Real observations
                     fTR_eval$knn_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$DIABETES,       #Real observations
                     fTS_eval$knn_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)


#-------------------------------------------------------------------------------------------------
#---------------------------- KNN MODEL (SELECTED VARIABLES)  ------------------------------------
#-------------------------------------------------------------------------------------------------
set.seed(150) #For replication
#Train knn model model.
#Knn contains 1 tuning parameter k (number of neigbors). Three options:
#  - Train with a fixed parameter: tuneGrid = data.frame(k = 5),
#  - Try with a range of values specified in tuneGrid: tuneGrid = data.frame(k = seq(2,120,4)),
#  - Caret chooses 10 values: tuneLength = 10,
knn.fit = train(form = DIABETES ~ PREGNANT + GLUCOSE + SKINTHICKNESS + INSULIN + BODYMASSINDEX, #formula for specifying inputs and outputs.
                data = fTR,   #Training dataset 
                method = "knn",
                preProcess = c("center","scale"),
                #tuneGrid = data.frame(k = 5),
                tuneGrid = data.frame(k = seq(1,15,1)),
                #tuneLength = 10,
                trControl = ctrl, 
                metric = "Accuracy")
knn.fit #information about the settings
ggplot(knn.fit) #plot the summary metric as a function of the tuning parameter
knn.fit$finalModel #information about final model trained

### A partir de una k = 13 el modelo pierde abruptamente precisión siendo 12 el valor el valor de k que mejor precisión obtiene.
### Aumentar el número de folds del trainControl de cross-validation aumenta la precisión de las predicciones sobre el training set (0.85 -> 0.87).
### Aumentar el número de folds del trainControl de cross-validation reduce la precisión de las precicciones sobre el test set (0.83 -> 0.75).

## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval$knn_prob <- predict(knn.fit, type="prob" , newdata = fTR) # predict probabilities
fTR_eval$knn_pred <- predict(knn.fit, type="raw" , newdata = fTR) # predict classes 
#test
fTS_eval$knn_prob <- predict(knn.fit, type="prob" , newdata = fTS) # predict probabilities
fTS_eval$knn_pred <- predict(knn.fit, type="raw" , newdata = fTS) # predict classes 


## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$knn_pred, #Predicted classes
                reference = fTR_eval$DIABETES, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$knn_pred, 
                fTS_eval$DIABETES, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$DIABETES,       #Real observations
                     fTR_eval$knn_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$DIABETES,       #Real observations
                     fTS_eval$knn_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)

#-------------------------------------------------------------------------------------------------
#---------------------------- DECISION TREE (ALL VARIABLES) --------------------------------------
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
tree.fit <- train(x = fTR[,1:8],  #Input variables.
                  y = fTR$DIABETES,   #Output variable
                  method = "rpart",   #Decision tree with cp as tuning parameter
                  control = rpart.control(minsplit = 5,  # Minimum number of obs in node to keep cutting
                                          minbucket = 5), # Minimum number of obs in a terminal node
                  parms = list(split = "gini"),          # impuriry measure
                  tuneGrid = data.frame(cp = 0.031), # TRY this: tuneGrid = data.frame(cp = 0.25),
                  #tuneLength = 10,
                  #tuneGrid = data.frame(cp = seq(0,0.1,0.0005)),
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
                reference = fTR_eval$DIABETES, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$tree_pred, 
                fTS_eval$DIABETES, 
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
#---------------------------- DECISION TREE (SELECTED VARIABLES) ---------------------------------
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
tree.fit <- train(x = fTR[,c(2,4,5)],  #Input variables.
                  y = fTR$DIABETES,   #Output variable
                  method = "rpart",   #Decision tree with cp as tuning parameter
                  control = rpart.control(minsplit = 5,  # Minimum number of obs in node to keep cutting
                                          minbucket = 5), # Minimum number of obs in a terminal node
                  parms = list(split = "gini"),          # impuriry measure
                  tuneGrid = data.frame(cp = 0.031), # TRY this: tuneGrid = data.frame(cp = 0.25),
                  #tuneLength = 10,
                  #tuneGrid = data.frame(cp = seq(0,0.1,0.0005)),
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
                reference = fTR_eval$DIABETES, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$tree_pred, 
                fTS_eval$DIABETES, 
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
svm.fit = train(form = DIABETES ~ ., #formula for specifying inputs and outputs.
                data = fTR,   #Training dataset 
                method = "svmRadial",
                preProcess = c("center","scale"),
                #tuneGrid = expand.grid(C = 100, sigma=seq(0.01,1,0.01)),
                tuneGrid =  data.frame(sigma = 0.01, C = 100),  
                #tuneGrid = expand.grid(C = seq(0.1,1000,length.out = 8), sigma=seq(0.01,50,length.out = 4)),
                #tuneLength = 10,
                trControl = ctrl, 
                metric = "Accuracy")
svm.fit #information about the resampling settings
svm.fit$resample[1]
ggplot(svm.fit) + scale_x_log10()
svm.fit$finalModel #information about the model trained
#Plot the svm support vectors:
isupvect <- alphaindex(svm.fit$finalModel)[[1]] #indexes for support vectors


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
                reference = fTR_eval$DIABETES, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$svm_pred, 
                fTS_eval$DIABETES, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$DIABETES,       #Real observations
                     fTR_eval$svm_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$DIABETES,       #Real observations
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
mlp.fit = train(form = DIABETES ~ GLUCOSE + INSULIN + SKINTHICKNESS + AGE, #formula for specifying inputs and outputs.
                data = fTR,   #Training dataset 
                method = "nnet",
                preProcess = c("center","scale"),
                maxit = 250,    # Maximum number of iterations
                tuneGrid = data.frame(size = 2, decay = 0.165),
                #tuneGrid = expand.grid(size = 5,
                #decay=seq(0.15,0.3, 0.005)),
                trControl = ctrl, 
                metric = "Accuracy")

mlp.fit #information about the resampling settings
mlp.fit$resample[1]
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
                reference = fTR_eval$DIABETES, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$mlp_pred, 
                fTS_eval$DIABETES, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$DIABETES,       #Real observations
                     fTR_eval$mlp_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$DIABETES,       #Real observations
                     fTS_eval$mlp_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)


