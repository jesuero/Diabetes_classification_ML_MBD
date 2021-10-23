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
# vemos los datos sin tocar
ggpairs(diabetes, aes(color=DIABETES))

# creamos un dataset en el que no hay valores de 0 para las columnas glusose, bloodpress, skinthickness, insulin y bodymassindex
diabetes_noCeros = filter(diabetes, GLUCOSE != 0, BLOODPRESS != 0, SKINTHICKNESS != 0, INSULIN != 0, BODYMASSINDEX != 0) 

# creamos dos dataframes con los diabeticos por un lado y los no diabeticos por otro
diabetes_SI = filter(diabetes_noCeros, DIABETES == 1)
diabetes_NO = filter(diabetes_noCeros, DIABETES == 0)

diabetes_tidy = diabetes

# se van a sustituir los ceros de las columnas por la mediana de esa columna dependiendo de si es o no diabetico
# columna glucose tratamiento de valores 0
count(diabetes_tidy, GLUCOSE == 0)
diabetes_tidy$GLUCOSE[diabetes_tidy$GLUCOSE == 0 & diabetes_tidy$DIABETES == 0] = median(diabetes_NO$GLUCOSE)
diabetes_tidy$GLUCOSE[diabetes_tidy$GLUCOSE == 0 & diabetes_tidy$DIABETES == 1] = median(diabetes_SI$GLUCOSE)

# columna bloodpress tratamiento de valores 0
count(diabetes_tidy, BLOODPRESS == 0)
diabetes_tidy$BLOODPRESS[diabetes_tidy$BLOODPRESS == 0 & diabetes_tidy$DIABETES == 0] = median(diabetes_NO$BLOODPRESS)
diabetes_tidy$BLOODPRESS[diabetes_tidy$BLOODPRESS == 0 & diabetes_tidy$DIABETES == 1] = median(diabetes_SI$BLOODPRESS)

# columna skinthickness tratamiento de valores 0
count(diabetes_tidy, SKINTHICKNESS == 0)
diabetes_tidy$SKINTHICKNESS[diabetes_tidy$SKINTHICKNESS == 0 & diabetes_tidy$DIABETES == 0] = median(diabetes_NO$SKINTHICKNESS)
diabetes_tidy$SKINTHICKNESS[diabetes_tidy$SKINTHICKNESS == 0 & diabetes_tidy$DIABETES == 1] = median(diabetes_SI$SKINTHICKNESS)

# columna insulin tratamiento de valores 0
count(diabetes_tidy, INSULIN == 0)
diabetes_tidy$INSULIN[diabetes_tidy$INSULIN == 0 & diabetes_tidy$DIABETES == 0] = median(diabetes_NO$INSULIN)
diabetes_tidy$INSULIN[diabetes_tidy$INSULIN == 0 & diabetes_tidy$DIABETES == 1] = median(diabetes_SI$INSULIN)

# columna bodymassindex tratamiento de valores 0
count(diabetes_tidy, BODYMASSINDEX == 0)
diabetes_tidy$BODYMASSINDEX[diabetes_tidy$BODYMASSINDEX == 0 & diabetes_tidy$DIABETES == 0] = median(diabetes_NO$BODYMASSINDEX)
diabetes_tidy$BODYMASSINDEX[diabetes_tidy$BODYMASSINDEX == 0 & diabetes_tidy$DIABETES == 1] = median(diabetes_SI$BODYMASSINDEX)

# mostramos los datos tras el tratamiento
ggpairs(diabetes_tidy, aes(color=DIABETES))

# analizar el valor de 99 de skinthickness procemos a eliminarlo
# no cuadra mucho, 99 es un valor extraño, podria ser error de medida
# si skinthickness tan alto deberia tener mayor bodymassindex
# procemos a eliminarlo
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
# convertimos la columna diabetes para sus valores sean YES or NO en vez de 0 o 1
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
#---------------------------- LOGISTIC REGRESSION MODEL ----------------------------------------------
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


