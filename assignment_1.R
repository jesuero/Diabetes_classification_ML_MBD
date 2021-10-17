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

# vemos valores de 0 por columnas
diabetes %>% 
  select(GLUCOSE) %>% 
  filter(GLUCOSE == 0) %>% 
  count()

diabetes %>% 
  select(BLOODPRESS) %>% 
  filter(BLOODPRESS == 0) %>% 
  count()

diabetes %>% 
  select(SKINTHICKNESS) %>% 
  filter(SKINTHICKNESS == 0) %>% 
  count()

diabetes %>% 
  select(INSULIN) %>% 
  filter(INSULIN == 0) %>% 
  count()

diabetes %>% 
  select(BODYMASSINDEX) %>% 
  filter(BODYMASSINDEX == 0) %>% 
  count()

# eliminamos los registros que tienen 0s
diabetes_tidy = diabetes %>% 
  filter(GLUCOSE != 0, BLOODPRESS != 0, SKINTHICKNESS != 0, INSULIN != 0, BODYMASSINDEX != 0) 
# vemos como han cambiado los datos
ggpairs(diabetes_tidy, aes(color=DIABETES))

# eliminamos outliers insulina, edad y pedigreefunction
diabetes_tidy2 = diabetes_tidy %>% 
  filter(INSULIN < 650, AGE != 81, PEDIGREEFUNC < 2)
ggpairs(diabetes_tidy2, aes(color=DIABETES))


# Step 6 - check out for class imbalances
table(diabetes_tidy$DIABETES)
table(diabetes$DIABETES)

# step 7 - data partition
## Divide the data into training and test sets ---------------------------------------------------
set.seed(150) #For replication
#create random 80/20 % split
trainIndex <- createDataPartition(diabetes_tidy2$DIABETES,      #output variable. createDataPartition creates proportional partitions
                                  p = 0.8,      #split probability for training
                                  list = FALSE, #Avoid output as a list
                                  times = 1)    #only one partition
#obtain training and test sets
fTR <- diabetes_tidy2[trainIndex,]
fTS <- diabetes_tidy2[-trainIndex,]
inputs <- 1:(ncol(fTR) - 1)
#Create dataset to include model predictions
fTR_eval <- fTR
fTS_eval <- fTS

# regresion logistica
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
                    #trControl = ctrl,                 #trainControl Object
                    metric = "Accuracy")              #summary metric used for selecting hyperparameters
LogReg.fit          #information about the resampling
summary(LogReg.fit) #detailed information about the fit of the final model
str(LogReg.fit)     #all information stored in the model


# regresion logistica agrupando edad
table(diabetes_tidy2$PREGNANT)
diabetes_tidy3 = diabetes_tidy2
diabetes_tidy3$PREGNANT = factor(diabetes_tidy3$PREGNANT)
levels(diabetes_tidy3$PREGNANT)[11:17] = "+10"

## Divide the data into training and test sets ---------------------------------------------------
set.seed(150) #For replication
#create random 80/20 % split
trainIndex <- createDataPartition(diabetes_tidy3$DIABETES,      #output variable. createDataPartition creates proportional partitions
                                  p = 0.8,      #split probability for training
                                  list = FALSE, #Avoid output as a list
                                  times = 1)    #only one partition
#obtain training and test sets
fTR <- diabetes_tidy3[trainIndex,]
fTS <- diabetes_tidy3[-trainIndex,]
inputs <- 1:(ncol(fTR) - 1)
#Create dataset to include model predictions
fTR_eval <- fTR
fTS_eval <- fTS

# regresion logistica
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
                    #trControl = ctrl,                 #trainControl Object
                    metric = "Accuracy")              #summary metric used for selecting hyperparameters
LogReg.fit          #information about the resampling
summary(LogReg.fit) #detailed information about the fit of the final model
str(LogReg.fit)     #all information stored in the model


ggpairs(diabetes_tidy3, aes(color=DIABETES))
  
