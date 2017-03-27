library(caret)
library(utils)
##Input the csv
#Dataset publicly available at https://www.kaggle.com/uciml/glass/downloads/glass.csv
input <- "~//Kaggle//Glass//glass.csv" ##this is the file location
gdata <- data.frame(read.csv(input, header = TRUE, sep = ","))
#Initial Exploration
head(gdata)
#Subsetting data index
trainindex <- createDataPartition(gdata$Type, p = 0.8, list = FALSE)

###Rerun from here for reproduceable results
#Creating data partitions
training <- gdata[trainindex, ]
testing <- gdata[-trainindex, ]

#Training Set
cnames <-
  colnames(training)
obscnames <-
  cnames[-length(cnames)]
precnames <-
  cnames[length(cnames)]
trainobs <-
  training[, c(obscnames)]
trainpred <- training[, c(precnames)]

#Test Set
ctnames <-
  colnames(testing)
obsctnames <-
  ctnames[-length(ctnames)]
prectnames <-
  ctnames[length(ctnames)]
testobs <-
  testing[, c(obsctnames)]
testpred <- testing[, c(prectnames)]

#CV methods
timeControl <-
  trainControl(
    method = "repeatedcv",
    number = 5,
    repeats = 5,
    allowParallel = FALSE,
    classProbs = TRUE
  )

#Change to factors for caret
trainpredtemp <-
  as.factor(trainpred)
trainpred <- make.names(trainpredtemp)

###Function Creation
tryModel <- function(t) {
  res <- try({
    #ML with grid search
    modinput <-
      t
    modeltype <-
      paste(modinput, sep = "")
    set.seed(214)
    premodel <-
      train(trainobs,
            trainpred,
            method = modeltype,
            trControl = timeControl)
    
    #Tune values found experimentally using grid search
    tgrid <-
      premodel$bestTune
    set.seed(214)
    model <-
      train(
        trainobs,
        trainpred,
        method = modeltype,
        trControl = timeControl,
        tuneGrid = tgrid
      )
    
    #In sample training accuracy
    trainaccuracy <- model$results$Accuracy
    #Extracting predictions
    predictions <-
      data.frame(predict(model, testobs))
    testpred1 <-
      as.data.frame(make.names(testpred))
    results <-
      cbind.data.frame(testpred1, predictions)
    colnames(results) <- c("Actual", "Predicted")
    
    #Finding the testing sample accuracy
    outcome <- data.frame()
    for (i in 1:nrow(results)) {
      if (results$Actual[i] == results$Predicted[i]) {
        tresult <- 1
      } else{
        tresult <- 0
      }
      outcome <- rbind.data.frame(outcome, tresult)
    }
    #Calculating training and testing accuracies
    testaccuracy <-
      mean(outcome[, 1])
    cat("Train accuracy: ", trainaccuracy)
    cat("Test accuracy: ", testaccuracy)
    #Creating outputs
    trainacc <- trainaccuracy
    testacc <- testaccuracy
    #Final output
    output <- list(modeltype, tgrid, trainacc, testacc)
    return(output)
  }, silent = TRUE)
  if (inherits(res, "try-error")) {
    #Final output
    modeltype <- NA
    trainacc <- NA
    testacc <- NA
    tgrid <- NA
    
    #Final output
    output <- list(modeltype, tgrid, trainacc, testacc)
    return(output)
  }
}
variousmodels <-
  c("gbm", "svmRadial", "rf", "nnet")
modelresults <- sapply(variousmodels, tryModel, simplify = TRUE)
modelsdf <-
  as.data.frame(t(modelresults))
colnames(modelsdf) <- c("ModelType", "TuneValues", "Train", "Test")
#modelsdf is the final resulting table
