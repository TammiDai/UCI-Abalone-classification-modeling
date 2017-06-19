# import libraries we need
libs<-c("readr","neuralnet")
sapply(libs,require,character.only=TRUE)
options(stringsAsFactors = FALSE)
abalone <- read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", 
                    col_names=c("Sex","Length", "Diameter", "Height", "Whole_weight","Shucked_Weight","Viscera_weight","Shell_weight","Rings"))
View(abalone)
dim(abalone)

# set seed for reproducable results
set.seed(1234)
# check whether our dataset abalone has missing values, otherwise we have to clean/mining the data
apply(abalone,2,function(x) sum(is.na(x)))
# response variable Rins the mean approximately equal the variance, maybe consider about Possion regression
c(mean(abalone$Rings),var(abalone$Rings))

# split train 80% of the dataset and test 20% 
abalone$Sex<-factor(abalone$Sex)
index <- sample(1:nrow(abalone),round(0.8*nrow(abalone)))
train <- abalone[index,] # split original data, but do not scale, in order for comparison to neural network scaling data
test <- abalone[-index,]

# run a possion generalized linear regression on this dataset
summary(lm.fit<-glm(Rings~.,data=train,family=quasipoisson(link=log)))  # quasi poisson makes model's coefs more stable
# using the testing data to calculate MSE for this model
pr.glm<-predict(lm.fit,test,type="response")
(MSE.glm<-sum((pr.glm-test$Rings)^2)/nrow(test))
# Possion model MSE 4.72
# plot(lm.fit$residuals)  # check residual plot......


# Now fit a neural network model to compare the MSE with Possion MSE
# encoding categorical variable M,F,I into 1-of-N encoding
set.seed(1234)
abalone$Sex<-as.character(abalone$Sex)
abalone$Sex[abalone$Sex=="M"] <- 1
abalone$Sex[abalone$Sex=="F"] <- -1
abalone$Sex[abalone$Sex=="I"] <- 0
abalone$Sex<-as.numeric(abalone$Sex)

# normalized the data with max-min method
# before scale Abalone dataset, first convert it into data frame
maxs <- apply(abalone, 2, max) 
mins <- apply(abalone, 2, min)
scaled <- as.data.frame(scale(abalone, center = mins, scale = maxs - mins))
# 80% training data, leftover for testing
index <- sample(1:nrow(abalone),round(0.8*nrow(abalone)))
training <- scaled[index,]
testing <- scaled[-index,]
n<-names(training)
# create the formula to fit the model, avoid repeating input, since neuralnet package doesn't support Y~.
f<-as.formula(paste("Rings~",paste(n[!n %in% "Rings"],collapse = " + ")))
# we are going to use 2 hidden layers with this configuration, 2/3 of input variables for the 1st hidden, and 3 neurons for 2nd hidden
ann<-neuralnet(f,data=training,hidden=c(5,3),linear.output=FALSE) # specify linear.output=FALSE to create classification model
# plot to see artificial neural network 
plot(ann)
pr.ann<-compute(ann,testing[,-ncol(abalone)])
# scale the results back to orginal unit in order to make a meaningful comparison (or just a simple prediction)
pr.nn<-pr.ann$net.result*(max(abalone$Rings)-min(abalone$Rings))+min(abalone$Rings)
# also scale back testing data to be original format
test.r<-(testing$Rings)*(max(abalone$Rings)-min(abalone$Rings))+min(abalone$Rings)
(MSE.ann<-sum((test.r-pr.nn)^2/nrow(testing)))   # smaller than glm method

# compare MSE of Possion regression and neural network model, nerual network doing a better job than glm
paste(c("Possion MSE","Neural Network MSE"),c(MSE.glm,MSE.ann),sep=" ")

# let us compare the two method's plot to visualize the difference
par(mfrow=c(1,2))
plot(test$Rings,pr.nn,col='red',xlim=c(0,30),ylim=c(0,30),main='Real vs predicted ANN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='ANN',pch=18,col='red', bty='n')

plot(test$Rings,pr.glm,col='blue',xlim=c(0,30),ylim=c(0,30),main='Real vs predicted glm',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='GLM',pch=18,col='blue', bty='n', cex=.95)

# more useful visual comparison
# let us see more clearly with plotting the two in one same frame, glm method more scatter from the line
plot(test$Rings,pr.nn,col='red',xlim=c(0,30),ylim=c(0,30),main='Real vs predicted ANN',pch=18,cex=0.7,xlab="Real Rings",ylab="Predicted Values")
points(test$Rings,pr.glm,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('ANN','GLM'),pch=18,col=c('red','blue'))


# 10 folds cross validated MSE for the linear model
library(boot)
set.seed(1234)
lm.fit <- glm(Rings~.,data=abalone)
cv.glm(abalone,lm.fit,K=10)$delta[1]  # cross validation of glm, the average error

# run neural network for 10 times to compare each MSE and get the mean of MSE
set.seed(1234)
cv.error <- NULL
k <- 10

library(plyr) 
pbar <- create_progress_bar('text') #  to see the process of this loop
pbar$init(k)

for(i in 1:k){
  index <- sample(1:nrow(abalone),round(0.9*nrow(abalone)))
  train.cv <- scaled[index,]
  test.cv <- scaled[-index,]
  
  nn <- neuralnet(f,data=train.cv,hidden=c(5,2),linear.output=FALSE)
  
  pr.nn <- compute(nn,test.cv[,-ncol(abalone)])
  pr.nn <- pr.nn$net.result*(max(abalone$Rings)-min(abalone$Rings))+min(abalone$Rings)
  test.cv.r <- (test.cv$Rings)*(max(abalone$Rings)-min(abalone$Rings))+min(abalone$Rings)
  
  cv.error[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)
  
  pbar$step()
}

mean(cv.error) # cross validation average mse of neural network models, smaller than CV error of glm

# boxplot these MSE to see the distribution of these MSEs
boxplot(cv.error,xlab='MSE CV',col='chocolate1',
        border='aquamarine3',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)


# Random forest method ,fix overfitting problem,
# advantage of RF: even missing data,tree bagging, random selection of dataset
# we can see random forest work not well on such multi-layer response dataset
set.seed(1234)
library(randomForest)
abalone$Sex<-factor(abalone$Sex)
index <- sample(1:nrow(abalone),round(0.8*nrow(abalone)))
train_ <- abalone[index,]
test_ <- abalone[-index,]
rfm<-randomForest(Rings~.,train_,ntrees=1000,proximity=TRUE, importance=TRUE, nodesize=5)  # build 1000 random trees
pred<-round(predict(rfm,test_))  # maybe round up the prediction for comparasion
# we can see random forest work not well on such multi-layer response dataset
# table(as.numeric(test_$Rings),pred) # not very meaningful here to see confusion table, we can only predict the rings between some range, not exact number
# mean(test_[,ncol(abalone)]==pred)  # again, not very useful to see the accuracy in this particular dataset
importance(rfm,scale=TRUE)  # variable importance measurements, the higher number, the more important
# getTree(rfm,k=1,labelVar = T)  #extract the structure of a tree from a randomForest 
plot(test_$Rings,pred)
(MSE.rf<-sum((pred-test_$Rings)^2)/nrow(test_))  # MSE of RF model, better than glm but still cannot beat neural network


# now think about penalied regressioin, for vaiable shrinkage
# plot the pair plot for this dataset, and we can see multicollinearity beteween variables
library(readr)
options(stringsAsFactors = FALSE)
abalone <- read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", 
                    col_names=c("Sex","Length", "Diameter", "Height", "Whole_weight","Shucked_Weight","Viscera_weight","Shell_weight","Rings"))

View(abalone)
dim(abalone)
abalone$Sex[abalone$Sex=="M"] <- 1
abalone$Sex[abalone$Sex=="F"] <- -1
abalone$Sex[abalone$Sex=="I"] <- 0
abalone$Sex<-as.numeric(abalone$Sex)
pairs(abalone)
hist(abalone$Rings)  # the histogram looks like normal distribution, let us try fit gaussian family penalized regression models
qqnorm(abalone$Rings)  # a little bit right skewed, but since our data is limited, and Rings comparison to human ages, thus let us try normal family 
qqline(abalone$Rings)

# Fit penalized regression, lasso, or Ridge regression with Gaussian family
require(glmnet)
set.seed(1234)
index <- sample(1:nrow(abalone),round(0.8*nrow(abalone)))
train <- abalone[index,]
test <- abalone[-index,]
x<-as.matrix(train[,-ncol(abalone)])
y<-train$Rings
# Fitting the model (Ridge: Alpha = 0)
cv.ridge <- cv.glmnet(x, y, family='gaussian', alpha=0, standardize=TRUE, type.measure='auc')
pred.ridge <- predict(object=cv.ridge, as.matrix(test[,1:8]), type="response")
plot(cv.ridge, xvar="lambda")
(MSE.ridge<-sum((pred.ridge-test$Rings)^2)/nrow(test)) # The MSE still cannot beat neural network


# Fitting the model (Lasso: Alpha = 1)
cv.lasso <- cv.glmnet(x, y, family='gaussian', alpha=1, standardize=TRUE, type.measure='auc')
pred.lasso <- predict(object=cv.lasso, as.matrix(test[,1:8]), type="response")
(MSE.ridge<-sum((pred.lasso-test$Rings)^2)/nrow(test))
# results of lasso regression
plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)       
cv.lasso$lambda.min
cv.lasso$lambda.1se
coef(cv.lasso, s=cv.lasso$lambda.min)  # calculate lasso regression's coefficients with the minimun lambda


# Since the variables are highly correlated with each other, let us fit principal components regression
# import libraries we need
libs<-c("readr","pls")
sapply(libs,require,character.only=TRUE)
options(stringsAsFactors = FALSE)
abalone <- read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", 
                    col_names=c("Sex","Length", "Diameter", "Height", "Whole_weight","Shucked_Weight","Viscera_weight","Shell_weight","Rings"))
View(abalone)
dim(abalone)
abalone$Sex[abalone$Sex=="M"] <- 1
abalone$Sex[abalone$Sex=="F"] <- -1
abalone$Sex[abalone$Sex=="I"] <- 0
abalone$Sex<-as.numeric(abalone$Sex)
# principal components of orignal abalone data exclude Rings column
abalone.pc <- prcomp(abalone[,-ncol(abalone)])
cumsum(abalone.pc$sdev^2)/sum(abalone.pc$sdev^2)

# principal components of scaled abalone data exclude Rings column
abalone.pcr <- prcomp(abalone[,-9], scale = TRUE,center=TRUE)
(cumvar <- cumsum(abalone.pcr$sdev^2)/sum(abalone.pcr$sdev^2))

plot(abalone.pcr,type='l')  # Scree plot
summary(abalone.pcr)  # will chose the first 3 components can explian 95% of the total variation

# require(pls)
set.seed (1234)
# standardized before running the pcr algorithm, perform 10 fold cross-validation and therefore set the validation = CV
summary(pcr_model <- pcr(Rings~., data = abalone, scale = TRUE, validation = "CV"))
validationplot(pcr_model)  # plot Root Mean Square Error againt number of components
validationplot(pcr_model, val.type = "R2")  # plot R^2 again numbeer of components
# plot the predicted vs measured values using the predplot function 
predplot(pcr_model)
# regression coefficients can be plotted using the coefplot function
coefplot(pcr_model)

# Train-test split
set.seed(1234)
index <- sample(1:nrow(abalone),round(0.8*nrow(abalone)))
train <- abalone[index,]
test <- abalone[-index,]
# convert categorical variables into balanced numeric data
abalone$Sex[abalone$Sex=="M"] <- 1
abalone$Sex[abalone$Sex=="F"] <- -1
abalone$Sex[abalone$Sex=="I"] <- 0
abalone$Sex<-as.numeric(abalone$Sex)

# only choose the first 3 components to fit the model and predict
pcr_model <- pcr(Rings~., data = train,scale =TRUE, validation = "CV",ncomp = 3)
pcr_pred <- predict(pcr_model, test, ncomp = 3)  
mean((pcr_pred - test$Rings)^2)/nrow(test)
# so far Principal components model gives us the best MSE results for Abalone dataset due to the variables highly collineararity
plot(pcr_model$fitted.values)  # no pattern, so far, the best model
plot(pcr_model$residuals)