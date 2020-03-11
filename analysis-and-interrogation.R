

# install.packages(c("ggrepel", "viridis"))

library(e1071)
library(glm)
library(keras)
library(pROC)
library(rsample)
library(tidyverse)
library(xgboost)

isBuggy <- function (bug) {
return(ifelse(bug > 0, 1, 0))
}

unified_class <- as_tibble(read_csv("Unified-class.csv"))
unified_class$has_bugs <- as.numeric(lapply(unified_class$bug, isBuggy))

bugs_per_nl <- unified_class %>%
select(bug, NL)

bugs_per_nlm <- unified_class %>%
select(bug, NLM) 



bugs_per_lldc <- unified_class %>%
select(bug, LLDC)

bugs_per_ldc <- unified_class %>%
select(bug, LDC)

bugs_per_nlg <- unified_class %>%
select(bug, NLG)

bugs_per_nle <- unified_class %>%
select(bug, NLE, Type)


bugs_per_tnos <- unified_class %>%
select(bug, TNOS, Type)


bugs_per_tlloc <- unified_class %>%
select(bug, TLLOC)


# Sample / split the data.
x_split <- initial_split(unified_class, prop = 0.7, strata = has_bugs)
x_train <- training(x_split)
x_test <- testing(x_split)
y_train <- x_train$has_bugs
y_test <- x_test$has_bugs




set.seed(123)
model_glm <- glm(y_train~NL+NLM+LLDC+LDC+NLG+NLE+TNOS+TLLOC, data = x_train, family = "binomial")
predictions_glm <- predict(model_glm, x_test, type = "response")
acc_glm <- mean(round(predictions_glm) == y_test)
auc_glm = auc(y_test, predictions_glm)

data_train <- select(x_train, NL, NLM, LLDC, LDC, NLG, NLE, TNOS, TLLOC)
data_test <- select(x_test, NL, NLM, LLDC, LDC, NLG, NLE, TNOS, TLLOC)
train_m <- as.matrix(data_train)
test_m <- as.matrix(data_test)


set.seed(123)
model_svm <- svm(train_m, y_train)
predictions_svm <- predict(model_svm, test_m)
acc_svm <- mean(round(predictions_svm) == y_test)
auc_svm = auc(y_test, predictions_svm)

set.seed(123)
model_xgb <- xgboost(data = train_m, label = y_train, max_depth = 8, eta= 0.3, nrounds = 4, objective = "binary:logistic")

predictions_xgb <- predict(model_xgb, test_m)
acc_xgb <- mean(round(predictions_xgb) == y_test)
auc_xgb = auc(y_test, predictions_xgb)


# one hot encoding labels is what keras requires
trainLabels = to_categorical(y_train)
testLabels = to_categorical(y_test)

# Get the number of units and the input shape for the model.
x_units <- 2*nrow(train_m)
x_shape <- ncol(train_m)

# create neural network
model = keras_model_sequential()
model %>% 
  layer_dense(units=x_units,
              activation='sigmoid',
              input_shape=c(x_shape))  %>%
  layer_dense(units=2,
              activation='sigmoid')

# compile model
model %>%
  compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics='accuracy')

# train model
history = model %>%
  fit(train_m,
      trainLabels,
      epoch = 10,
      batch_size=1)

# make predictions
predictions_nn <- model %>%
predict_proba(test_m)

acc_nn <- mean(round(predictions_nn[, 2]) == y_test)
auc_nn = auc(y_test, predictions_nn[, 2])

model_data <- data.frame(Model_name = c("Logistic Regression", "Support Vector Machine", "Extreme Gradient Boosting", "Keras Neural Network"),
model_acc = c(acc_glm, acc_svm, acc_xgb, acc_nn),
model_auc = c(auc_glm, auc_svm, auc_xgb, auc_nn))


# Calcculate the variable importance for XG Boost, as that is the best model.
important_df_xgb <- data.frame(xgb.importance(model = model_xgb))

