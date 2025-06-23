rm(list = ls())
## Load libraries
library(ggplot2)
library(tidyr)
library(rstatix)
library(reshape2)
library(reshape)
library(ggpubr)
library(gridExtra)
library(pROC)
library(nnet)
library(ROCR)
library(survival)
library(survminer)
library(tidyverse)
library(dplyr)
library(plotmo) # for plot_glmnet
library(glmnet)
library(mlr)
library(MASS)
library(magrittr)
library(cutpointr)

## load data
path_Baseline <- "/Path/to/data/Baseline/"
path <- "/Path/to/data/Change15/"
suffix <- ""

OLINK_X_train <- read.csv(paste(path, "X_train_Olink",suffix, ".csv", sep=""), header=TRUE)
CBC_X_train <- read.csv(paste(path, "X_train_OlinkCBC",suffix, ".csv", sep=""), header=TRUE)

y_train <- read.csv(paste(path, "y_train",suffix, ".csv", sep=""), header=TRUE)

OLINK_X_test <- read.csv(paste(path, "X_test_Olink",suffix, ".csv", sep=""), header=TRUE)
CBC_X_test <- read.csv(paste(path, "X_test_OlinkCBC",suffix, ".csv", sep=""), header=TRUE)

y_test <- read.csv(paste(path, "y_test",suffix, ".csv", sep=""), header=TRUE)


######################################################################################################################################################
########################################################### LASSO RUN CV TO GET LAMBDA MIN ###########################################################
######################################################################################################################################################

### Set parameters
X_train_data <- OLINK_X_train
y_train_data <- y_train
X_holdout_data <- OLINK_X_test
y_holdout_data <- y_test

X_train_scaled <- scale(X_train_data)
X_train <- data.matrix(X_train_scaled)

y_train <- data.matrix(y_train_data)
X_holdout <- data.matrix(scale(X_holdout_data))
y_holdout <- data.matrix(y_holdout_data)


plotname <- 'Olink_Baseline'
##################################################
########## RUN LOOP TO CHOOSE LAMBDA MIN #########
##################################################
list_seednum <- c()
list_LambdaMin <- c()
list_pval <- c()
list_num_features <- c()
list_holdout_pval <- c()
list_accuracy_train <- c()
list_accuracy_holdout <- c()
list_auc_train <- c()
list_auc_holdout <- c()
features_greater_than_1 <- c()
features_less_than_1 <- c()

for(seednum in 1:100){
  set.seed(seednum)
  list_seednum <- c(list_seednum, seednum)
  
  # Fit lasso model with cross-validation
  lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1, family = 'binomial', nfolds = 5)
  list_LambdaMin <- c(list_LambdaMin, lasso_cv$lambda.min)
  
  # Get the coefficients at lambda.min and identify non-zero coefficients
  coef_min <- coef(lasso_cv, s = lasso_cv$lambda.min)
  non_zero_indices <- which(coef_min != 0)  # Indices of non-zero coefficients
  odds_ratios <- exp(coef_min[non_zero_indices])
  
  features_greater_than_1_temp <- ""
  features_less_than_1_temp <- ""
  
  # Flag to control whether to add a comma
  first_greater_than_1 <- TRUE
  first_less_than_1 <- TRUE
  
  for (i in seq_along(non_zero_indices)) {
    feature_name <- rownames(coef_min)[non_zero_indices[i]]
    print(feature_name)
    if (feature_name != "(Intercept)"){
      coef_value <- coef_min[non_zero_indices[i]]
      or_value <- odds_ratios[i]
      print(or_value)
      # Classify features based on the OR
      if (or_value > 1) {
        if (first_greater_than_1) {
          features_greater_than_1_temp <- feature_name
          first_greater_than_1 <- FALSE
        } else {
          features_greater_than_1_temp <- paste(features_greater_than_1_temp, feature_name, sep = ", ")
        }}
        
       else if (or_value < 1) {
        print(paste("Odds rato less than 1: ", feature_name, " ", or_value))
        if (first_less_than_1) {
          features_less_than_1_temp <- feature_name
          first_less_than_1 <- FALSE
        } else {
          features_less_than_1_temp <- paste(features_less_than_1_temp, feature_name, sep = ", ")
          
      }
  }}}
  
  features_greater_than_1 <- c(features_greater_than_1, features_greater_than_1_temp)
  features_less_than_1 <- c(features_less_than_1,features_less_than_1_temp)
  
  num_features <- length(non_zero_indices) - 1  # Exclude intercept
  list_num_features <- c(list_num_features, num_features)
  
  
  # Prepare data for plotting coefficients
  dataframe_coef <- data.frame(coef = coef_min[non_zero_indices], 
                               analyte_full = rownames(coef_min)[non_zero_indices])
  
  # Remove intercept and prepare for ggplot
  dataframe_coef <- dataframe_coef[dataframe_coef$analyte_full != "(Intercept)", ]
  dataframe_coef$s <- ifelse(dataframe_coef$coef < 0, "negative", "positive")
  
  # Sort by absolute value of coefficients (in decreasing order)
  dataframe_coef <- dataframe_coef[order(abs(dataframe_coef$coef), decreasing = TRUE),]
  level_order <- dataframe_coef$analyte_full
  
  # Plot the coefficients using ggplot
  png(paste(path, "/EffectSizePlots/",plotname, "_", seednum, ".png", sep=""), 
      width=4, height=5, units="in", res=250)
  
  plot <- ggplot(dataframe_coef, aes(x = factor(analyte_full, level = level_order),
                             y = coef, fill = s)) +
    geom_col(colour = "black") +
    xlab("") +
    ylab("Coefficients") +
    scale_fill_manual(values = c('#B8C0BB', '#D2CBAF'), limits = c("positive", "negative")) +
    ggtitle(paste("lambda = ", round(lasso_cv$lambda.min, 3), sep = "")) +
    custom_theme()
  
  print(plot)
  
  dev.off()
  
  # Get feature names for non-zero coefficients, excluding the intercept
  feature_names <- rownames(coef_min)[non_zero_indices]
  feature_names <- feature_names[feature_names != "(Intercept)"]
  feature_names <- paste(feature_names, collapse = ", ")  # Concatenate feature names with commas
  # list_feature_names <- c(list_feature_names, feature_names)
  
  # **Fit final lasso model using lambda.min**
  final_lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = lasso_cv$lambda.min, family = 'binomial')
  
  # Predict on training and holdout sets using the final fitted model
  preds_train <- predict(final_lasso_model, newx = X_train, type = "response")
  preds_holdout <- predict(final_lasso_model, newx = X_holdout, type = "response")
  
  # Debugging: Check the range of predicted probabilities for holdout set
  print(paste("Predicted probabilities for holdout: Min =", min(preds_holdout), "Max =", max(preds_holdout)))
  
  # Convert continuous probabilities to binary predictions (0 or 1) based on 0.5 threshold
  pred_train_binary <- ifelse(preds_train > 0.5, 1, 0)
  pred_holdout_binary <- ifelse(preds_holdout > 0.5, 1, 0)
  
  # Debugging: Check the first few binary predictions for the holdout set
  print(paste("First few predicted binary values (holdout):", paste(pred_holdout_binary, collapse = ", ")))
  
  # Calculate Accuracy for training and holdout sets
  accuracy_train <- mean(pred_train_binary == y_train)  # Proportion of correct predictions for training
  accuracy_holdout <- mean(pred_holdout_binary == y_holdout)  # Proportion of correct predictions for holdout
  
  # Debugging: Print accuracy values
  #print(paste("Accuracy (train):", accuracy_train))
  #print(paste("Accuracy (holdout):", accuracy_holdout))
  
  # Calculate AUC for training and holdout sets
  auc_train <- pROC::roc(y_train, preds_train)$auc  # AUC for training set
  auc_holdout <- pROC::roc(y_holdout, preds_holdout)$auc  # AUC for holdout set
  
  # Append Accuracy and AUC values to respective lists
  list_accuracy_train <- c(list_accuracy_train, accuracy_train)
  list_accuracy_holdout <- c(list_accuracy_holdout, accuracy_holdout)
  list_auc_train <- c(list_auc_train, auc_train)
  list_auc_holdout <- c(list_auc_holdout, auc_holdout)
  
  # Perform Wilcoxon rank-sum test for p-value
  pval_train <- wilcox.test(preds_train ~ y_train)$p.value
  list_pval <- c(list_pval, pval_train)
  
  pval_holdout <- wilcox.test(preds_holdout ~ y_holdout)$p.value
  list_holdout_pval <- c(list_holdout_pval, pval_holdout)
}

# Output results in a data frame
output_seedNum <- data.frame('seed' = list_seednum, 'lambdaMin' = list_LambdaMin,
                             'pval' = list_pval, 'accuracy_train' = list_accuracy_train,
                             'auc_train' = list_auc_train,
                             'accuracy_holdout' = list_accuracy_holdout, 'auc_holdout' = list_auc_holdout,
                             'numfeatures' = list_num_features, 'OR>1 Features' = features_greater_than_1,
                             'OR<1 Features' = features_less_than_1, check.names=FALSE
)
#### HOW TO CHOOSE LAMBDA MIN:
write.csv(output_seedNum, paste(path, "OR/Olink_Baseline.csv", sep=""), row.names=FALSE)




# Fit final lasso model using lambda.min
lambda_min = 0.199
final_lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = lambda_min, family = 'binomial')

# Predict on training and holdout sets using the final fitted model
preds_train <- predict(final_lasso_model, newx = X_train, type = "response")
preds_holdout <- predict(final_lasso_model, newx = X_holdout, type = "response")

pred_holdout_binary <- ifelse(preds_holdout > 0.5, 1, 0)
accuracy_holdout <- mean(pred_holdout_binary == y_holdout)
accuracy_holdout



##############################################################
#################### TWO EFFECT SIZE PLOTS ###################
##############################################################

fit <- glmnet(X_train, y_train, alpha = 1, lambda = lambda_min, family = 'binomial')
coef_min <- coef(final_lasso_model, s = lambda_min)
data_coef <- coef_min
non_zero_indices <- which(coef_min != 0)  # Indices of non-zero coefficients
rownames(coef_min)[non_zero_indices]


dataframe_coef <- as.data.frame(summary(data_coef))
row1 <- colnames(X_train)[unlist(dataframe_coef$i[2:length(dataframe_coef$i)])-1] #remove the intercept column
row2 <- dataframe_coef$x[2:length(dataframe_coef$x)] #remove the intercept column
nonzero_df = data.frame("analyte_full" = row1, "coef" = row2)
nonzero_df$analyte <- gsub("\\.","-", as.character(nonzero_df$analyte))
nonzero_df$analyte <- gsub("_"," ", as.character(nonzero_df$analyte))
nonzero_df <- nonzero_df[sort(abs(nonzero_df$coef),decreasing=T,index.return=T)[[2]],]
nonzero_df$s <- ifelse(nonzero_df$coef < 0, "negative", "positive")
level_order <- nonzero_df$analyte

dataframe <- X_train_data

important_list = nonzero_df$analyte_full
dataframe[,important_list] <- lapply(dataframe[,important_list],as.numeric)
new_x <- data.matrix(scale(dataframe[,important_list]))

fit_elim <- glmnet(new_x, y_train, family = 'binomial', alpha = 1)
beta = coef(fit_elim)
tmp <- as.data.frame(as.matrix(beta))
tmp$coef <- row.names(tmp)
tmp <- reshape::melt(tmp, id = "coef")
tmp$variable <- as.numeric(gsub("s", "", tmp$variable))
tmp$lambda <- fit_elim$lambda[tmp$variable+1] # extract the lambda values
tmp$norm <- apply(abs(beta[-1,]), 2, sum)[tmp$variable+1] # compute L1 norm
tmp$coef <- gsub("\\.","-",as.character(tmp$coef))
tmp$coef <- sub("_", " ", tmp$coef)


custom_bw <- function() {
  theme_bw() %+replace%
    theme(
      plot.title=element_text(hjust=0.5, face="bold", size=18),
      legend.text=element_text(size=10, face="bold"),
      legend.key.width = unit(.2, "cm"),
      legend.key.height = unit(0.5, "cm"),
      legend.spacing.x = unit(0.2, "cm"),
      axis.text.y = element_text(size=12),
      axis.text.x = element_text(size=12),
      axis.title.y = element_text(size=14, angle=90, face="bold"),
      axis.title.x = element_text(size=14, face="bold"),
      axis.line = element_blank(),
      axis.line.x.bottom = element_line(size = .3, color = "black"),
      axis.line.y.left = element_line(size = .3, color = "black")
    )
}


plotname = "Train_Final_Model_Baseline_Plot1"
png(paste(path, "/EffectSizePlots/",plotname, ".png", sep=""), 
    width=5, height=5, units="in", res=250)

ggplot(tmp[tmp$coef != "(Intercept)",], aes(lambda, value, color = coef, linetype = coef)) + 
  geom_line(linewidth=.6) + xlab("Lambda") + ylab("Coefficients") +
  guides(color = guide_legend(title = "", override.aes = list(size = 0.1)), 
         linetype = guide_legend(title = "", override.aes = list(linewidth = 0.6))) +
  custom_bw() + theme(legend.key.width = unit(3,"lines"))

dev.off()



custom_theme <- function() {
  theme_minimal() %+replace%
    theme(
      legend.position = "none",
      plot.title=element_text(hjust=0.5, face="bold", size=16),
      axis.text.y = element_text(size=12),
      axis.text.x = element_text(face="bold", size=12, angle=70),
      axis.title.y = element_text(size=14, angle = 90, margin = margin(t = 0, r = 5, b = 0, l = 0), face = "bold"),
      #margin = margin(t = 0, r = 6, b = 0, l = 0)
      axis.title.x = element_text(size=14, face="bold"),
    )
}

plotname = "Train_Final_Model_TwoTimes_Plot2"
png(paste(path, "/EffectSizePlots/",plotname, ".png", sep=""), 
    width=4, height=5, units="in", res=250)

ggplot(nonzero_df, aes(x=factor(analyte, level = level_order),
                       y=coef, fill=s)) + geom_col(colour="black") + custom_theme() + xlab("") + 
  ylab('Coefficients') + scale_fill_manual(values = c('#98DDDE', '#C9B27C'), limits = c("positive", "negative")) +
  ggtitle(paste("lambda = ", round(lambda_min,3), sep=""))

dev.off()



##############################################################
#################### Probability Comparison ###################
##############################################################

dataframe <- Olink
values <- c(0, 1)
index <- c("NCB", "CB")
dataframe$Response1_num <- values[match(dataframe$Response1, index)]

Patient_ID_Test <- c(28, 41, 29, 38, 4, 27, 24, 50, 25)
## [1, 2, 5, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 26, 30, 31, 32, 35, 36, 42, 43, 44, 45, 47, 48, 49, 51]
Patient_ID_Train <- setdiff(dataframe$Patient, Patient_ID_Test)

X_train_df <- dataframe[dataframe$Patient %in% Patient_ID_Train, ]
X_test_df  <- dataframe[dataframe$Patient %in% Patient_ID_Test, ]

y_train_V2 <- X_train_df[, c("Response1_num")]
y_test_V2  <- X_test_df[, c("Response1_num")]

X_train_V2 <- X_train_df[, colnames(OLINK_X_train)]
X_test_V2  <- X_test_df[,colnames(OLINK_X_train)]

# Ensure all columns are numeric
X_train_V2[] <- lapply(X_train_V2, as.numeric)
X_test_V2[]  <- lapply(X_test_V2, as.numeric)

# Scale features
X_train_scaled_V2 <- scale(data.matrix(X_train_V2))
X_test_scaled_V2 <- scale(data.matrix(X_test_V2))

# Convert to matrix for glmnet
X_train_mat <- as.matrix(X_train_scaled_V2)
X_test_mat <- as.matrix(X_test_scaled_V2)



final_lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = lambda_min, family = 'binomial')




new_x = X_train_mat

preds_1 <- predict(final_lasso_model, type="response", family = 'binomial', newx=new_x, s = lambda_min)
current_df = X_train_df

probability_df <- data.frame("Patient" = current_df$Patient, 
                             "Response" = current_df$Response1, 
                             "preds" = preds_1, 
                             "ICI" = current_df$ICI)
colnames(probability_df)[colnames(probability_df) == "s1"] <- "preds"

pred_holdout_binary <- ifelse(preds_1 > 0.5, 1, 0)
accuracy_holdout <- mean(pred_holdout_binary == y_test_V2)




write.csv(probability_df, 
          paste(path, "/OR/Probabilities_Test.csv", sep=""), 
          row.names=FALSE)





probs_theme <- function() {
  theme_minimal() %+replace%
    theme(
      plot.title=element_text(hjust=0.5, size=12, face="bold", margin = margin(t = 0, r = 0, b = 5, l = 0)),
      legend.position = "none",
      axis.text.y = element_text(size=10),
      axis.text.x = element_text(size=10, face="bold"),
      axis.title.y = element_text(size=12, angle = 90, margin = margin(t = 0, r = 5, b = 0, l = 0), face="bold"),
      axis.title.x = element_text(size=12, margin = margin(t = 5, r = 0, b = 0, l = 0), face="bold"),
      axis.line = element_blank(),
      axis.line.x.bottom = element_line(size = .3, color = "black"),
      axis.line.y.left = element_line(size = .3, color = "black")
      
    )
}



plotname = "Train_Model_Baseline_BoxPlot"
png(paste(path, "/EffectSizePlots/",plotname, ".png", sep=""), 
    width=2.5, height=3.5, units="in", res=250)

my_comparisons <- list( c('NCB', 'CB'))
probability_df$Response <- as.factor(probability_df$Response)
probability_df$Response = factor(probability_df$Response, level=c('NCB', 'CB'))

ggplot(probability_df, aes(x = as.factor(Response), y = as.numeric(unlist(probability_df["preds"])), fill=as.factor(Response))) + 
  geom_boxplot(outlier.size=.75) +
  ylab("Predicted P(Response)") + xlab("") +
  geom_jitter(size=.75) + 
  stat_compare_means(comparisons = my_comparisons, method = "wilcox.test", paired=FALSE, size=4) +
  ggtitle(paste("LASSO\n lambda = ", round(lambda_min,3), sep="")) + 
  scale_x_discrete()  + ylim(c(.15,.85)) + scale_fill_manual(values = c("#A4B279", "#A4B9DA")) + probs_theme()
dev.off()
#



# Define output PNG file
png(filename = "/path/to/output.png", width=2.5, height=4, units="in", res=250)
my_comparisons <- list( c('NCB', 'CB'))




####################################################################
######## Make ROC curve and calculate AUC for LASSO models ########
####################################################################
# library(pROC)
# Assuming `probability_df` contains your predictions and actual responses
roc_curve <- pROC::roc(as.factor(probability_df$Response), probability_df$preds, levels = c("NCB", "CB"), direction = "<")

# Plot the ROC curve
plot(roc_curve, main="ROC Curve for LASSO Model")

# Calculate AUC
auc_value <- pROC::auc(roc_curve)
print(paste("AUC:", auc_value))

png("./path/to/output.png", width = 5, height = 5, units = "in", res = 300)

plot(roc_curve, main = "ROC Curve for Model Performance",col = "#1c61b6", lwd = 2)
auc_value <- pROC::auc(roc_curve)
legend("bottomright",legend = paste("AUC =", round(auc_value, 3)),col = "#1c61b6",lwd = 2)

dev.off()





