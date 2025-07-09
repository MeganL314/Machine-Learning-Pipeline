rm(list = ls())

# Load required libraries
library(ggplot2)
library(tidyr)
library(rstatix)
library(ggpubr)
library(gridExtra)
library(pROC)
library(nnet)
library(ROCR)
library(survival)
library(survminer)
library(dplyr)
library(glmnet)
library(MASS)
library(magrittr)
library(cutpointr)
library(reshape)

# Load data
input_path <- "/path/to/data/"
features_train <- read.csv(paste0(input_path, "X_train.csv"), header = TRUE)
features_test <- read.csv(paste0(input_path, "X_test.csv"), header = TRUE)
labels_train <- read.csv(paste0(input_path, "y_train.csv"), header = TRUE)
labels_test <- read.csv(paste0(input_path, "y_test.csv"), header = TRUE)


######################################################################################################################################################
########################################################### LASSO RUN CV TO GET LAMBDA MIN ###########################################################
######################################################################################################################################################

# Set working matrices
train_matrix <- scale(data.matrix(features_train))
test_matrix <- scale(data.matrix(features_test))
y_train <- data.matrix(labels_train)
y_test <- data.matrix(labels_test)

# Prepare storage
seeds <- 1:100
results <- data.frame()
features_gt1 <- c()
features_lt1 <- c()


plotname <- 'Olink_Baseline'
##################################################
########## RUN LOOP TO CHOOSE LAMBDA MIN #########
##################################################
# Run LASSO for each seed
for (seed in seeds) {
  set.seed(seed)
  
  lasso_cv <- cv.glmnet(train_matrix, y_train, alpha = 1, family = 'binomial', nfolds = 5)
  lambda_min <- lasso_cv$lambda.min
  
  coef_min <- coef(lasso_cv, s = lambda_min)
  nonzero_idx <- which(coef_min != 0)
  or_values <- exp(coef_min[nonzero_idx])

  feature_names <- rownames(coef_min)[nonzero_idx]
  gt1 <- feature_names[or_values > 1 & feature_names != "(Intercept)"]
  lt1 <- feature_names[or_values < 1 & feature_names != "(Intercept)"]

  features_gt1 <- c(features_gt1, paste(gt1, collapse = ", "))
  features_lt1 <- c(features_lt1, paste(lt1, collapse = ", "))
  
  # Final model
  final_model <- glmnet(train_matrix, y_train, alpha = 1, lambda = lambda_min, family = 'binomial')
  
  pred_train <- predict(final_model, newx = train_matrix, type = "response")
  pred_test <- predict(final_model, newx = test_matrix, type = "response")

  acc_train <- mean(ifelse(pred_train > 0.5, 1, 0) == y_train)
  acc_test <- mean(ifelse(pred_test > 0.5, 1, 0) == y_test)

  auc_train <- pROC::roc(y_train, pred_train)$auc
  auc_test <- pROC::roc(y_test, pred_test)$auc

  pval_train <- wilcox.test(pred_train ~ y_train)$p.value
  pval_test <- wilcox.test(pred_test ~ y_test)$p.value

  results <- rbind(results, data.frame(
    seed = seed,
    lambda = lambda_min,
    pval_train = pval_train,
    acc_train = acc_train,
    auc_train = auc_train,
    acc_test = acc_test,
    auc_test = auc_test,
    num_features = length(nonzero_idx) - 1,
    features_gt1 = paste(gt1, collapse = ", "),
    features_lt1 = paste(lt1, collapse = ", ")
  ))


}

write.csv(results, paste0(input_path, "lasso_summary.csv"), row.names = FALSE)




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