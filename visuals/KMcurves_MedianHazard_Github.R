library(ggplot2)
library(tidyr)
library(rstatix)
library(reshape2)
library(ggpubr)
library(gridExtra)
library(pROC)
library(nnet)
library(ROCR)
library(survival)
library(survminer)
library(tidyverse)
library(survcomp)



## Functions
create_low_high_list <- function(feature_data, cutoff) { 
  
  ## feature_data: a numeric vector of feature values (e.g., dataframe[[feature]])
  ## cutoff: a value between 0 and 1 indicating the percentile cutoff (e.g., 0.5 = median)
  
  threshold <- quantile(feature_data, probs = cutoff, na.rm = TRUE)
  low_high_list <- ifelse(feature_data > threshold, 'Upper', 'Lower')
  return(list(labels = low_high_list, threshold = threshold))
  
}

create_labels <- function(data_for_labels, df_followup) { 
  label_list <- c()
  names_split <- strsplit(names(data_for_labels), "=")
  group_names <- sapply(names_split, function(x) x[2])
  data_for_labels =  unlist(data_for_labels)
  
  for (i in 1:length(data_for_labels)){
    if (is.na(data_for_labels[i])){
      label = paste(group_names[i],": Median = not reached at ", max(df_followup), " days", sep="")
    }else{label = paste(group_names[i],": Median = ", data_for_labels[i], " days", sep="")}
    label_list <- c(label_list, label)
  }
  return(label_list)
}

## Create themes
custom_theme <- function() {
  theme_survminer() %+replace%
    theme(
      plot.title=element_text(hjust=0.5, face="bold", size=12),
      legend.key.height = unit(.4, 'cm'),
      legend.key.width = unit(.4, 'cm'),
      legend.text=element_text(size=15),
      legend.title=element_text(size=15),
      axis.text.y = element_text(size=16),
      axis.text.x = element_text(size=16),
      axis.title.y = element_text(size=17, angle = 90,  face="bold", margin = margin(t = 0, r = 6, b = 0, l = 0)),
      axis.title.x = element_text(size=17, face="bold"),
      
    )
}


run_survival_loop <- function(train_df, test_df, Feature_list, time, event, prefix) {
  
  surv = Surv(train_df$time, train_df$event_num)
  
  f <- as.formula(paste("surv ~", Features[1]))
  g <- as.formula(paste("surv ~",  Features[2]))
  h <- as.formula(paste("surv ~",  Features[1], "+",  Features[2]))
  
  ## FIT ALL MODELS ON TRAINING DATA
  res.cox.1 <- coxph(f, data = train_df)
  res.cox.2 <- coxph(g, data = train_df)
  res.cox.3 <- coxph(h, data = train_df)
  
  
  fit_list <- list(res.cox.1, res.cox.2, res.cox.3)
  num_features <- length(fit_list)

  probability_df <- data.frame("followup" = test_df[[time]], "dead" = test_df[[event]])
  
  for (i in seq_along(fit_list)) {
    preds <- predict(fit_list[[i]], type = "risk", newdata = test_df)
    probability_df[[paste0("RelativeRisk_", i)]] <- preds
    
    output_title <- gsub("\\.", "_", paste(names(coef(fit_list[[i]])), collapse="_"))
    
    # C-index with fallback
    ci_obj <- concordance.index(x = preds,
                                surv.time = test_df[[time]],
                                surv.event = as.numeric(as.logical(test_df[[event]])))
    c_index_value <- ci_obj$c.index
    if (is.na(c_index_value)) {
      surv_obj <- Surv(test_df[[time]], test_df[[event]])
      c_index_value <- 1 - (survConcordance(surv_obj ~ preds)$concordance)
    }
    cat(sprintf("%s - Model %d C-index: %.3f\n", output_title, i, c_index_value))
    
    # Stratify
    med_list <- create_low_high_list(as.numeric(unlist(preds)), .5)
    stratify_df <- data.frame("time" = test_df[[time]], "event" = test_df[[event]], 'group' = med_list$labels)
    
    fit <- surv_fit(Surv(time, event) ~ group, data = stratify_df)
    data_for_labels <- summary(fit)$table[,"median"]
    legend_labels <- create_labels(data_for_labels, test_df[[time]])
    
    # Plot
    plot <- ggsurvplot(fit, data = stratify_df, risk.table = TRUE, pval = TRUE, conf.int = TRUE,
                       ggtheme = custom_theme(), pval.size = 6, legend.labs = legend_labels,
                       tables.y.text = FALSE, pval.coord = c(8, 0.1))
    plot$plot <- plot$plot + guides(fill = guide_legend(nrow = 2),
                                    color = guide_legend(nrow = 2))
    
    combined_plot <- grid.arrange(plot$plot, plot$table, ncol = 1,
                                  heights = c(2/3, 1/3))
    
    ggsave(filename = paste0("outputs/visuals/KMCurves/10_23_25/", prefix,"_", output_title, "_Median.png"), plot = combined_plot, width = 5, height = 5.75)
  }
}






## load data - moved to the bottom 
test <- cbind(X_test_data, y_test)

## Modify event to create numeric values
values <- c(0, 1)
index <- c("False", "True")
train$event_num <- values[match(train$event, index)]
test$event_num <- values[match(test$event, index)]




### Define variables
Features <- c("transf.HPV16.18.copies.per.ml.of.plasma.D1", "IL6")
HR_gt_1 <- c("IL6", "IL8", "Neutrophil.Abs..D1", "MUC.16", "CSF.1", "TIE2", "HGF", "ICI_num")
HR_lt_1 <- c("TNF", "CD8A", "GZMA", "GZMB", "GZMH", "CXCL5", "Lymphocytes.Abs..D1", "Age")
Biomarker_list <- c(HR_gt_1, HR_lt_1)


for (each in Biomarker_list) {
  Features <- c("transf.HPV16.18.copies.per.ml.of.plasma.D1", each)
  print(each)
  print(Features)
  
  run_survival_loop(train, test, Feature_list = Features, time = "time", event = "event_num",  prefix = "TESTset")
  run_survival_loop(train, train, Feature_list = Features, time = "time", event = "event_num",  prefix = "TRAINset")
  combine <- rbind(train, test)
  run_survival_loop(train, combine, Feature_list = Features, time = "time", event = "event_num",  prefix = "COMBINEDset")
}





