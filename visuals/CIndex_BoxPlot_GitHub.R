setwd("/Users/lynchmt/Documents/HPVctDNA-local")

library(ggplot2)
library(ggpubr)



## Create themes
probs_theme <- function() {
  theme_minimal() %+replace%
    theme(
      plot.title=element_text(hjust=0.5, size=12, face="bold", margin = margin(t = 0, r = 0, b = 5, l = 0)),
      legend.position = "none",
      axis.text.y = element_text(size=10),
      axis.text.x = element_text(size=10, face="bold", angle = 45),
      axis.title.y = element_text(size=12, angle = 90, margin = margin(t = 0, r = 5, b = 0, l = 0), face="bold"),
      axis.title.x = element_text(size=12, margin = margin(t = 5, r = 0, b = 0, l = 0), face="bold"),
      axis.line = element_blank(),
      axis.line.x.bottom = element_line(size = .3, color = "black"),
      axis.line.y.left = element_line(size = .3, color = "black")
      
    )
}




### == MAIN FUNCTION == ###
boxplots <- function(df_ctDNA, df_NOctDNA, biomarker, direction) {

  print(paste("Function - biomarker: ", biomarker))
  df_ct <- df_ctDNA[df_ctDNA$NumFeatures == "1" & df_ctDNA['HR...1.Features'] == 'transf HPV16/18 copies per ml of plasma D1', ]
  
  df_ct['HR...1.Features'][df_ct['HR...1.Features'] == "transf HPV16/18 copies per ml of plasma D1"] <- "transf ctDNA"
  
  df_biomarker <- df_NOctDNA[df_NOctDNA$NumFeatures == "1" & df_NOctDNA[[direction]] == biomarker, ]
  
  if (direction == 'HR...1.Features'){
    biomark_ct <- df_ctDNA[df_ctDNA$NumFeatures == "2" & df_ctDNA['HR...1.Features'] == paste("transf HPV16/18 copies per ml of plasma D1,", biomarker), ]
    biomark_ct['HR...1.Features'][biomark_ct['HR...1.Features'] ==  paste("transf HPV16/18 copies per ml of plasma D1,", biomarker)] <- paste("transf ctDNA, ", biomarker)
  }
  if (direction == 'HR...1.Features.1'){
    biomark_ct <- df_ctDNA[df_ctDNA$NumFeatures == "2" & df_ctDNA[[direction]] == biomarker, ]
    biomark_ct['HR...1.Features'][biomark_ct['HR...1.Features'] ==  "transf HPV16/18 copies per ml of plasma D1"] <- "transf ctDNA"
  }
  
  
  print(paste("MAX CINDEX FOR TWO FEATURE MODEL TEST SET: ",max(biomark_ct$Test_Set_Concordance)))

  print(paste("MAX CINDEX FOR TWO FEATURE MODEL TRAIN SET: ",max(biomark_ct$Validation_Set_Concordance)))
  
  models <- rbind(df_ct, df_biomarker, biomark_ct)
  
  ## Combine feature columns for plotting
  a <- as.character(models$HR...1.Features)
  b <- as.character(models$HR...1.Features.1)
  
  # Row-wise combine
  models$Features <- ifelse(nzchar(a) & nzchar(b), paste(a, b, sep = "+"), ifelse(nzchar(a), a, b))

  # my_comparisons <- list(c('R', 'NR'))
  png(filename = paste("outputs/visuals/CIndex_BoxPlots/10_23_25/CIndex_", biomarker, "_ctDNA_Validation.png"), width=6, height=6, units="in", res=250)
  
  p1 <- ggplot(models, aes(x = Features, 
                     y = as.numeric(Validation_Set_Concordance), fill = Features)) +
    geom_boxplot(outlier.shape = NA) +
    geom_jitter(size = 1.5, height = 0.01, width = 0.2) +
    ylab("Validation C-Index") + xlab("") +
    scale_x_discrete() + probs_theme()
  
  print(p1)
  dev.off()
  
  
  png(filename = paste("outputs/visuals/CIndex_BoxPlots/10_23_25/CIndex_", biomarker, "_ctDNA_Test.png"), width=6, height=6, units="in", res=250)
  p2 <- ggplot(models, aes(x = Features, 
                     y = as.numeric(Test_Set_Concordance), fill = Features)) +
    geom_boxplot(outlier.shape = NA) +
    geom_jitter(size = 1.5, height = 0.01, width = 0.2) +
    ylab("Test C-Index") + xlab("") +
    scale_x_discrete() + probs_theme()
  
  print(p2)
  
  dev.off()
  
  

}







## Load data - load 3-fold, 5-fold, clinical and non-clinical, and then row bind

## Set variables

direction_gt = 'HR...1.Features'
direction_lt = 'HR...1.Features.1'
#biomarker = 'Age'
df_NOctDNA <- data_NOctDNA
df_ctDNA <- data_ctDNA

HR_gt_1 <- c("IL6", "IL8", "Neutrophil Abs  D1", "MUC-16", "CSF-1", "TIE2", "HGF", "ICI_num")
HR_lt_1 <- c("TNF", "CD8A", "GZMA", "GZMB", "GZMH", "CXCL5", "Lymphocytes.Abs..D1", "Age")


for (each in HR_gt_1){
  print(each)
  boxplots(data_ctDNA, data_NOctDNA, each, direction_lt)
}




## Make 1 dataframe (rbind) for plotting

## Median = upper and lower??


