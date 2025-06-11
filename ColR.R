df <- read.csv("/home/bule/TramDag/dev_experiment_logs/tramdagpaper_exp6_1_linearDGP_ls_9/tramdagpaper_exp6_1_linearDGP_ls_9.csv")



library(colr)




install.packages("colr")


model <- Colr(x2 ~ x1  data = df)
