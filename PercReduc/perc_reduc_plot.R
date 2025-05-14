library(ggplot2)

Anova(modle)

data = read.csv("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/PercReduc/sample_mses.csv")
ggplot(data, mapping=aes(x=X, y=Y,color="firebrick")) +
  geom_point() +
  geom_line() +
  geom_smooth(color ="black")

x

ggsave("filename", plot = x, path = "", device = "png", dpi = 300, units = "in", width = "7", height "4")

library(tseries)

acf(data$Y, pl=TRUE)
plot(acf(data$Y, pl=FALSE, lag=96))
