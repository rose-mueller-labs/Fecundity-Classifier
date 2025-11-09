setwd("~/Documents/Fecundity/Fecundity-Classifier/2.Testing")
x <- read.csv("~/Documents/Fecundity/Fecundity-Classifier/2.Testing/Alex_FecundityModelMoDataV1_sums_COMPLETE_CD.csv")

plot(x$BotSum, x$HumanSum)

library(ggplot2)

x$CorD <- 1
for (i in 1:7901)
  x$CorD[i] <- strsplit(strsplit(x$CD_RootImage[i], " ")[[1]][3], "")[[1]][1]

c <- x$dif < 30
ggplot(x, mapping = aes(x = HumanSum, y = BotSum,
                        color = CorD)) +
  geom_point(alpha = 0.6) +
  theme_light() +
  scale_color_manual(values = c("purple", "forestgreen")) +
  geom_smooth(method = "lm", color = "red") +
  xlim(0, 125) +
  ylim(0, 150) 

cor(x$BotSum, x$HumanSum)
cor(x$BotSum[x$dif < 30], x$HumanSum[x$dif < 30])

mean(x$BotSum - x$HumanSum)

x$dif <- x$BotSum - x$HumanSum
