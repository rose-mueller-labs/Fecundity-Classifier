library(readxl)

theirs <- read_excel("Downloads/12864_2018_5118_MOESM4_ESM.xlsx")

setwd("~/Documents/Fecundity/Fecundity-Classifier/2.Testing")
# x <- read.csv("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_complete_CD_results/Alex_FecundityModelMoDataV1_sums_COMPLETE_CD.csv")
# x <- read.csv("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/Alex_4-30_5-1_CC_A_v0.0_sums_COMPLETE_CD.csv")
# x <- read.csv("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_lithium_5-4_results/Alex_5-1_5-2S_v0.0_sums__lith54_CSV.csv")
x <- read.csv("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_complete_CD_results/Alex_4-30_5-1_CC_A_v0.0_sums_COMPLETE_CD.csv")
x2 <- read.csv("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_complete_CD_results/Alex_5-1_5-2S_CC_A_v0.0_sums_COMPLETE_CD.csv")
x$dif <- x$BotSum - x$HumanSum
x$absDif <- abs(x$dif)
x2$dif <- x2$BotSum - x2$HumanSum
x2$absDif <- abs(x2$dif)


cor(x$BotSum, x$HumanSum)
cor(x$BotSum[x$dif < 30], x$HumanSum[x$dif < 30])

mean(x$BotSum - x$HumanSum)

plot(x$BotSum, x$HumanSum)

library(ggplot2)
library(ggpubr)
library(dplyr)
library(lme4)

x$sel <- 1
x$rep <- 1
x$month <- 1
x$day <- 1
for (i in 1:length(x$CD_RootImage)) {
  x$sel[i] <- strsplit(strsplit(x$CD_RootImage[i], " ")[[1]][3], "")[[1]][1]
  x$rep[i] <- as.integer(strsplit(strsplit(x$CD_RootImage[i], " ")[[1]][3], "")[[1]][2])
  x$month[i] <- as.integer(strsplit(x$CD_RootImage[i], " ")[[1]][1])
  x$day[i] <- as.integer(strsplit(x$CD_RootImage[i], " ")[[1]][2])
}
x$age <- (x$month - 2) * 28 + x$day
x$age <- x$age - as.integer(x$rep) - 5
x$ageRepSel <- paste(x$age, x$rep, x$sel)
  
c <- x$dif < 30
ggplot(x, mapping = aes(x = HumanSum, y = BotSum,
                        color = sel,
                        group = sel)) +
  geom_point(alpha = 0.6) +
  theme_light() +
  scale_color_manual(values = c("purple", "forestgreen")) +
  geom_smooth(method = "lm", color = "red") +
  xlim(0, 125) +
  ylim(0, 150) 

test <- x %>% 
  group_by(ageRepSel) %>% 
  reframe(HumanSum = mean(HumanSum), across(c(1:10)))

#[x$absDif < 50,]
test <- x %>% 
  group_by(ageRepSel) %>% 
  reframe(HumanSum = mean(HumanSum), BotSum = mean(BotSum), across(c(2)))
test$age <- 1
test$rep <- 1
test$sel <- 1
for (i in 1:140) {
  test$age[i] <- strsplit(test$ageRepSel[i], " ")[[1]][1]
  test$rep[i] <- strsplit(test$ageRepSel[i], " ")[[1]][2]
  test$sel[i] <- strsplit(test$ageRepSel[i], " ")[[1]][3]
}
test$age <- as.integer(test$age)
test$dif <- test$HumanSum - test$BotSum
test$absDif <- abs(test$HumanSum - test$BotSum)
test$interval <- floor(test$age / 3)
test$intervalSel <- paste(test$interval, test$sel)
test$repSel <- paste(test$rep, test$sel)

mean(x$HumanSum[x$ageRepSel == "21 4 C"])
mean(x$HumanSum[x$absDif < 100 & x$ageRepSel == "21 4 C"])

g1 <- ggplot(test, 
       mapping = aes(x = age, y = BotSum,
                        color = sel,
                        group = intervalSel)) +
  geom_point(alpha = 1) +
  theme_light() +
  scale_color_manual(values = c("purple", "forestgreen")) +
  geom_smooth(method = "lm", se = TRUE) +
  ylim(0, 79)

g2 <- ggplot(test, 
             mapping = aes(x = age, y = HumanSum,
                           color = sel,
                           group = intervalSel)) +
  geom_point(alpha = 1) +
  theme_light() +
  scale_color_manual(values = c("purple", "forestgreen")) +
  geom_smooth(method = "lm", se = TRUE) +
  ylim(0, 79)

ggarrange(g1, g2, nrow = 1)


car::Anova(lm(data = test[test$interval == "6",], 
              HumanSum ~ age * sel))
car::Anova(lmer(data = test[test$interval == "6",], 
              HumanSum ~ age + sel + (1 | repSel)))
car::Anova(lmer(data = test[test$interval == "6",], 
                BotSum ~ age + sel + (1 | repSel)))
summary(lmer(data = test[test$interval == "6",], 
             HumanSum ~ age + sel + (1 | repSel)))


lm(data = x[x$absDif < 100,], BotSum ~ CorD)
summary(lm(data = x[x$absDif < 100,], BotSum ~ CorD))
car::Anova(lm(data = x, BotSum ~ CorD))



ggqqplot(x$dif)
hist(x$dif)

silly <- x %>% count(dif)
ggplot(silly, mapping = aes(dif, n)) +
  theme_light() +
  geom_point() +
  xlim(-55, 150)



plot(x$BotSum, x2$BotSum)
plot(x$BotSum, x2$HumanSum)

cor(x$BotSum, x2$BotSum)
cor(x$BotSum, x2$HumanSum)
cor(x2$BotSum, x2$HumanSum)

cor(x$absDif, x2$absDif)
ggplot(mapping = aes(x = x$absDif, y = x2$absDif)) +
  geom_point() +
  theme_light() +
  ylim(0, 75) +
  xlim(0, 75)

cor(x$dif, x2$dif)
ggplot(mapping = aes(x = x$dif, y = x2$dif)) +
  geom_point() +
  theme_light()

cor((x$BotSum + x2$BotSum) / 2, x$HumanSum)

ggplot(mapping = aes(x = x$HumanSum, y = (x$BotSum + x2$BotSum) / 2)) +
  geom_point(alpha = 0.2) +
  theme_light() +
  xlim(0, 120) +
  ylim(0, 150) +
  geom_smooth(color = "red", method = "lm")

ggplot(mapping = aes(x = x$HumanSum, y = x$BotSum)) +
  geom_point(alpha = 0.2) +
  theme_light() +
  xlim(0, 120) +
  ylim(0, 150) +
  geom_smooth(color = "red", method = "lm")


twoModel <- data.frame(RootImage = x$CD_RootImage, HumanSum = x$HumanSum,
                       first = x$BotSum, firstDif = x$dif,
                       second = x2$BotSum, secondDif = x2$dif,
                       mean = (x$BotSum + x2$BotSum) / 2)
twoModel$meanDif <- twoModel$HumanSum - twoModel$mean

twoModel$sel <- 1
twoModel$rep <- 1
twoModel$month <- 1
twoModel$day <- 1
twoModel$expDay <- 1
for (i in 1:length(twoModel$RootImage)) {
  twoModel$sel[i] <- strsplit(strsplit(twoModel$RootImage[i], " ")[[1]][3], "")[[1]][1]
  twoModel$rep[i] <- as.integer(strsplit(strsplit(twoModel$RootImage[i], " ")[[1]][3], "")[[1]][2])
  twoModel$month[i] <- as.integer(strsplit(twoModel$RootImage[i], " ")[[1]][1])
  twoModel$day[i] <- as.integer(strsplit(twoModel$RootImage[i], " ")[[1]][2])
  twoModel$expDay[i] <- paste(as.integer(strsplit(twoModel$RootImage[i], " ")[[1]][1]),
                           as.integer(strsplit(twoModel$RootImage[i], " ")[[1]][2]))
}
twoModel$age <- (twoModel$month - 2) * 28 + twoModel$day
twoModel$age <- twoModel$age - as.integer(twoModel$rep) - 5
twoModel$ageRepSel <- paste(twoModel$age, twoModel$rep, twoModel$sel)
twoModel$interval <- floor(twoModel$age / 3)
twoModel$intervalSel <- paste(twoModel$interval, twoModel$sel)


c <- abs(twoModel$firstDif) < 90
test <- twoModel[c,] %>% 
  group_by(ageRepSel) %>% 
  reframe(Human = mean(HumanSum), Bot1 = mean(first), Bot2 = mean(second))
test$age <- 1
test$rep <- 1
test$sel <- 1
for (i in 1:140) {
  test$age[i] <- strsplit(test$ageRepSel[i], " ")[[1]][1]
  test$rep[i] <- strsplit(test$ageRepSel[i], " ")[[1]][2]
  test$sel[i] <- strsplit(test$ageRepSel[i], " ")[[1]][3]
}
test$age <- as.integer(test$age)
test$interval <- floor(test$age / 3)
test$intervalSel <- paste(test$interval, test$sel)
test$repSel <- paste(test$rep, test$sel)
test$BotMean <- (test$Bot1 + test$Bot2) / 2

g1 <- ggplot(test, mapping = aes(x = age, y = Human,
                           color = sel, group = intervalSel)) +
  geom_point(alpha = 1) +
  theme_light() +
  scale_color_manual(values = c("purple", "forestgreen")) +
  geom_smooth(method = "lm", se = TRUE) +
  ylim(0, 79)
g2 <- ggplot(test, mapping = aes(x = age, y = Bot1,
                                 color = sel, group = intervalSel)) +
  geom_point(alpha = 1) +
  theme_light() +
  scale_color_manual(values = c("purple", "forestgreen")) +
  geom_smooth(method = "lm", se = TRUE) +
  ylim(0, 79)
g3 <- ggplot(test, mapping = aes(x = age, y = Bot2,
                                 color = sel, group = intervalSel)) +
  geom_point(alpha = 1) +
  theme_light() +
  scale_color_manual(values = c("purple", "forestgreen")) +
  geom_smooth(method = "lm", se = TRUE) +
  ylim(0, 79)
g4 <- ggplot(test, mapping = aes(x = age, y = (Bot1 + Bot2) / 2,
                                 color = sel, group = intervalSel)) +
  geom_point(alpha = 1) +
  theme_light() +
  scale_color_manual(values = c("purple", "forestgreen")) +
  geom_smooth(method = "lm", se = TRUE) +
  ylim(0, 79)

ggarrange(g1, g2, g3, g4, nrow = 2, ncol = 2)

mean(test$BotMean)

c <- abs(twoModel$firstDif) < 80
hist(twoModel$firstDif[c])
hist(twoModel$secondDif[c])
hist(twoModel$meanDif[c])

ggpllo

