theirs <- read_excel("Downloads/12864_2018_5118_MOESM4_ESM.xlsx")

setwd("~/Documents/Fecundity/Fecundity-Classifier/2.Testing")
# x <- read.csv("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_complete_CD_results/Alex_FecundityModelMoDataV1_sums_COMPLETE_CD.csv")
# x <- read.csv("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/Alex_4-30_5-1_CC_A_v0.0_sums_COMPLETE_CD.csv")
# x <- read.csv("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_lithium_5-4_results/Alex_5-1_5-2S_v0.0_sums__lith54_CSV.csv")
x <- read.csv("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_lithium_5-4_results/Alex_5-1_5-2S_v0.0_tile_counts_lith.csv")
x$dif <- x$BotSum - x$HumanSum
x$absDif <- abs(x$dif)

cor(x$BotSum, x$HumanSum)
cor(x$BotSum[x$dif < 30], x$HumanSum[x$dif < 30])

mean(x$BotSum - x$HumanSum)

plot(x$BotSum, x$HumanSum)

library(ggplot2)
library(ggpubr)
library(dplyr)

x$sel <- 1
x$rep <- 1
x$month <- 1
x$day <- 1
for (i in 1:length(x$RootImage)) {
  x$sel[i] <- strsplit(strsplit(x$RootImage[i], " ")[[1]][3], "")[[1]][1]
  x$rep[i] <- as.integer(strsplit(strsplit(x$RootImage[i], " ")[[1]][3], "")[[1]][2])
  x$month[i] <- as.integer(strsplit(x$RootImage[i], " ")[[1]][1])
  x$day[i] <- as.integer(strsplit(x$RootImage[i], " ")[[1]][2])
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

library(lme4)
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

