library(EBImage)

temp1 = which(twoModel$secondDif[twoModel$HumanSum < 10] < 3)
temp2 = which(twoModel$firstDif[twoModel$HumanSum < 10] < 3)

sum(temp2 %in% temp1)

crap <- read.csv("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_complete_CD_results/Alex_4-30_5-2S_v0.0_sums_COMPLETE_CD.csv")
crap$absDif <- abs(crap$BotSum - crap$HumanSum)
crap$dif <- (crap$BotSum - crap$HumanSum)
hist(crap$absDif[crap$HumanSum < 20])
# eggs71.0count3 3 D4 16.jpg pt78.jpg
#69 is 6 bot count, 85 is 10 bot count, 75 is 4 bot count, 29 is 6 count, 86 is 6 count
image = (readImage("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/ALL_CD_CAPS-sliced/eggs70.0count2 28 C1 32.jpg pt60.jpg"))
display(image)

temp <- read.csv("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_complete_CD_results/Alex_4-30_5-1_CC_A_v0.0_tile_counts_CD_Complete.csv")
temp <- read.csv("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_complete_CD_results/Alex_5-1_5-2S_v0.0_tile_counts_CD_Complete.csv")

hist(temp$Bot)

temp %>%
  count(Bot)
which(temp$Bot == 42)

image = (readImage("/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/ALL_CD_CAPS-sliced/eggs51.0count3 7 D4 29.jpg pt13.jpg"))
display(image)
temp$CD_RootImage[436572]
temp$Sum[436572]
