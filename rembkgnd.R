library(EBImage)

args <- commandArgs(trailingOnly = TRUE)
print(args)

files = list.files(args, pattern = ".*\\.png")

for (path in files) {
  
  #reads image in and converts it to a grayscale image
  image = channel(readImage(paste(args, path, sep="/")), "gray")
  #display(image)
  
  #initial pass at filtering for large areas of darker coloration
  nmask = thresh(image < 0.3, w=20.5, h=20.5, offset=0.05)
  nmask = opening(nmask, makeBrush(3, shape='disc'))
  nmask = fillHull(nmask)
  nmask = bwlabel(nmask)
  thinky = computeFeatures.shape(nmask)
  nmask = rmObjects(nmask, which(thinky[,1] < max(thinky[,1])))
  nmask[nmask > 0] <- 1
  
  #applies original filter to big
  image3 <- image
  image3@.Data <- image@.Data * nmask
  #display(image3)
  
  #another weird image inversion
  image4 <- image3
  image4@.Data[image4@.Data == 0] <- 1
  image4@.Data <- 1 - image4@.Data
  image4@.Data[image4@.Data > 0.7] <- 1
  
  #redoes the original filter to expand the boarder to encompass more of the cap
  nmask = thresh(image4 < 0.7, w=20.5, h=20.5, offset=0.05)
  nmask = opening(nmask, makeBrush(3, shape='disc'))
  nmask = fillHull(nmask)
  nmask = bwlabel(nmask)
  nmask[nmask > 0] <- 1
  
  #display(nmask)
  #display(image * nmask)
  
  #then converts the final filter to a proper pure circle and overlays that to the original cap
  #computerFeatures in its multiple operations are the important thing
  x <- computeFeatures.moment(nmask)
  y <- computeFeatures.shape(nmask)
  bg <- Image(0, c(800, 800))
  circle <- drawCircle(bg, x = x[1], y = x[2], r = y[3], col = 1, fill = TRUE)
  
  #display(circle)
  capImage = image * circle
  #display(capImage)
  
  colorImage <- readImage(paste(args, path, sep="/"))
  colorCircle = c(circle, circle, circle)
  #display(colorImage * colorCircle)
  if (!file.exists(paste(args, "clean", sep="/"))){
    dir.create(file.path(args, "clean"), showWarnings = FALSE)
  }
  writeImage(colorImage * colorCircle, paste(args, paste("clean", path, sep="/"), sep="/"))
}

