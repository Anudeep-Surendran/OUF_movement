#.........Code to construct variogram using ctmm package in R...................

# loading required packages
library(ctmm) 
library(dplyr)
library(here)

# Import animal movement data and convert to `telemetry` format for ctmm analysis
# We use movement tracking data of lowland tapir (Tapirus terrestris) from Medici et al., 2022 Movement Ecology, 10(1):14. https://doi.org/10.1186/s40462-022-00313-w and the data can be downloaded from the SI of this article.
myAnimals <- read.csv('pantanal.csv') %>%
  as.telemetry(timeformat = '%Y-%m-%d %H:%M', mark.rm = TRUE)

# Choosing an animal for analysis
Cilla <-myAnimals[[19]] 

# Plotting variogram
vg.Cilla <- variogram(Cilla) 

# Generating initial parameter guesses for model fitting.
# These can be adjusted visually if needed, but the default values generally work well.
variogram.fit(vg.Cilla,fraction=0.002) 

# Fitting and selecting the best model
fitted.mods <- ctmm.select(Cilla, CTMM=GUESS, verbose=TRUE) 

# Selected best fit model
ouf <- fitted.mods [[1]] 

# Saving the figure into a pdf file
pdf(
  file = here("fig1b.pdf"),
  width = 6,
  height = 5
)

# Plotting the results
plot(vg.Cilla, CTMM=ouf, col.CTMM="black", fraction=0.002)  

dev.off()