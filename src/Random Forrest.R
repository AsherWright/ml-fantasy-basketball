library(tidyverse)
library(rpart) # Will use to construct a CART model.
library(rpart.plot) # Will use to plot CART tree.
library(randomForest)

data = read.csv("../data/games_7_players.csv")


