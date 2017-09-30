library(lubridate)
library(forecast)
library(ggplot2)
library(car)


# TRAIN MODEL

## Load data for Dengue fever in San Juan
dengue_data <- read_csv("dengue_sj_experiments/San_Juan_Training_Data.csv")

## Restrict to data from 1990/1991 through 2004/2005 seasons
train_seasons <- paste0(1990:2004, "/", 1991:2005)
dengue_data <- dengue_data[dengue_data$season %in% train_seasons, ]

## Transform via Yeo-Johnson transform (similar to Box-Cox, slightly more flexible)
est_power <- car::powerTransform(dengue_data$total_cases, family = "yjPower")
dengue_data$total_cases_yj <- yjPower(U = dengue_data$total_cases, lambda = coef(est_power))

## Subtract mean from Y-J transformed values
dengue_data$total_cases_yj_centered <- dengue_data$total_cases_yj - mean(dengue_data$total_cases_yj)


## Make some comparative plots
### original
ggplot() +
  geom_point(aes(x = week_start_date, y = total_cases), data = dengue_data)

ggplot() +
  geom_density(aes(x = total_cases), data = dengue_data)

### Y-J transformed
ggplot() +
  geom_point(aes(x = week_start_date, y = total_cases_yj), data = dengue_data)

ggplot() +
  geom_density(aes(x = total_cases_yj), data = dengue_data)

### Y-J transformed and then centered
ggplot() +
  geom_point(aes(x = week_start_date, y = total_cases_yj_centered), data = dengue_data)

ggplot() +
  geom_density(aes(x = total_cases_yj_centered), data = dengue_data)

