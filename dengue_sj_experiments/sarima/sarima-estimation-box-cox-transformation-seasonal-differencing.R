library(plyr)
library(dplyr)
library(lubridate)
library(forecast)
library(ggplot2)
library(readr)
library(sarimaTD)

### TRAIN MODEL

## Load data for Dengue fever in San Juan
san_juan_dengue <- read_csv("dengue_sj_experiments/San_Juan_Training_Data.csv")

## Train on data from 1990/1991 through 2004/2005 seasons
train_seasons <- paste0(1990:2004, "/", 1991:2005)
train_indices <- (san_juan_dengue$season %in% train_seasons)

sarimaTD_fit <- fit_sarima(
  y = san_juan_dengue$total_cases[train_indices],
  ts_frequency = 52,
  transformation = "box-cox",
  seasonal_difference = TRUE)



### GET PREDICTIONS

## predictions_df contains combinations of season/season_week/prediction_horizon for which we want predictions
prediction_times <- expand.grid(
  last_obs_season = c("2004/2005", "2005/2006", "2006/2007", "2007/2008", "2008/2009"),
  last_obs_week = seq_len(52),
  stringsAsFactors = FALSE
)
prediction_time_ind <- 1L

incidence_bins <- data.frame(
  start = seq(from = 0, to = 1000, by = 1) - 0.5,
  end = seq(from = 0, to = 1000, by = 1) + 0.5
)
num_bins <- nrow(incidence_bins)

nsim <- 10^5

for(prediction_time_ind in seq_len(nrow(prediction_times))) {
	last_obs_ind <- which(
    san_juan_dengue$season == prediction_times$last_obs_season[prediction_time_ind] &
		san_juan_dengue$season_week == prediction_times$last_obs_week[prediction_time_ind])

  # get nsim simulated trajectories of incidence for next 52 weeks
  sampled_trajectories <- simulate(
      object = sarimaTD_fit,
      nsim = nsim,
      seed = 1,
      newdata = san_juan_dengue$total_cases[seq_len(last_obs_ind)],
      h = 52)

  predictions_cdc_format <- lapply(
      seq_len(52),
      function(ph) {
        ph_preds <- data.frame(
          Location = "San Juan",
          Target = paste0(ph, " wk ahead"),
          Type = c("Point", rep("Bin", num_bins)),
          Unit = "total cases",
          Bin_start_incl = c(NA, incidence_bins$start),
          Bin_end_notincl = c(NA, incidence_bins$end),
          Value = NA,
          Log_value = NA
        )
        
        ph_preds$Value[1] <- median(sampled_trajectories[, ph])
        ph_preds$Value[1 + seq_len(num_bins)] <-
          sapply(seq_len(num_bins),
            function(bin_ind) {
              sum(sampled_trajectories[, ph] >= incidence_bins$start[bin_ind] &
                sampled_trajectories[, ph] < incidence_bins$end[bin_ind]) / nsim
              })
        ph_preds$Log_value[1 + seq_len(num_bins)] <-
          sapply(seq_len(num_bins),
            function(bin_ind) {
              log(sum(sampled_trajectories[, ph] >= incidence_bins$start[bin_ind] &
                sampled_trajectories[, ph] < incidence_bins$end[bin_ind])) -
                log(nsim)
              })
        return(ph_preds)
      }
    ) %>%
    plyr::rbind.fill()
  
  write_csv(predictions_cdc_format,
    path = paste0("dengue_sj_experiments/sarima/eval_predictions_box_cox_transformation_seasonal_differencing/last_obs_",
      gsub("/", "-", prediction_times$last_obs_season[prediction_time_ind]),
      "_week_",
      prediction_times$last_obs_week[prediction_time_ind],
      ".csv"
      ))
}
