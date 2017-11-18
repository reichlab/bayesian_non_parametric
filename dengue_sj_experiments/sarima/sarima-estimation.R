library(lubridate)
library(forecast)
library(ggplot2)
library(readr)

### TRAIN MODEL

## Load data for Dengue fever in San Juan
data <- read_csv("dengue_sj_experiments/San_Juan_Training_Data.csv")

## Restrict to data from 1990/1991 through 2004/2005 seasons
train_seasons <- paste0(1990:2004, "/", 1991:2005)
data <- data[data$season %in% train_seasons, ]

## Form variable with total cases + 1 which can be logged
data$total_cases_plus_1 <- data$total_cases + 1

## convert dates
data$time <- ymd(data$week_start_date)

## Add time_index column.  This is used for calculating the periodic kernel.
## Here, this is calculated as the number of days since some origin date (1970-1-1 in this case).
## The origin is arbitrary.
data$time_index <- as.integer(data$time -  ymd(paste("1970", "01", "01", sep = "-")))

prediction_target_var <- "total_cases_plus_1"

data <- as.data.frame(data)

log_prediction_target <- log(data[, prediction_target_var])


seasonally_differenced_log_prediction_target <-
  ts(log_prediction_target[seq(from = 53, to = length(log_prediction_target))] -
      log_prediction_target[seq(from = 1, to = length(log_prediction_target) - 52)],
    frequency = 52)

seasonally_differenced_log_sarima_fit <-
  auto.arima(seasonally_differenced_log_prediction_target)




### GET PREDICTIONS

data <- read_csv("dengue_sj_experiments/San_Juan_Training_Data.csv")

## Restrict to data from 1990/1991 through 2008/2009 seasons
train_seasons <- paste0(1990:2008, "/", 1991:2009)
data <- data[data$season %in% train_seasons, ]

## Form variable with total cases + 1 which can be logged
data$total_cases_plus_1 <- data$total_cases + 1

## convert dates
data$time <- ymd(data$week_start_date)

## Add time_index column.  This is used for calculating the periodic kernel.
## Here, this is calculated as the number of days since some origin date (1970-1-1 in this case).
## The origin is arbitrary.
data$time_index <- as.integer(data$time -  ymd(paste("1970", "01", "01", sep = "-")))

prediction_target_var <- "total_cases_plus_1"

data <- as.data.frame(data)

log_prediction_target <- log(data[, prediction_target_var])


seasonally_differenced_log_prediction_target <-
  ts(log_prediction_target[seq(from = 53, to = length(log_prediction_target))] -
       log_prediction_target[seq(from = 1, to = length(log_prediction_target) - 52)],
     frequency = 52)




## predictions_df contains combinations of season/season_week/prediction_horizon for which we want predictions
# predictions_df <- data.frame(ph=rep(seq_len(52), times = 3 * 52),
# 	last_obs_season=rep(c("2005/2006", "2006/2007", "2008/2009"), each = 52, times = 52),
# 	last_obs_week=rep(seq_len(52) - 1, each = 52 * 3),
# 	model="sarima",
# 	stringsAsFactors=FALSE)
sarima_ph <- 10
predictions_df <- data.frame(ph=rep(sarima_ph, times = 1 * 1),
  last_obs_season=rep(c("2005/2006"), each = 52, times = 1),
  last_obs_week=seq_len(52) - sarima_ph,
  model="sarima",
  stringsAsFactors=FALSE)
inds_adj <- predictions_df$last_obs_week < 0
predictions_df$last_obs_season[inds_adj] <- "2004/2005"
predictions_df$last_obs_week[inds_adj] <- predictions_df$last_obs_week[inds_adj] + 52

#predictions_df <- data.frame(ph=rep(seq_len(52), times = 1 * 1),
#  last_obs_season=rep(c("2005/2006"), each = 52, times = 1),
#  last_obs_week=rep(0, each = 52 * 1),
#  model="sarima",
#  stringsAsFactors=FALSE)

predictions_df$prediction_season <- predictions_df$last_obs_season
predictions_df$prediction_week <- predictions_df$last_obs_week + predictions_df$ph

inds_last_obs_season_prev_year <- which(predictions_df$last_obs_week == 0)
predictions_df$last_obs_season[inds_last_obs_season_prev_year] <-
	sapply(predictions_df$last_obs_season[inds_last_obs_season_prev_year],
		function(next_season) {
			start_year <- as.integer(substr(next_season, 1, 4)) - 1L
			paste0(start_year, "/", start_year + 1)
		}
	)
predictions_df$last_obs_week[inds_last_obs_season_prev_year] <- 52L

inds_prediction_season_next_year <- which(predictions_df$prediction_week > 52)
predictions_df$prediction_season[inds_prediction_season_next_year] <-
	sapply(predictions_df$prediction_season[inds_prediction_season_next_year],
		function(next_season) {
			start_year <- as.integer(substr(next_season, 1, 4)) + 1L
			paste0(start_year, "/", start_year + 1)
		}
	)
predictions_df$prediction_week[inds_prediction_season_next_year] <-
	predictions_df$prediction_week[inds_prediction_season_next_year] - 52L

predictions_df$log_score <- NA
predictions_df$prediction <- NA
predictions_df$AE <- NA
predictions_df$predictive_50pct_lb <- NA
predictions_df$predictive_50pct_ub <- NA
predictions_df$predictive_90pct_lb <- NA
predictions_df$predictive_90pct_ub <- NA
predictions_df$week_start_date <- data$week_start_date[1]



#sj_log_sarima_fit <- sj_log_sarima_fit_manual1  ## very bad
#sj_log_sarima_fit <- sj_log_sarima_fit_manual2  ## quite bad
#sj_log_sarima_fit <- sj_log_sarima_fit_manual3  ## very bad

sarima_inds <- which(predictions_df$model == "sarima")
#sarima_inds <- which(predictions_df$model == "sarima" & predictions_df$ph %in% c(1, 13, 26, 39, 52))
for(predictions_df_row_ind in sarima_inds) {
	ph <- as.numeric(predictions_df$ph[predictions_df_row_ind])
	last_obs_ind <- which(data$season == predictions_df$last_obs_season[predictions_df_row_ind] &
		data$season_week == predictions_df$last_obs_week[predictions_df_row_ind])

	predictions_df$week_start_date[predictions_df_row_ind] <- data$week_start_date[last_obs_ind + as.numeric(ph)]


#	new_data <- ts(log(San_Juan_test$total_cases[seq_len(last_obs_ind)] + 1), frequency = 52)
#	updated_sj_log_sarima_fit <- Arima(new_data, model = sj_log_sarima_fit)


	new_data <- log(data$total_cases[seq_len(last_obs_ind)] + 1)
	seasonal_diff_new_data <- ts(new_data[seq(from = 53, to = length(new_data))] -
	    new_data[seq(from = 1, to = length(new_data) - 52)], frequency = 52)
	updated_sj_log_sarima_fit <- Arima(seasonal_diff_new_data, model = seasonally_differenced_log_sarima_fit)

	predict_result <- predict(updated_sj_log_sarima_fit, n.ahead = ph)

#	predictive_log_mean <- as.numeric(predict_result$pred[ph])
#	predictions_df$prediction[predictions_df_row_ind] <- exp(predictive_log_mean) - 1

	predictive_log_mean <- as.numeric(predict_result$pred[ph]) + new_data[last_obs_ind + ph - 52]
	predictions_df$prediction[predictions_df_row_ind] <- exp(as.numeric(predict_result$pred[ph]) + new_data[last_obs_ind + ph - 52]) - 1


	predictions_df$AE[predictions_df_row_ind] <- abs(predictions_df$prediction[predictions_df_row_ind] - data$total_cases[last_obs_ind + ph])

		## THIS LOG SCORE IS NOT DISCRETIZED FOR COUNT DATA
	predictions_df$log_score[predictions_df_row_ind] <- dlnorm(data$total_cases[last_obs_ind + ph] + 1,
		meanlog = predictive_log_mean,
		sdlog = as.numeric(predict_result$se[ph]),
		log = TRUE)

	temp <- qlnorm(c(0.05, 0.25, 0.75, 0.95),
		meanlog = predictive_log_mean,
		sdlog = as.numeric(predict_result$se[ph]))
	predictions_df[predictions_df_row_ind, c("predictive_90pct_lb", "predictive_50pct_lb", "predictive_50pct_ub", "predictive_90pct_ub")] <-
		temp - 1
}

predictions_df$ph <- as.factor(predictions_df$ph)

ggplot() +
	geom_line(aes(x = week_start_date, y = total_cases), data = data[data$season %in% c("2003/2004", "2004/2005", "2005/2006"),]) +
	geom_line(aes(x = week_start_date, y = prediction), color = "red", data = predictions_df) +
	theme_bw()
