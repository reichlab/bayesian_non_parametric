predict_SARIMA_sj_dengue <- function(sanJuanDengueData,
  n_ahead,
  last_training_point,
  repo_path) {
  require(dplyr)
  require(readr)

  # load dengue data including year and week
  san_juan_dengue <- read_csv(paste0(repo_path, "dengue_sj_experiments/San_Juan_Training_Data.csv"))
  
  last_obs_season <- gsub("/", "-", san_juan_dengue$season[last_training_point])
  last_obs_week <- san_juan_dengue$season_week[last_training_point]
  
  predictions_path <- paste0(repo_path,
    paste0("dengue_sj_experiments/sarima/eval_predictions_box_cox_transformation_seasonal_differencing/last_obs_",
      last_obs_season,
      "_week_", last_obs_week,
      ".csv"))
  
  if(file.exists(predictions_path)) {
    sarima_predictions <- read_csv(predictions_path)
  } else {
    stop(paste0("Predictions not available for last observed time index ",
        last_training_point,
        ", corresponding to week ", last_obs_week,
        " of season ", last_obs_season))
  }
  
  point_pred <- sarima_predictions$Value[
    sarima_predictions$Target == paste0(n_ahead, " wk ahead") &
    sarima_predictions$Type == "Point"]
  
  obs_inc <- san_juan_dengue$total_cases[last_training_point + n_ahead]
  log_prob <- sarima_predictions$Log_value[
    sarima_predictions$Target == paste0(n_ahead, " wk ahead") &
    sarima_predictions$Type == "Bin" &
    sarima_predictions$Bin_start_incl == obs_inc - 0.5]
  
  return(c(obs_inc, log_prob))
}
