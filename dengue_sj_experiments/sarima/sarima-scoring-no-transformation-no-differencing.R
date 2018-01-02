library(dplyr)
library(readr)
library(purrr)

eval_seasons <- c("2005/2006", "2006/2007", "2007/2008", "2008/2009")

## Load data for Dengue fever in San Juan
san_juan_dengue <- read_csv("dengue_sj_experiments/San_Juan_Training_Data.csv")

prediction_files <- Sys.glob("dengue_sj_experiments/sarima/eval_predictions_no_transformation_no_differencing/*.csv")

# drop predictions made in last week of last evaluation season
prediction_files <- prediction_files[
  substr(prediction_files,
    nchar(prediction_files) - 20,
    nchar(prediction_files)) != "2008-2009_week_52.csv"
  ]

prediction_scores <- map_df(
  prediction_files,
  function(file_name) {
    cat(file_name)
    cat("\n")
    predictions <- suppressMessages(read_csv(file_name))
    
    h_ind <- regexpr("-", file_name)
    csv_ind <- regexpr(".csv", file_name)
    last_obs_season <- gsub("-", "/", substr(file_name, h_ind - 4, h_ind + 4))
    last_obs_week <- as.integer(substr(file_name, h_ind + 11, csv_ind - 1))
    
    last_obs_ind <- which(san_juan_dengue$season == last_obs_season &
      san_juan_dengue$season_week == last_obs_week)
    
    prediction_scores_one_time <- expand.grid(
      last_obs_season = last_obs_season,
      last_obs_week = last_obs_week,
      h = seq_len(52)) %>%
      mutate(
        target_season = san_juan_dengue$season[last_obs_ind + seq_len(52)],
        target_week = san_juan_dengue$season_week[last_obs_ind + seq_len(52)],
        target_total_cases = san_juan_dengue$total_cases[last_obs_ind + seq_len(52)],
        predicted_total_cases = predictions %>% filter(Type == "Point") %>% `[[`("Value"),
        predictive_ae = abs(target_total_cases - predicted_total_cases),
        predictive_log_score = sapply(seq_len(52),
          function(ph) {
            if(last_obs_ind + ph > nrow(san_juan_dengue)) {
              return(NA)
            } else {
              predictions$Log_value[
              predictions$Type == "Bin" &
              predictions$Target == paste0(ph, " wk ahead") &
              predictions$Bin_start_incl == san_juan_dengue$total_cases[last_obs_ind + ph] - 0.5]
            }
          })
      ) %>% 
      filter(
        target_season %in% eval_seasons
      )
    
    return(prediction_scores_one_time)
  }
)

write_csv(prediction_scores,
  "dengue_sj_experiments/sarima/sarima-scores-eval-predictions-no-transformation-no-differencing.csv"
)
