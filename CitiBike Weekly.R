library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(tidyverse)
library(posterior)
library(bayesplot)
library(cmdstanr)
library(viridis)

## ------------------------------------------------
## 1. Load data
## ------------------------------------------------

bike_data_Jan_full <- read_csv(
  "https://raw.githubusercontent.com/LukeZingg/DAT494/refs/heads/main/JC-202501-citibike-tripdata.csv"
)
bike_data_Mar_full <- read_csv(
  "https://raw.githubusercontent.com/LukeZingg/DAT494/refs/heads/main/JC-202503-citibike-tripdata.csv"
)
bike_data_May_full <- read_csv(
  "https://raw.githubusercontent.com/LukeZingg/DAT494/refs/heads/main/JC-202505-citibike-tripdata.csv"
)
bike_data_Sep_full <- read_csv(
  "https://raw.githubusercontent.com/LukeZingg/DAT494/refs/heads/main/Citibike%20Data.csv"
)

## For computation: use a *subsample* of September (10,000 rides)
bike_data_Sep <- bike_data_Sep_full

## ------------------------------------------------
## 2. Cleaning function (includes time-of-week t_week)
## ------------------------------------------------

clean_citibike <- function(df) {
  df %>%
    mutate(
      start_time = ymd_hms(started_at),
      
      # Hour of day in decimal (0–24)
      hour = hour(start_time) +
        minute(start_time) / 60 +
        second(start_time) / 3600,
      
      # Day-of-week as factor (for plotting) and numeric (for t_week)
      dow_num     = wday(start_time, week_start = 1),           # 1 = Mon, ..., 7 = Sun
      day_of_week = wday(start_time, label = TRUE, abbr = FALSE, week_start = 1),
      
      week  = isoweek(start_time),
      month = month(start_time, label = TRUE, abbr = TRUE),
      year  = year(start_time),
      
      # Time-of-week in hours: 0–168
      # Mon 00:00 = 0, Tue 00:00 = 24, ..., Sun 24:00 ≈ 168
      t_week = (dow_num - 1) * 24 + hour
    )
}

bike_data_Jan <- clean_citibike(bike_data_Jan_full)
bike_data_Mar <- clean_citibike(bike_data_Mar_full)
bike_data_May <- clean_citibike(bike_data_May_full)
bike_data_Sep <- clean_citibike(bike_data_Sep)
bike_data_Sep_full <- clean_citibike(bike_data_Sep_full)

## ------------------------------------------------
## 3. Data for Wednesday comparisons across months
## ------------------------------------------------

wed_Jan <- filter(bike_data_Jan, day_of_week == "Wednesday")
wed_Mar <- filter(bike_data_Mar, day_of_week == "Wednesday")
wed_May <- filter(bike_data_May, day_of_week == "Wednesday")
wed_Sep <- filter(bike_data_Sep, day_of_week == "Wednesday")

all_weds <- bind_rows(
  wed_Jan %>% filter(month == "Jan"),
  wed_Mar %>% filter(month == "Mar"),
  wed_May %>% filter(month == "May"),
  wed_Sep %>% filter(month == "Sep")
)

## ================================================================
## STORY PLOTS
## ================================================================

## ------------------------------------------------
## 4A. Weekly pattern within September (histograms by day-of-week)
## ------------------------------------------------

ggplot(bike_data_Sep_full, aes(x = hour)) +
  geom_histogram(binwidth = 0.5, fill = "steelblue",
                 color = "white", boundary = 0) +
  facet_wrap(~ day_of_week, ncol = 2) +
  scale_x_continuous(breaks = seq(0, 24, by = 4)) +
  labs(
    title = "Citibike Ride Start Times in September",
    subtitle = "Histograms by day of week",
    x = "Hour of Day (Decimal)",
    y = "Number of Rides"
  ) +
  theme_minimal(base_size = 14)

## ------------------------------------------------
## 4B. Weekly pattern in *time-of-week* t_week (0–168)
## ------------------------------------------------

ggplot(bike_data_Sep_full, aes(x = t_week)) +
  geom_histogram(binwidth = 1, fill = "darkorange",
                 color = "white", boundary = 0) +
  scale_x_continuous(
    breaks = seq(0, 168, by = 24),
    labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon")
  ) +
  labs(
    title = "Time-of-Week Histogram for September",
    x = "Day of Week",
    y = "Number of Rides"
  ) +
  theme_minimal(base_size = 14)

## ------------------------------------------------
## 5. Wednesday densities within September (by ISO week) 
## (Maybe change this to whole week)
## ------------------------------------------------

wednesday_data <- filter(bike_data_Sep, day_of_week == "Wednesday")

ggplot(wednesday_data, aes(x = hour, color = factor(week), group = week)) +
  geom_density(size = 1.2, adjust = 1.3, alpha = 0.8) +
  scale_color_brewer(palette = "Dark2", name = "ISO Week #") +
  scale_x_continuous(breaks = seq(0, 24, by = 4)) +
  labs(
    title = "Ride Start Time Densities for Wednesdays in September",
    subtitle = "Each curve = one Wednesday (subsampled September data)",
    x = "Hour of Day (Decimal)",
    y = "Density"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

## ------------------------------------------------
## 6. Wednesday densities for multiple months
## ------------------------------------------------

ggplot(all_weds, aes(x = hour, group = week, color = factor(week))) +
  geom_density(size = 1, adjust = 1.2, alpha = 0.9) +
  facet_wrap(~ month, ncol = 2, scales = "free_y") +
  scale_color_viridis_d(option = "turbo", guide = "none") +  # many colors, no legend
  scale_x_continuous(breaks = seq(0, 24, 4)) +
  labs(
    title = "Kernel Density of Wednesday Ride Start Times",
    subtitle = "Each line is one Wednesday — variation within and between months",
    x = "Hour of Day (Decimal)",
    y = "Density"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    panel.grid.minor = element_blank()
  )

## ================================================================
## WEEKLY MODEL CONSTRUCTION (t in [0, 168), September only)
## ================================================================

## We model *September* as a stationary weekly Poisson process with
## intensity λ(t), t in [0, 168) hours. We use the subsample here.

# 1. Time-of-week covariate
t_week <- bike_data_Sep$t_week      # in [0,168)
N      <- nrow(bike_data_Sep)
K      <- 20                       # Fourier terms (you can reduce later if needed)

# 2. Effective number of weeks W
#    Use the full September range, NOT the subsample range.
time_span_days <- as.numeric(
  difftime(max(bike_data_Sep_full$started_at),
           min(bike_data_Sep_full$started_at),
           units = "days")
)

W <- time_span_days / 7    # e.g. ≈ 4.3 weeks in the September dataset

# 3. Compile Stan model (weekly)
model_weekly <- cmdstan_model("fourier_poisson_weekly.stan")

stan_data_weekly <- list(
  N = N,
  t = t_week,
  K = K,
  W = W
)

# 4. Sample from the weekly model (using subsample for speed)
fit_weekly <- model_weekly$sample(
  data = stan_data_weekly,
  chains = 2,          # for testing; increase later
  iter_warmup = 500,
  iter_sampling = 1000,
  parallel_chains = 2
)

## ------------------------------------------------
## 0. Extract posterior draws once
## ------------------------------------------------

params <- c("alpha0",
            sprintf("alpha[%d]", 1:K),
            sprintf("beta[%d]", 1:K),
            "tau")

posterior_draws <- fit_weekly$draws(variables = params)
posterior_df    <- as_draws_df(posterior_draws)

## ------------------------------------------------
## 1. Printed summary of model diagnostics
## ------------------------------------------------

summary_df <- fit_weekly$summary(variables = params)

print(
  summary_df[, c("variable", "mean", "sd", "rhat", "ess_bulk", "ess_tail")],
  n = nrow(summary_df)
)

## ------------------------------------------------
## 2. Traceplots for a few key parameters
## ------------------------------------------------

draws_array <- as_draws_array(posterior_draws)

mcmc_trace(
  draws_array,
  pars = c("alpha[1]", "beta[1]", "tau"),
  n_warmup = 0
)

## ------------------------------------------------
## 3. Posterior of alpha0
## ------------------------------------------------

posterior_df %>%
  ggplot(aes(x = alpha0)) +
  geom_density(fill = "skyblue", alpha = 0.6) +
  geom_vline(xintercept = mean(posterior_df$alpha0), linetype = "dashed") +
  labs(
    title = expression("Posterior Distribution of " * alpha[0]),
    x = expression(alpha[0]),
    y = "Density"
  ) +
  theme_minimal(base_size = 14)

## ------------------------------------------------
## 4. Posterior of alpha[1:k] and beta[1:k]
##    (same idea as before, but slightly cleaned up)
## ------------------------------------------------

coef_long <- posterior_df %>%
  select(matches("^alpha\\[|^beta\\[")) %>%
  pivot_longer(
    cols      = everything(),
    names_to  = "param",
    values_to = "value"
  ) %>%
  mutate(
    term_type = if_else(str_detect(param, "^alpha"), "alpha", "beta"),
    index     = as.integer(str_extract(param, "\\d+")),
    label     = if_else(
      term_type == "alpha",
      paste0("alpha[", index, "]"),
      paste0("beta[",  index, "]")
    ),
    label = factor(
      label,
      levels = c(paste0("alpha[", 1:K, "]"), paste0("beta[", 1:K, "]"))
    )
  )

# alpha only
coef_long %>%
  filter(term_type == "alpha") %>%
  ggplot(aes(x = value)) +
  geom_density(fill = "darkred", alpha = 0.4) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  facet_wrap(~ label, scales = "fixed", ncol = 5, labeller = label_parsed) +
  coord_cartesian(xlim = c(-1.25, 1.25)) +
  labs(
    title = "Posterior Distributions of Alpha Coefficients",
    x = "Coefficient Value",
    y = "Density"
  ) +
  theme_minimal(base_size = 12)

# beta only
coef_long %>%
  filter(term_type == "beta") %>%
  ggplot(aes(x = value)) +
  geom_density(fill = "darkblue", alpha = 0.4) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  facet_wrap(~ label, scales = "fixed", ncol = 5, labeller = label_parsed) +
  coord_cartesian(xlim = c(-1, 1)) +
  labs(
    title = "Posterior Distributions of Beta Coefficients",
    x = "Coefficient Value",
    y = "Density"
  ) +
  theme_minimal(base_size = 12)


## ------------------------------------------------
## 5. Posterior of tau
## ------------------------------------------------

posterior_df %>%
  ggplot(aes(x = tau)) +
  geom_density(fill = "purple", alpha = 0.5) +
  labs(
    title = "Posterior Distribution of τ (Shrinkage Parameter)",
    x = expression(tau),
    y = "Density"
  ) +
  theme_minimal(base_size = 14)

## ------------------------------------------------
## Helper: λ(t) given parameters
## ------------------------------------------------

lambda_from_params <- function(t, alpha0, alpha, beta, K) {
  period <- 168   # one week in hours
  log_lambda <- alpha0
  for (k in 1:K) {
    log_lambda <- log_lambda +
      alpha[k] * cos(2 * base::pi * k * t / period) +
      beta[k]  * sin(2 * base::pi * k * t / period)
  }
  exp(log_lambda)
}

# grid for whole week
t_grid <- seq(0, 168, length.out = 1000)

## ------------------------------------------------
## 6. λ(t) for three posterior draws
## ------------------------------------------------

draw_indices_3 <- sample(1:nrow(posterior_df), 3)

lambda_draws <- purrr::map_dfr(1:3, function(i) {
  d     <- posterior_df[draw_indices_3[i], ]
  alpha0 <- d$alpha0
  alpha  <- as.numeric(d[grep("^alpha\\[", names(d))])
  beta   <- as.numeric(d[grep("^beta\\[",  names(d))])
  
  tibble(
    t      = t_grid,
    lambda = lambda_from_params(t_grid, alpha0, alpha, beta, K),
    draw   = paste0("Draw ", i)
  )
})

ggplot(lambda_draws, aes(x = t, y = lambda, color = draw)) +
  geom_line(linewidth = 1) +
  scale_x_continuous(
    breaks = seq(0, 168, by = 24),
    labels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun","Mon")
  ) +
  labs(
    title = expression(lambda(t) ~ "from Three Posterior Draws"),
    x = "Day of Week",
    y = expression(lambda(t) ~ "(rides per hour)")
  ) +
  theme_minimal(base_size = 14)

## ------------------------------------------------
## 7. 95% credible interval for λ(t) (NO histogram)
## ------------------------------------------------
n_draws <- 1000
draw_indices <- sample(1:nrow(posterior_df), n_draws)

lambda_matrix <- matrix(NA_real_, nrow = n_draws, ncol = length(t_grid))

for (i in seq_along(draw_indices)) {
  d     <- posterior_df[draw_indices[i], ]
  alpha0 <- d$alpha0
  alpha  <- as.numeric(d[grep("^alpha\\[", names(d))])
  beta   <- as.numeric(d[grep("^beta\\[",  names(d))])
  lambda_matrix[i, ] <- lambda_from_params(t_grid, alpha0, alpha, beta, K)
}

lambda_summary <- data.frame(
  t     = t_grid,
  mean  = apply(lambda_matrix, 2, mean),
  lower = apply(lambda_matrix, 2, quantile, probs = 0.025),
  upper = apply(lambda_matrix, 2, quantile, probs = 0.975)
)

ggplot(lambda_summary, aes(x = t)) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              fill = "darkred", alpha = 0.5) +
  geom_line(aes(y = mean),
            color = "steelblue", linewidth = 1) +
  scale_x_continuous(
    breaks = seq(0, 168, by = 24),
    labels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun","Mon")
  ) +
  labs(
    title = expression("Posterior Mean and 95% Credible Interval for " * lambda(t)),
    x     = "Day of Week",
    y     = expression(lambda(t) ~ "(rides~per~hour)")
  ) +
  theme_minimal(base_size = 14)

## ------------------------------------------------
## 8. 95% credible interval for λ(t) (With histogram)
## ------------------------------------------------

## compute histogram over 168 hours
hist_data_week <- hist(
  bike_data_Sep$t_week,
  breaks = seq(0, 168, by = 1),
  plot = FALSE
)

# number of rides in subsample
N_sub <- nrow(bike_data_Sep)

# per-week scaling based on subsample
weekly_level <- N_sub / W

hist_df_week <- data.frame(
  mids = hist_data_week$mids,
  # convert histogram to a per-hour rate
  density = (hist_data_week$counts /
               (sum(hist_data_week$counts) * diff(hist_data_week$breaks)[1])) * weekly_level
)


ggplot(lambda_summary, aes(x = t)) +
  # histogram column layer (background)
  geom_col(
    data = hist_df_week,
    aes(x = mids, y = density),
    fill = "black",
    color = "white",
    alpha = 0.5,
    width = 1
  ) +
  
  # 95% CI ribbon
  geom_ribbon(aes(ymin = lower, ymax = upper),
              fill = "lightcoral", alpha = 0.5) +
  
  # posterior mean
  geom_line(aes(y = mean),
            color = "steelblue", linewidth = 1) +
  
  scale_x_continuous(
    breaks = seq(0,168,24),
    labels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun","Mon")
  ) +
  
  labs(
    title = "Weekly Ride Rate: Posterior λ(t) with Observed Histogram",
    x = "Day of Week",
    y = "Estimated rides per hour"
  ) +
  theme_minimal(base_size = 14)


## ------------------------------------------------
## Using the Model - Finding Intervals on Demand at Set Times
## ------------------------------------------------


# Interval integral helper (numerical integrate over posterior sample)
expected_rides_interval <- function(t_start, t_end, posterior_df, K, n_draws = 500){
  
  # draw random posterior rows
  idx <- sample(1:nrow(posterior_df), n_draws)
  
  # numerical integration grid
  t_seq <- seq(t_start, t_end, length.out = 300)
  
  exp_vals <- sapply(idx, function(i){
    d     <- posterior_df[i, ]
    alpha0 <- d$alpha0
    alpha  <- as.numeric(d[grep("^alpha\\[", names(d))])
    beta   <- as.numeric(d[grep("^beta\\[",  names(d))])
    
    lambdas <- lambda_from_params(t_seq, alpha0, alpha, beta, K)
    
    # simple Riemann sum (trapezoid)
    trapz <- sum(lambdas) * (t_seq[2] - t_seq[1])
    return(trapz)
  })
  
  return(exp_vals) # vector of posterior expected rides
}

# 95% credible interval for the expected number of rides 8-10AM Wednesday.

exp_wed_8_10 <- expected_rides_interval(
  t_start = 56,
  t_end   = 58,
  posterior_df,
  K = K
)

quantile(exp_wed_8_10, c(.025,.5,.975))

# 95% credible interval for the expected number of rides 8-10AM Sunday.

exp_sun_8_10 <- expected_rides_interval(
  t_start = 152,
  t_end   = 154,
  posterior_df,
  K = K
)

quantile(exp_sun_8_10, c(.025,.5,.975))


# Now a 95% interval for the number of rides from 4-7 AM on Thursday 
# We expect a low number from our graph

exp_thu_4_7 <- expected_rides_interval(
  t_start = 76,
  t_end   = 79,
  posterior_df,
  K = K
)

# Posterior predictive draws (simulate from Poisson)
ppc_thu_4_7 <- rpois(length(exp_thu_4_7), lambda = exp_thu_4_7)

quantile(ppc_thu_4_7, c(.025,.5,.975))

# As a sanity check for this interval, we will consider our acutal data
# Looking at the 4 weeks we have, we will count the observed number of rides 
# Between 4 and 7 AM 

# Using your cleaned September data (bike_data_Sep)
thursday_4_7 <- bike_data_Sep %>%
  filter(
    day_of_week == "Thursday",
    hour >= 4,
    hour < 7      # include 4–5–6 AM, exclude 7
  ) %>%
  group_by(week) %>%
  summarise(
    n_rides = n(),
    .groups = "drop"
  )

thursday_4_7

# We see that 3/4 of our data falls within our interval range! 



## ------------------------------------------------
## Model Comparison: Assessing LOO for k = 15, 20, 25
## ------------------------------------------------


library(loo)

K_values <- c(15, 20, 25)

fits <- list()
loos <- list()

for (K in K_values) {
  cat("\n============================\n")
  cat("Fitting weekly model with K =", K, "\n")
  cat("============================\n")
  
  stan_data_weekly <- list(
    N = N,
    t = t_week,
    K = K,
    W = W
  )
  
  fit_k <- model_weekly$sample(
    data = stan_data_weekly,
    chains = 2,
    iter_warmup = 500,
    iter_sampling = 1000,
    parallel_chains = 2,
    seed = 42
  )
  
  # store fit
  fits[[paste0("K", K)]] <- fit_k
  
  # extract log_lik and compute PSIS-LOO
  log_lik_mat <- fit_k$draws("log_lik") |> as_draws_matrix()
  loo_k       <- loo(log_lik_mat)
  
  loos[[paste0("K", K)]] <- loo_k
  
  print(loo_k)
}

# After running the loop above...

sig_results <- list()

for (K in K_values) {
  
  fit_k <- fits[[paste0("K", K)]]
  posterior_df <- as_draws_df(fit_k$draws())
  
  # extract alpha[k] and beta[k]
  coef_post <- posterior_df %>%
    select(matches("^alpha\\[|^beta\\[")) 
  
  # pivot long so each param has its own column
  coef_long <- coef_post %>%
    pivot_longer(cols = everything(),
                 names_to = "param",
                 values_to = "value")
  
  # compute CI for each param
  coef_ci <- coef_long %>%
    group_by(param) %>%
    summarise(
      lower = quantile(value, 0.025),
      upper = quantile(value, 0.975),
      .groups = "drop"
    ) %>%
    mutate(
      includes_zero = lower <= 0 & upper >= 0,
      type = if_else(str_detect(param, "^alpha"), "alpha", "beta")
    )
  
  # summarise by type
  summary_k <- coef_ci %>%
    group_by(type) %>%
    summarise(
      total = n(),
      n_excludes_zero = sum(!includes_zero),
      prop_excludes_zero = mean(!includes_zero),
      .groups = "drop"
    ) %>%
    mutate(K = K)
  
  sig_results[[paste0("K", K)]] <- summary_k
  
  cat("\n==== Results for K =", K, "====\n")
  print(summary_k)
}