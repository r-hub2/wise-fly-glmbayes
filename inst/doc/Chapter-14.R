## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(glmbayes)
library(coda)

## ----bikesharing-setup--------------------------------------------------------
data("BikeSharing")

# Center continuous predictors
cont_vars <- c("temp", "atemp", "hum", "windspeed", "hr_sin", "hr_cos", "mon_sin", "mon_cos")
BikeSharing_c <- BikeSharing
BikeSharing_c[cont_vars] <- scale(BikeSharing[cont_vars], center = TRUE, scale = FALSE)


# Formula (all variable model)
form <- cnt ~ part_of_day + quarter + holiday + workingday + weathersit +
  hr_sin + hr_cos + mon_sin + mon_cos + temp + atemp + hum + windspeed

# Formula (Limited variable model)
form2 <- cnt ~ part_of_day + quarter + holiday + workingday + weathersit +
  hr_sin + hr_cos + mon_sin + mon_cos

# Train/test split: indices bundled with precomputed Gibbs output (matches demo, set.seed(42))
pct_train <- 0.01
n <- nrow(BikeSharing_c)
ch14_path <- system.file("extdata", "BikeSharing_ch14_gibbs.rds", package = "glmbayes")
stopifnot(nzchar(ch14_path), file.exists(ch14_path))
ch14_saved <- readRDS(ch14_path)
stopifnot(length(ch14_saved$idx_train) == round(pct_train * n))
idx_train <- ch14_saved$idx_train
idx_test  <- setdiff(seq_len(n), idx_train)

Bike_train <- BikeSharing_c[idx_train, ]
Bike_test  <- BikeSharing_c[idx_test, ]

X_train <- model.matrix(form2, data = Bike_train)
X_test  <- model.matrix(form2, data = Bike_test)
y_train <- Bike_train$cnt
y_test  <- Bike_test$cnt
n_train <- length(y_train)
n_test  <- length(y_test)
p       <- ncol(X_train)

# Initial theta and prior for population block
theta <- log(y_train + 0.5)
data_pop <- data.frame(theta = theta, Bike_train)
form_pop <- theta ~ part_of_day + quarter + holiday + workingday + weathersit +
  hr_sin + hr_cos + mon_sin + mon_cos
ps_pop <- Prior_Setup(form_pop, family = gaussian(), data = data_pop)

## ----bikesharing-gibbs, eval = FALSE------------------------------------------
# n_burn <- 200
# n_sim  <- 1000
# 
# beta_out   <- matrix(0, nrow = n_sim, ncol = p)
# sigma_out  <- numeric(n_sim)
# theta_out  <- matrix(0, nrow = n_sim, ncol = n_train)
# 
# set.seed(123)
# 
# # Burn-in
# burn_time <- system.time({
#   for (k in seq_len(n_burn)) {
#     out_pop <- rglmb(1, theta, X_train, family = gaussian(),
#       pfamily = dNormal_Gamma(ps_pop$mu, Sigma_0 = ps_pop$Sigma_0,
#         ps_pop$shape, ps_pop$rate))
#     beta   <- as.vector(out_pop$coefficients[1, ])
#     sigma_theta_sq <- out_pop$dispersion[1]
#     mu_all <- as.vector(X_train %*% beta)
#     for (i in seq_len(n_train)) {
#       theta[i] <- rglmb(1, y_train[i], matrix(1, 1, 1), family = poisson(),
#         pfamily = dNormal(mu = mu_all[i], Sigma = sigma_theta_sq))$coefficients[1, 1]
#     }
#   }
# })
# 
# burn_time
# 
# # Main simulation
# sim_time <- system.time({
#   for (k in seq_len(n_sim)) {
#     out_pop <- rglmb(1, theta, X_train, family = gaussian(),
#       pfamily = dNormal_Gamma(ps_pop$mu, Sigma_0 = ps_pop$Sigma_0,
#         ps_pop$shape, ps_pop$rate))
#     beta   <- as.vector(out_pop$coefficients[1, ])
#     sigma_theta_sq <- out_pop$dispersion[1]
#     mu_all <- as.vector(X_train %*% beta)
#     for (i in seq_len(n_train)) {
#       theta[i] <- rglmb(1, y_train[i], matrix(1, 1, 1), family = poisson(),
#         pfamily = dNormal(mu = mu_all[i], Sigma = sigma_theta_sq))$coefficients[1, 1]
#     }
#     beta_out[k, ]  <- beta
#     sigma_out[k]   <- sqrt(sigma_theta_sq)
#     theta_out[k, ] <- theta
#   }
# })
# 
# 
# sim_time

## ----bikesharing-gibbs-loaded-------------------------------------------------
beta_out  <- ch14_saved$beta_out
sigma_out <- ch14_saved$sigma_out
n_burn    <- ch14_saved$n_burn
n_sim     <- ch14_saved$n_sim
mcmc_main <- ch14_saved$mcmc_main
stopifnot(nrow(beta_out) == n_sim, length(sigma_out) == n_sim, ncol(beta_out) == p)

## ----bikesharing-coda---------------------------------------------------------
summary(mcmc_main)

es <- coda::effectiveSize(mcmc_main)
knitr::kable(
  data.frame(parameter = names(es), effective_size = as.numeric(es)),
  row.names = FALSE,
  digits = 4,
  caption = "Effective sample size (coda::effectiveSize)"
)

ac1 <- coda::autocorr(mcmc_main, lag = 1)
ac1_mat <- drop(ac1)
own_ac1 <- diag(ac1_mat)
names(own_ac1) <- colnames(mcmc_main)
knitr::kable(
  data.frame(parameter = names(own_ac1), lag_1_autocorr = as.numeric(own_ac1)),
  row.names = FALSE,
  digits = 4,
  caption = "Lag-1 autocorrelation (diagonal of coda::autocorr, lag = 1)"
)

## ----bikesharing-pred---------------------------------------------------------
beta_mean <- colMeans(beta_out)
sigma_mean <- mean(sigma_out)

# Option A: conditional mean
y_pred_cond <- exp(X_test %*% beta_mean)
mae_cond  <- mean(abs(y_test - y_pred_cond))
rmse_cond <- sqrt(mean((y_test - y_pred_cond)^2))

# Option B: posterior predictive mean
n_pred <- 500
y_pred_samples <- matrix(0, nrow = n_pred, ncol = n_test)
for (s in seq_len(n_pred)) {
  idx_s <- sample(n_sim, 1)
  beta_s  <- beta_out[idx_s, ]
  sigma_s <- sigma_out[idx_s]
  theta_test <- rnorm(n_test, mean = X_test %*% beta_s, sd = sigma_s)
  y_pred_samples[s, ] <- rpois(n_test, lambda = exp(theta_test))
}
y_pred_mean <- colMeans(y_pred_samples)
mae_pp  <- mean(abs(y_test - y_pred_mean))
rmse_pp <- sqrt(mean((y_test - y_pred_mean)^2))

cat("Option A (conditional): MAE =", round(mae_cond, 2), " RMSE =", round(rmse_cond, 2), "\n")
cat("Option B (post. pred.): MAE =", round(mae_pp, 2), " RMSE =", round(rmse_pp, 2), "\n")

