## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(glmbayes)

## ----dobson-setup-------------------------------------------------------------
## Dobson (1990) "An Introduction to Generalized Linear Models", Page 9: Plant Weight Data
ctl <- c(4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14)
trt <- c(4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69)
group <- gl(2, 10, 20, labels = c("Ctl", "Trt"))
weight <- c(ctl, trt)

ps <- Prior_Setup(weight ~ group)
x    <- ps$x
y    <- ps$y
mu   <- ps$mu
V    <- ps$Sigma
shape <- ps$shape
rate  <- ps$rate
rate_dg <- if (!is.null(ps$rate_gamma)) ps$rate_gamma else rate
disp_ML <- ps$dispersion

## ----dobson-gibbs, eval = TRUE------------------------------------------------
set.seed(180)
dispersion_current <- disp_ML

## Burn-in
n_burn  <- 1000
for (i in seq_len(n_burn)) {
  out_beta <- rlmb(n = 1, y = y, x = x, pfamily = dNormal(mu = mu, Sigma = V, dispersion = dispersion_current))
  out_phi  <- rlmb(n = 1, y = y, x = x, pfamily = dGamma(shape = shape, rate = rate_dg, beta = out_beta$coefficients[1, ]))
  dispersion_current <- out_phi$dispersion
}

## Sampling
n_sim   <- 1000
beta_out <- matrix(0, nrow = n_sim, ncol = 2)
disp_out <- numeric(n_sim)
for (i in seq_len(n_sim)) {
  out_beta <- rlmb(n = 1, y = y, x = x, pfamily = dNormal(mu = mu, Sigma = V, dispersion = dispersion_current))
  out_phi  <- rlmb(n = 1, y = y, x = x, pfamily = dGamma(shape = shape, rate = rate_dg, beta = out_beta$coefficients[1, ]))
  dispersion_current    <- out_phi$dispersion
  beta_out[i, ]         <- out_beta$coefficients[1, ]
  disp_out[i]           <- out_phi$dispersion
}

## ----dobson-compare-----------------------------------------------------------
lmb_D9 <- lmb(weight ~ group, dIndependent_Normal_Gamma(mu, V, shape = ps$shape_ING, rate = ps$rate))
print(lmb_D9)

## Compare Gibbs vs lmb: means and SDs
coef_names <- colnames(ps$x)
tbl <- data.frame(
  Parameter  = c(coef_names, "dispersion"),
  Gibbs_mean = c(colMeans(beta_out), mean(disp_out)),
  Gibbs_SD   = c(apply(beta_out, 2, sd), sd(disp_out)),
  lmb_mean   = c(colMeans(lmb_D9$coefficients), mean(lmb_D9$dispersion)),
  lmb_SD     = c(apply(lmb_D9$coefficients, 2, sd), sd(lmb_D9$dispersion))
)
knitr::kable(tbl, digits = 4, caption = "Dobson plant weight: Gibbs sampler vs conjugate lmb fit")

## ----schools-data-------------------------------------------------------------
school  <- c("A", "B", "C", "D", "E", "F", "G", "H")
estimate <- c(28.39, 7.94, -2.75, 6.82, -0.64, 0.63, 18.01, 12.16)
sd_obs  <- c(14.9, 10.2, 16.3, 11.0, 9.4, 11.4, 10.4, 17.6)
J       <- length(school)
sigma_y_sq <- sd_obs^2   # known sampling variances for each school

mu_mu     <- mean(estimate)
sigma_mu  <- var(estimate)
n_prior   <- 0.5
disp_ML   <- var(estimate)
shape     <- n_prior / 2
rate      <- disp_ML * shape

## ----chapter13-load-schools-gibbs---------------------------------------------
ch13_path <- system.file(
  "extdata", "Chapter13_Eight_Schools_two_gibbs.rds",
  package = "glmbayes"
)
stopifnot(nzchar(ch13_path), file.exists(ch13_path))
ch13 <- readRDS(ch13_path)
ng <- ch13$normal_gamma
ind <- ch13$indep_norm_gamma
stopifnot(
  ncol(ng$theta_out) == J,
  nrow(ng$theta_out) == ng$n_sim,
  ncol(ind$theta_out) == J,
  nrow(ind$theta_out) == ind$n_sim
)
theta_out_ng <- ng$theta_out
mu_out_ng <- ng$mu_out
sigma_theta_out_ng <- ng$sigma_theta_out
n_burn_ng <- ng$n_burn
n_sim_ng <- ng$n_sim
theta_out <- ind$theta_out
mu_out <- ind$mu_out
sigma_theta_out <- ind$sigma_theta_out
iters_out1 <- ind$iters_out1
iters_out2 <- ind$iters_out2
n_burn_schools <- ind$n_burn
n_sim_schools <- ind$n_sim

## ----schools-ng-gibbs, eval = FALSE-------------------------------------------
# set.seed(101)
# x_one <- as.matrix(rep(1, J), nrow = J, ncol = 1)
# theta_ng <- estimate
# 
# n_burn_ng <- 1000
# n_sim_ng  <- 1000
# theta_out_ng <- matrix(0, nrow = n_sim_ng, ncol = J)
# mu_out_ng    <- numeric(n_sim_ng)
# sigma_theta_out_ng <- numeric(n_sim_ng)
# 
# for (k in seq_len(n_burn_ng)) {
#   out_pop <- rlmb(1, y = theta_ng, x = x_one,
#                   pfamily = dNormal_Gamma(mu_mu, sigma_mu / disp_ML, shape = shape, rate = rate))
#   mu_theta     <- out_pop$coefficients[1, 1]
#   sigma_theta_sq <- out_pop$dispersion
#   for (j in seq_len(J)) {
#     theta_ng[j] <- rlmb(1, y = estimate[j], x = as.matrix(1),
#                         pfamily = dNormal(mu_theta, Sigma = sigma_theta_sq, dispersion = sigma_y_sq[j]))$coefficients[1, 1]
#   }
# }
# 
# for (k in seq_len(n_sim_ng)) {
#   out_pop <- rlmb(1, y = theta_ng, x = x_one,
#                   pfamily = dNormal_Gamma(mu_mu, sigma_mu / disp_ML, shape = shape, rate = rate))
#   mu_theta     <- out_pop$coefficients[1, 1]
#   sigma_theta_sq <- out_pop$dispersion
#   for (j in seq_len(J)) {
#     theta_ng[j] <- rlmb(1, y = estimate[j], x = as.matrix(1),
#                         pfamily = dNormal(mu_theta, Sigma = sigma_theta_sq, dispersion = sigma_y_sq[j]))$coefficients[1, 1]
#   }
#   theta_out_ng[k, ] <- theta_ng
#   mu_out_ng[k]      <- mu_theta
#   sigma_theta_out_ng[k] <- sqrt(sigma_theta_sq)
# }

## ----schools-ng-summary-------------------------------------------------------
colMeans(theta_out_ng)
mean(mu_out_ng)
mean(sigma_theta_out_ng)
sqrt(diag(var(theta_out_ng)))

## ----schools-indep-setup, eval = FALSE----------------------------------------
# theta <- estimate
# n_burn_schools <- 1000
# n_sim_schools  <- 1000
# theta_out <- matrix(0, nrow = n_sim_schools, ncol = J)
# mu_out    <- numeric(n_sim_schools)
# sigma_theta_out <- numeric(n_sim_schools)
# iters_out1 <- numeric(n_burn_schools)
# iters_out2 <- numeric(n_sim_schools)

## ----schools-burnin, eval = FALSE---------------------------------------------
# set.seed(102)
# x_one <- as.matrix(rep(1, J), nrow = J, ncol = 1)
# 
# for (k in seq_len(n_burn_schools)) {
#   out_pop <- rlmb(1, y = theta, x = x_one,
#                   pfamily = dIndependent_Normal_Gamma(mu_mu, sigma_mu, shape = shape, rate = rate))
#   mu_theta     <- out_pop$coefficients[1, 1]
#   sigma_theta_sq <- out_pop$dispersion
#   for (j in seq_len(J)) {
#     theta[j] <- rlmb(1, y = estimate[j], x = as.matrix(1),
#                      pfamily = dNormal(mu_theta, Sigma = sigma_theta_sq, dispersion = sigma_y_sq[j]))$coefficients[1, 1]
#   }
#   iters_out1[k] <- out_pop$iters
# }

## ----schools-iters-burnin, eval = TRUE----------------------------------------
## Mean draws per acceptance (population block) after burn-in
## Lower = better. In demo Ex_07_Schools.R this was ~6.01; compare to check for regression.
mean(iters_out1)

## ----schools-gibbs, eval = FALSE----------------------------------------------
# for (k in seq_len(n_sim_schools)) {
#   out_pop <- rlmb(1, y = theta, x = x_one,
#                   pfamily = dIndependent_Normal_Gamma(mu_mu, sigma_mu, shape = shape, rate = rate))
#   mu_theta     <- out_pop$coefficients[1, 1]
#   sigma_theta_sq <- out_pop$dispersion
#   for (j in seq_len(J)) {
#     theta[j] <- rlmb(1, y = estimate[j], x = as.matrix(1),
#                      pfamily = dNormal(mu_theta, Sigma = sigma_theta_sq, dispersion = sigma_y_sq[j]))$coefficients[1, 1]
#   }
#   theta_out[k, ] <- theta
#   mu_out[k]      <- mu_theta
#   sigma_theta_out[k] <- sqrt(sigma_theta_sq)
#   iters_out2[k]  <- out_pop$iters
# }

## ----schools-iters-main, eval = TRUE------------------------------------------
## Mean draws per acceptance (population block) during main run
mean(iters_out2)

## ----schools-summary, eval = TRUE---------------------------------------------
colMeans(theta_out)
mean(mu_out)
mean(sigma_theta_out)
sqrt(diag(var(theta_out)))

