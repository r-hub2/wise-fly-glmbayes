## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  warning= FALSE,
  message=FALSE
)

## ----setup--------------------------------------------------------------------
library(glmbayes)

## ----dobson-------------------------------------------------------------------
## Annette Dobson (1990) "An Introduction to Generalized Linear Models".
## Page 9: Plant Weight Data.
ctl <- c(4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14)
trt <- c(4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69)
group  <- gl(2, 10, 20, labels = c("Ctl", "Trt"))
weight <- c(ctl, trt)


## ----Prior_Setup,results = "hide"---------------------------------------------
ps <- Prior_Setup(weight ~ group)
x  <- ps$x
mu <- ps$mu
V  <- ps$Sigma
y <- ps$y
shape    <- ps$shape
rate     <- ps$rate

## ----glm_call_gamma_prior-----------------------------------------------------
out_rlmb <- rlmb(
    n = 1000,
    y = y, x = x,
    pfamily = dGamma(shape = shape, rate = rate,
                     beta = ps$coefficients)
  )



## ----glm_call_gamma_mean------------------------------------------------------
  mean(out_rlmb$dispersion)

## ----Conjugate_prior----------------------------------------------------------
## Conjugate Normal_Gamma Prior

lmb.D9_v2 <- lmb(
  weight ~ group,
  pfamily = dNormal_Gamma(
    ps$mu,
    Sigma_0 = ps$Sigma_0,
    shape = ps$shape,
    rate  = ps$rate
  )
)

summary(lmb.D9_v2)$dispersion


## ----chapter11-load-two-sampler-----------------------------------------------
ch11_path <- system.file(
  "extdata", "Chapter11_Dobson_two_sampler.rds",
  package = "glmbayes"
)
stopifnot(nzchar(ch11_path), file.exists(ch11_path))
ch11_saved <- readRDS(ch11_path)
stopifnot(
  ncol(ch11_saved$gibbs_two_block$beta_out) == ncol(ps$x),
  nrow(ch11_saved$indep_norm_gamma$coefficients) == ch11_saved$indep_norm_gamma$n_draws
)

## ----Independent_normal_gamma_prior, eval = FALSE-----------------------------
# lmb.D9_v3 <- lmb(n = 10000,
#   weight ~ group,
#   dIndependent_Normal_Gamma(
#     ps$mu,
#     ps$Sigma,
#     shape = ps$shape_ING,
#     rate  = ps$rate,
#     max_disp_perc = 0.99,
#     disp_lower = NULL,
#     disp_upper = NULL
#   )
# )
# summary(lmb.D9_v3)$coefficients
# summary(lmb.D9_v3)$dispersion
# sd(lmb.D9_v3$dispersion)

## ----Independent_normal_gamma_loaded------------------------------------------
inc <- ch11_saved$indep_norm_gamma
coef_summ_iid <- data.frame(
  posterior_mean = colMeans(inc$coefficients),
  posterior_SD = apply(inc$coefficients, 2, sd),
  row.names = inc$coef_colnames
)
coef_summ_iid
cat("Dispersion: posterior mean =", mean(inc$dispersion),
    "  SD =", sd(inc$dispersion), "\n")

## ----two_block_Gibbs_sampler, eval = FALSE------------------------------------
# set.seed(180)
# dispersion2 <- ps$dispersion
# 
# ## Burn-in
# for (i in 1:1000) {
#   out1 <- rlmb(
#     n = 1, y = y, x = x,
#     pfamily = dNormal(mu = mu, Sigma = V, dispersion = dispersion2)
#   )
#   out2 <- rlmb(
#     n = 1, y = y, x = x,
#     pfamily = dGamma(shape = shape, rate = rate,
#                      beta = out1$coefficients[1, ])
#   )
#   dispersion2 <- out2$dispersion
# }
# 
# ## Sampling
# beta_out <- matrix(0, nrow = 10000, ncol = 2)
# disp_out <- rep(0, 10000)
# 
# for (i in 1:10000) {
#   out1 <- rlmb(
#     n = 1, y = y, x = x,
#     pfamily = dNormal(mu = mu, Sigma = V, dispersion = dispersion2)
#   )
#   out2 <- rlmb(
#     n = 1, y = y, x = x,
#     pfamily = dGamma(shape = shape, rate = rate,
#                      beta = out1$coefficients[1, ])
#   )
#   beta_out[i, ] <- out1$coefficients[1, 1:2]
#   disp_out[i]   <- out2$dispersion
# }
# 
# colMeans(beta_out)
# sd(beta_out[, 1])
# sd(beta_out[, 2])
# mean(disp_out)
# sd(disp_out)

## ----two_block_Gibbs_loaded---------------------------------------------------
gb <- ch11_saved$gibbs_two_block
beta_out <- gb$beta_out
disp_out <- gb$disp_out
colMeans(beta_out)
sd(beta_out[, 1])
sd(beta_out[, 2])
mean(disp_out)
sd(disp_out)

## ----chapter11-sampler-comparison-table, echo = FALSE-------------------------
inc <- ch11_saved$indep_norm_gamma
gb <- ch11_saved$gibbs_two_block
cn <- inc$coef_colnames
iid_m <- colMeans(inc$coefficients)
iid_s <- apply(inc$coefficients, 2, sd)
gib_m <- colMeans(gb$beta_out)
gib_s <- apply(gb$beta_out, 2, sd)
cmp <- data.frame(
  Parameter = c(cn, "Dispersion"),
  iid_Mean = c(iid_m, mean(inc$dispersion)),
  iid_SD = c(iid_s, sd(inc$dispersion)),
  Gibbs_Mean = c(gib_m, mean(gb$disp_out)),
  Gibbs_SD = c(gib_s, sd(gb$disp_out))
)
knitr::kable(
  cmp,
  digits = 4,
  caption = paste(
    "Dobson plant weight: independent Normal-Gamma (iid) vs",
    "two-block Gibbs"
  )
)

## ----Carinsca-----------------------------------------------------------------
data(carinsca)
carinsca$Merit <- ordered(carinsca$Merit)
carinsca$Class <- factor(carinsca$Class)
options(contrasts=c("contr.treatment","contr.treatment"))
Claims=carinsca$Claims
Insured=carinsca$Insured
Merit=carinsca$Merit
Class=carinsca$Class
Cost=carinsca$Cost
Claims_Adj<-Claims/1000

glm.carinsca <- glm(Cost / Claims ~ Merit + Class,
                    family = Gamma(link = "log"),
                    weights = Claims_Adj, x = TRUE)


## ----Prior_Setup_gamma,results = "hide"---------------------------------------
ps <- Prior_Setup(
  Cost / Claims ~ Merit + Class,
  family = Gamma(link = "log"),
  weights = Claims_Adj
)
mu=ps$mu
V=ps$Sigma
shape    <- ps$shape
rate     <- ps$rate
x  <- ps$x
y  <- ps$y

m<-nrow(x)
p<-ncol(x)

## Starting dispersion for beta | tau in the two-block Gibbs sampler: same quasi-likelihood
## estimate as used elsewhere for this Carinsca Gamma GLM (not the Dobson Gaussian ps).
dispersion2 <- gamma.dispersion(glm.carinsca)


## ----glm_call_gamma_prior2----------------------------------------------------
## Carinsca Gamma GLM (already fitted in the Carinsca chunk for gamma.dispersion etc.)
gamma.dispersion(glm.carinsca)

out2 <- rglmb(
    n = 1000, y = y, x = x,
    family  =Gamma(link="log"),
    pfamily = dGamma(shape = shape, rate = rate,
                     beta = ps$coefficients),
weights = Claims_Adj
  )

mean(out2$dispersion)


## ----chapter11-load-carinsca-gamma--------------------------------------------
ch11_cg_path <- system.file(
  "extdata", "Chapter11_Carinsca_gamma_gibbs.rds",
  package = "glmbayes"
)
stopifnot(nzchar(ch11_cg_path), file.exists(ch11_cg_path))
ch11_cg <- readRDS(ch11_cg_path)
stopifnot(
  ncol(ch11_cg$gibbs_gamma$beta_out) == ncol(ps$x),
  length(ch11_cg$gibbs_gamma$disp_out) == nrow(ch11_cg$gibbs_gamma$beta_out)
)

## ----Block_Gibbs_gamma_Regression, eval = FALSE-------------------------------
# set.seed(190)
# dispersion2 <- gamma.dispersion(glm.carinsca)
# 
# suppressWarnings(
#   suppressMessages(
# for(i in 1:1000){
# 
#   ## --- Block 1: Regression coefficients ---
#   out1 <- rglmb(
#     n = 1, y = y, x = x,
#     family  = Gamma(link="log"),
#     pfamily = dNormal(mu = mu, Sigma = V, dispersion = dispersion2),
#     weights = Claims_Adj
#   )
# 
#   ## --- Block 2: Dispersion (quasi-likelihood sampler) ---
#   out2 <- rglmb(
#     n = 1, y = y, x = x,
#     family  = Gamma(link="log"),
#     pfamily = dGamma(shape = shape, rate = rate,
#                      beta = out1$coefficients[1,]),
#     weights = Claims_Adj
#   )
# 
#   ## --- SCALE dispersion for the next beta update ---
#   ## Convert quasi  (from out2) to MLE for consistency
#   dispersion2 <- out2$dispersion ##* ((m - p)/m)
# 
# }
# 
# )
# )
# 
# ## Run 1000 additional iterations and store output
# beta_out <- matrix(0, nrow = 1000, ncol = ncol(x))
# disp_out <- rep(0, 1000)
# iters_out <- rep(0, 1000)
# 
# 
# suppressWarnings(
#   suppressMessages(
#     for (i in 1:1000) {
#       out1 <- rglmb(
#         n = 1, y = y, x = x,
#         family = Gamma(link="log"),
#         pfamily = dNormal(mu = mu, Sigma = V, dispersion = dispersion2),
#         weights = Claims_Adj
#       )
#       out2 <- rglmb(
#         n = 1, y = y, x = x,
#         family = Gamma(link="log"),
#         pfamily = dGamma(shape = shape, rate = rate,
#                          beta = out1$coefficients[1,]),
#         weights = Claims_Adj
#       )
#       dispersion2 <- out2$dispersion ##* ((m - p) / m)
#       beta_out[i, ] <- out1$coefficients[1, seq_len(ncol(x))]
#       disp_out[i] <- out2$dispersion ##* ((m - p) / m)
#       iters_out[i]<-out2$iters
#     }
#   )
# )
# 
# ## Review output
# 
# 
# beta_mean  <- colMeans(beta_out)
# beta_sd    <- apply(beta_out, 2, sd)
# beta_tlike <- beta_mean / beta_sd   # analogous to GLM t-values
# 
# bayes_coef_table <- data.frame(
#   Estimate = beta_mean,
#   Std.Error = beta_sd,
#   t.like = beta_tlike
# )
# 
# bayes_coef_table
# 
# mean(disp_out)
# mean(iters_out)
# 

## ----Block_Gibbs_gamma_loaded-------------------------------------------------
cg <- ch11_cg$gibbs_gamma
beta_out <- cg$beta_out
disp_out <- cg$disp_out
iters_out <- cg$iters_out

beta_mean  <- colMeans(beta_out)
beta_sd    <- apply(beta_out, 2, sd)
beta_tlike <- beta_mean / beta_sd

bayes_coef_table <- data.frame(
  Estimate = beta_mean,
  Std.Error = beta_sd,
  t.like = beta_tlike
)

bayes_coef_table

mean(disp_out)
mean(iters_out)


