## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup,echo = FALSE-------------------------------------------------------
library(glmbayes)

## ----directional-tail-example-sketched, eval=FALSE----------------------------
# example("directional_tail", package = "glmbayes")
# 
# # Core objects used in interpretation:
# dt <- directional_tail(fit)
# dt$mahalanobis_shift
# dt$p_directional
# 
# # Same quantities are also surfaced in summary output:
# s <- summary(fit)
# s$dir_tail
# s$dir_tail_null

## ----directional-tail-plot-setup, include=FALSE-------------------------------
set.seed(360)

ctl <- c(4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14)
trt <- c(4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69)
group <- gl(2, 10, 20, labels = c("Ctl", "Trt"))
weight <- c(ctl, trt)
dat2 <- data.frame(weight, group)

ps <- Prior_Setup(
  weight ~ group,
  family = gaussian(),
  pwt = 0.01,
  intercept_source = "null_model",
  effects_source = "null_effects",
  data = dat2
)

fit <- lmb(
  weight ~ group,
  dNormal(ps$mu, ps$Sigma, dispersion = ps$dispersion),
  data = dat2,
  n = 3000
)

## ----directional-tail-plot-whitened, eval=TRUE, fig.width=7, fig.height=5-----
dt <- directional_tail(fit)

Z     <- dt$draws$Z
flag  <- dt$draws$is_tail
delta <- dt$delta
w     <- delta

plot(
  Z,
  col = ifelse(flag, "red", "blue"),
  pch = 19,
  xlab = "Z1",
  ylab = "Z2",
  main = "Directional Tail Diagnostic (Whitened Space)"
)

# Decision boundary orthogonal to direction vector
abline(a = 0, b = -w[1] / w[2], col = "darkgreen", lty = 2)

# Radius corresponding to Mahalanobis shift
r <- sqrt(sum(delta^2))
symbols(
  delta[1], delta[2],
  circles = r,
  inches  = FALSE,
  add     = TRUE,
  lwd     = 2,
  fg      = "gray"
)

points(0, 0, pch = 4, col = "black", lwd = 2)          # reference center in whitened space
points(delta[1], delta[2], pch = 3, col = "purple", lwd = 2) # posterior shift

## ----directional-tail-plot-raw, eval=TRUE, fig.width=7, fig.height=5----------
B       <- dt$draws$B
flag    <- dt$draws$is_tail
mu0     <- as.numeric(fit$Prior$mean)
mu_post <- colMeans(B)

plot(
  B,
  col  = ifelse(flag, "red", "blue"),
  pch  = 19,
  xlab = "Coefficient 1",
  ylab = "Coefficient 2",
  main = "Directional Tail Diagnostic (Raw Coefficient Space)"
)

points(mu0[1], mu0[2], pch = 4, col = "black", cex = 1.5)       # reference center
points(mu_post[1], mu_post[2], pch = 3, col = "darkgreen", cex = 1.5)  # posterior center

legend(
  "topright",
  legend = c("Tail draws", "Non-tail draws", "Reference", "Posterior"),
  col    = c("red", "blue", "black", "darkgreen"),
  pch    = c(19, 19, 4, 3),
  bty    = "n"
)

## ----directional-tail-direct, eval=FALSE--------------------------------------
# fit <- glmb(
#   counts ~ outcome + treatment,
#   family = poisson(),
#   pfamily = dNormal(mu = mu, Sigma = V)
# )
# 
# dt_prior <- directional_tail(fit)      # reference = prior mean
# dt_prior
# print(dt_prior)

## ----directional-tail-null, eval=FALSE----------------------------------------
# mu0 <- rep(0, length(fit$Prior$mean))
# names(mu0) <- names(fit$Prior$mean)
# mu0["(Intercept)"] <- coef(glm(counts ~ 1, family = poisson(), data = fit$data))[1]
# 
# dt_null <- directional_tail(fit, mu0 = mu0)
# dt_null

## ----directional-tail-summary, eval=FALSE-------------------------------------
# s <- summary(fit)
# 
# s$dir_tail        # vs prior
# s$dir_tail_null   # vs null

## ----directional-tail-minimal, eval=FALSE-------------------------------------
# ps <- Prior_Setup(weight ~ group, family = gaussian(), data = dat2)
# fit <- lmb(weight ~ group, dNormal(ps$mu, ps$Sigma, dispersion = ps$dispersion), data = dat2, n = 10000)
# 
# dt <- directional_tail(fit)
# print(dt)

