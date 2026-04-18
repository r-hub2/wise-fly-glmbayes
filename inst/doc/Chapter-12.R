## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----eval=FALSE---------------------------------------------------------------
# # Cleveland GPU-accelerated example (same pattern as example(Cleveland))
# # (Not executed in the vignette build.)
# 
# library(glmbayes)
# 
# # Load the dataset
# data("Cleveland")
# 
# # ------------------------------------------------------------------
# # OpenCL-accelerated Bayesian logistic regression example
# # This example only runs if OpenCL is available.
# # ------------------------------------------------------------------
# 
#   # Prior setup for the full model
#   ps <- Prior_Setup(
#     hd ~ age + sex + cp + trestbps + chol +
#       fbs + restecg + thalach + exang + oldpeak + slope + ca + thal,
#     family = binomial(logit),
#     data = Cleveland
#   )
# 
# t_non_opencl <- system.time({
#   fit_non_opencl <- glmb(
#     hd ~ age + sex + cp + trestbps + chol +
#       fbs + restecg + thalach + exang + oldpeak + slope + ca + thal,
#     family       = binomial(link = "logit"),
#     pfamily      = dNormal(mu = ps$mu, Sigma = ps$Sigma),
#     data         = Cleveland,
#     n            = 20000,
#     Gridtype     = 2,
#     use_parallel = TRUE,
#     use_opencl   = FALSE,
#     verbose      = FALSE
#   )
# })
# 
# t_non_opencl

## ----echo=FALSE, out.width="100%"---------------------------------------------
knitr::include_graphics(
  system.file("extdata", "cleveland_non_opencl_output_01.png", package = "glmbayes")
)

## ----eval=FALSE---------------------------------------------------------------
# t_opencl <- system.time({
#   fit_opencl <- glmb(
#     hd ~ age + sex + cp + trestbps + chol +
#       fbs + restecg + thalach + exang + oldpeak + slope + ca + thal,
#     family       = binomial(link = "logit"),
#     pfamily      = dNormal(mu = ps$mu, Sigma = ps$Sigma),
#     data         = Cleveland,
#     n            = 20000,
#     Gridtype     = 2,
#     use_parallel = TRUE,
#     use_opencl   = TRUE,
#     verbose      = FALSE
#   )
# })
# 
# t_opencl
# 

## ----echo=FALSE, out.width="100%"---------------------------------------------
knitr::include_graphics(
  system.file("extdata", "cleveland_opencl_output_01.png", package = "glmbayes")
)

## ----eval=FALSE---------------------------------------------------------------
# 
# summary(fit_opencl)

## ----echo=FALSE, out.width="100%"---------------------------------------------
knitr::include_graphics(
  system.file("extdata", "cleveland_summary_output_01.png", package = "glmbayes")
)
knitr::include_graphics(
  system.file("extdata", "cleveland_summary_output_02.png", package = "glmbayes")
)

