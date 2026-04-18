## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(glmbayes)

## ----carinsca-----------------------------------------------------------------

    data(carinsca)
    carinsca$Merit <- ordered(carinsca$Merit)
    carinsca$Class <- factor(carinsca$Class)
    options(contrasts = c("contr.treatment", "contr.treatment"))

    Claims  <- carinsca$Claims
    Insured <- carinsca$Insured
    Merit   <- carinsca$Merit
    Class   <- carinsca$Class
    Cost    <- carinsca$Cost

## ----carinsca_classical-------------------------------------------------------


    out <- glm(Cost/Claims ~ Merit + Class,
               family  = Gamma(link = "log"),
               weights = Claims,
               x       = TRUE)

    ## Estimate the dispersion using MLE
    disp <- gamma.dispersion(out)
    
    summary(out,dispersion=disp)


## ----carinsca_disp------------------------------------------------------------

    ## better than the crude estimate from the summary function
    disp <- gamma.dispersion(out)
    disp

## ----carinsca_prior-----------------------------------------------------------
    ps <- Prior_Setup(Cost/Claims ~ Merit + Class,
                      family  = Gamma(link = "log"),
                      weights = Claims)

    mu <- ps$mu
    V  <- ps$Sigma

## ----carinsca_glmb------------------------------------------------------------
    out_glmb <- glmb(Cost/Claims ~ Merit + Class,
                 family  = Gamma(link = "log"),
                 pfamily = dNormal(mu = mu, Sigma = V, dispersion = disp),
                 weights = Claims)

## ----carinsca_sumamry_glmb----------------------------------------------------

    summary(out_glmb)

