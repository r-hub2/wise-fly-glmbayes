## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(glmbayes)

## ----dobson-------------------------------------------------------------------
    ## Dobson (1990) Page 93: Randomized Controlled Trial :
    set.seed(333)
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    print(d.AD <- data.frame(treatment, outcome, counts))

## ----glm_call-----------------------------------------------------------------

    glm.D93 <- glm(counts ~ outcome + treatment,
                   family = poisson(link = "log"))
    summary(glm.D93)

## ----Prior_Setup--------------------------------------------------------------

    ps <- Prior_Setup(counts ~ outcome + treatment,
                      family = poisson())
    mu <- ps$mu
    V  <- ps$Sigma

## ----Call_glmb----------------------------------------------------------------

    glmb.D93 <- glmb(counts ~ outcome + treatment,
                     family = poisson(),
                     pfamily = dNormal(mu = mu, Sigma = V))

## ----summary_glmb-------------------------------------------------------------
    print(glmb.D93)
    summary(glmb.D93)

