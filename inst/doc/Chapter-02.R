## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(glmbayes)

## ----Plant_Data---------------------------------------------------------------
## Annette Dobson (1990) "An Introduction to Generalized Linear Models".
## Page 9: Plant Weight Data.
ctl <- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
trt <- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
group <- gl(2, 10, 20, labels = c("Ctl","Trt"))
weight <- c(ctl, trt)

## ----Plant_w_intercept--------------------------------------------------------
lm.D9 <- lm(weight ~ group)

## ----Plant_Prior--------------------------------------------------------------
ps=Prior_Setup(weight ~ group,family=gaussian())

## ----Plant_Prior_Spec---------------------------------------------------------
ps

## ----Plant_lmb_calls----------------------------------------------------------
lmb.D9=lmb(weight ~ group,dNormal(mu=ps$mu,Sigma  =ps$Sigma,dispersion=ps$dispersion))
##lmb.D9_v2=lmb(weight ~ group,dNormal_Gamma(mu=ps$mu,Sigma_0=ps$Sigma_0,shape=ps$shape,rate=ps$rate))
## lmb.D9_v3=lmb(weight ~ group,dIndependent_Normal_Gamma(mu=ps$mu,Sigma = ps$Sigma,shape=ps$shape_ING,rate=ps$rate))
summary(lmb.D9)

## ----Plant_summary------------------------------------------------------------
summary(lm.D9)

## ----Plant_lmb_summary--------------------------------------------------------
summary(lmb.D9)

## ----Plant_lmb_coefficients1--------------------------------------------------
sumlmb<-summary(lmb.D9)
sumlmb$coefficients1

## ----Plant_lmb_coefficients---------------------------------------------------
sumlmb<-summary(lmb.D9)
sumlmb$coefficients

## ----Plant_lmb_Percentiles----------------------------------------------------
sumlmb<-summary(lmb.D9)
sumlmb$Percentiles

