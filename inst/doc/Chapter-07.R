## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(glmbayes)

## ----Menarche-----------------------------------------------------------------
data(menarche,package="MASS")
Age2 <- menarche$Age - 13
Menarche_Model_Data <- data.frame(
  Age = menarche$Age,
  Total = menarche$Total,
  Menarche = menarche$Menarche,
  Age2 = Age2
)
Menarche_Model_Data

## ----Menarche_Prior_Logit-----------------------------------------------------
ps <- Prior_Setup(
  cbind(Menarche, Total - Menarche) ~ Age2,
  family = binomial(link = "logit"),
  data = Menarche_Model_Data
)
mu <- ps$mu
V  <- ps$Sigma

## ----Menarche_Model_Logit-----------------------------------------------------
glmb.logit <- glmb(
  cbind(Menarche, Total - Menarche) ~ Age2,
  family  = binomial(link = "logit"),
  pfamily = dNormal(mu = mu, Sigma = V),
  data    = Menarche_Model_Data,
  n       = 1000
)

## ----Menarche_Summary_Logit---------------------------------------------------
summary(glmb.logit)

## ----Menarche_Prior_Probit----------------------------------------------------
ps2 <- Prior_Setup(
  cbind(Menarche, Total - Menarche) ~ Age2,
  family = binomial(link = "probit"),
  data = Menarche_Model_Data
)
mu2 <- ps2$mu
V2  <- ps2$Sigma

## ----Menarche_Model_Probit----------------------------------------------------
glmb.probit <- glmb(
  cbind(Menarche, Total - Menarche) ~ Age2,
  family  = binomial(link = "probit"),
  pfamily = dNormal(mu = mu2, Sigma = V2),
  data    = Menarche_Model_Data,
  n       = 1000
)

## ----Menarche_Summary_Probit--------------------------------------------------
summary(glmb.probit)

## ----Menarche_Prior_cloglog---------------------------------------------------
ps3 <- Prior_Setup(
  cbind(Menarche, Total - Menarche) ~ Age2,
  family = binomial(link = "cloglog"),
  data = Menarche_Model_Data
)
mu3 <- ps3$mu
V3  <- ps3$Sigma

## ----Menarche_Model_Cloglog---------------------------------------------------
glmb.cloglog <- glmb(
  cbind(Menarche, Total - Menarche) ~ Age2,
  family  = binomial(link = "cloglog"),
  pfamily = dNormal(mu = mu3, Sigma = V3),
  data    = Menarche_Model_Data,
  n       = 1000
)

## ----Menarche_Summary_cloglog-------------------------------------------------
summary(glmb.cloglog)

## ----Menarche_DIC_Compare-----------------------------------------------------
DIC_comp<-rbind(
  extractAIC(glmb.logit),
  extractAIC(glmb.probit),
  extractAIC(glmb.cloglog))

rownames(DIC_comp)<-c("logit","probit","cloglog")
DIC_comp


