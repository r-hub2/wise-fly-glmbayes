## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  df_print = "default"
)


## ----setup,echo = FALSE-------------------------------------------------------
library(glmbayes)
## Annette Dobson (1990) "An Introduction to Generalized Linear Models".
## Page 9: Plant Weight Data.
ctl    <- c(4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14)
trt    <- c(4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69)
group  <- gl(2, 10, 20, labels = c("Ctl", "Trt"))
weight <- c(ctl, trt)

dat <- data.frame(weight, group)
set.seed(333)

## ----Plant_Prior--------------------------------------------------------------
ps = Prior_Setup(
  weight ~ group, data = dat, family = gaussian(), pwt = 0.001,
  n_prior = NULL, sd = NULL,
  intercept_source = c("null_model", "full_model"),
  effects_source = c("null_effects", "full_model"),
  mu = NULL
)

#mu       = ps$mu
#V        = ps$Sigma
#disp_ML  = ps$dispersion

## ----Plant_w_intercept--------------------------------------------------------
lm.D9 <- lm(weight ~ group)
sumlm<-summary(lm.D9)
sumlm$coefficients

## ----Plant_lmb_call1----------------------------------------------------------
lmb.D9=lmb(weight ~ group,dNormal(mu=ps$mu,Sigma=ps$Sigma,dispersion=ps$dispersion),n=1000)
sumlmb<-summary(lmb.D9)
sumlmb$coefficients

## ----Plant_lmb_call2----------------------------------------------------------
lmb.D9=lmb(weight ~ group,dNormal(mu=ps$mu,Sigma=ps$Sigma,dispersion=ps$dispersion),n=100000)
sumlmb<-summary(lmb.D9)
sumlmb$coefficients

## ----MTCars Data--------------------------------------------------------------
data(mtcars)

# Create gallons per mile
mtcars$gpm <- 1 / mtcars$mpg

## ----MeanCenter---------------------------------------------------------------
# Mean-center predictors
#mtcars$c_hp <- scale(mtcars$hp, center = TRUE, scale = FALSE)
#mtcars$c_drat <- scale(mtcars$drat, center = TRUE, scale = FALSE)
#mtcars$c_qsec <- scale(mtcars$qsec, center = TRUE, scale = FALSE)

mtcars$c_wt  <- as.numeric(scale(mtcars$wt, center = TRUE, scale = FALSE))
mtcars$c_cyl <- as.numeric(scale(mtcars$cyl, center = TRUE, scale = FALSE))

## ----weight_v_gpm-------------------------------------------------------------

# Fit linear model
fit <- lm(gpm ~ c_wt, data = mtcars)

# Create scatter plot
plot(mtcars$c_wt, mtcars$gpm,
     pch = 19, col = "steelblue",
     xlab = "Mean-Centered Weight (c_wt)",
     ylab = "Gallons per Mile (gpm)",
     main = "Fuel Consumption vs. Mean-Centered Weight")

# Add regression line
abline(fit, col = "darkred", lwd = 2)

# Optional: Add confidence band manually
x_vals <- seq(min(mtcars$c_wt), max(mtcars$c_wt), length.out = 100)
pred <- predict(fit, newdata = data.frame(c_wt = x_vals), interval = "confidence")

lines(x_vals, pred[, "lwr"], col = "darkred", lty = 2)
lines(x_vals, pred[, "upr"], col = "darkred", lty = 2)


## ----cyl_v_gpm----------------------------------------------------------------
# Ensure gpm is defined
mtcars$gpm <- 1 / mtcars$mpg

# Create boxplot
boxplot(gpm ~ factor(cyl), data = mtcars,
        col = "lightblue", border = "darkblue",
        xlab = "Number of Cylinders",
        ylab = "Gallons per Mile (gpm)",
        main = "Fuel Consumption by Number of Cylinders")

# Add jittered points
set.seed(123)  # for reproducibility
points(jitter(as.numeric(factor(mtcars$cyl)), amount = 0.2), mtcars$gpm,
       pch = 19, col = rgb(70/255, 130/255, 180/255, alpha = 0.6))


## ----Mt_cars_Default_Prior----------------------------------------------------
# Use Prior_Setup to illustrate prior strength
pscars  <- Prior_Setup(formula = gpm ~  c_wt+c_cyl,family=gaussian(),data = mtcars,pwt=0.01) ## pwt=0.01-->n_prior= 32/99
pscars2 <- Prior_Setup(formula = gpm ~  c_wt+c_cyl,family=gaussian(),data = mtcars,pwt=0.5)  ## pwt=0.5-->n_prior= 32
pscars3 <- Prior_Setup(formula = gpm ~  c_wt+c_cyl,family=gaussian(),data = mtcars,pwt=0.55) ## pwt=0.55-->n_prior= 39.111

## ----print_priors-------------------------------------------------------------
pscars

## ----print_priors2------------------------------------------------------------
pscars2

## ----print_priors3------------------------------------------------------------
pscars3

## ----run_lmb------------------------------------------------------------------

set.seed(333)

lmb_cars<- lmb(formula = gpm ~  c_wt+c_cyl,pfamily=dNormal(mu=pscars$mu,Sigma=pscars$Sigma, dispersion = pscars$dispersion),data =mtcars)
lmb_cars2<-lmb(formula = gpm ~  c_wt+c_cyl,pfamily=dNormal(mu=pscars2$mu,Sigma=pscars2$Sigma, dispersion = pscars2$dispersion),data =mtcars)
lmb_cars3<-lmb(formula = gpm ~  c_wt+c_cyl,pfamily=dNormal(mu=pscars3$mu,Sigma=pscars3$Sigma, dispersion = pscars3$dispersion),data =mtcars)


sum_lmbcars<-summary(lmb_cars)  
sum_lmbcars2<-summary(lmb_cars2)
sum_lmbcars3<-summary(lmb_cars3)


## ----sumlmb_out---------------------------------------------------------------
sum_lmbcars$coefficients

## ----sumlmb_out2--------------------------------------------------------------
sum_lmbcars2$coefficients

## ----sumlmb_out3--------------------------------------------------------------
sum_lmbcars3$coefficients

## ----Mt_cars_Default_Prior2---------------------------------------------------
# Use Prior_Setup to illustrate prior strength
pscars4 <- Prior_Setup(formula = gpm ~  c_wt+c_cyl,family=gaussian(),data = mtcars,n_prior=1)  ## n_prior=1--> pwt=0.0303
pscars5 <- Prior_Setup(formula = gpm ~  c_wt+c_cyl,family=gaussian(),data = mtcars,n_prior=10)  ## n_prior=10--> pwt=0.2381
pscars6 <- Prior_Setup(formula = gpm ~  c_wt+c_cyl,family=gaussian(),data = mtcars,n_prior=32)  ## n_prior=32--> pwt=0.5

set.seed(333)

lmb_cars4<-lmb(formula = gpm ~  c_wt+c_cyl,pfamily=dNormal(mu=pscars4$mu,Sigma=pscars4$Sigma, dispersion = pscars4$dispersion),data =mtcars)
lmb_cars5<-lmb(formula = gpm ~  c_wt+c_cyl,pfamily=dNormal(mu=pscars5$mu,Sigma=pscars5$Sigma, dispersion = pscars5$dispersion),data =mtcars)
lmb_cars6<-lmb(formula = gpm ~  c_wt+c_cyl,pfamily=dNormal(mu=pscars6$mu,Sigma=pscars6$Sigma, dispersion = pscars6$dispersion),data =mtcars)

sum_lmbcars4<-summary(lmb_cars4)
sum_lmbcars5<-summary(lmb_cars5)
sum_lmbcars6<-summary(lmb_cars6)


## ----sumlmb_out4--------------------------------------------------------------
sum_lmbcars4$coefficients

## ----sumlmb_out5--------------------------------------------------------------
sum_lmbcars5$coefficients

## ----sumlmb_out6--------------------------------------------------------------
sum_lmbcars5$coefficients

## ----Mt_cars_sd_Prior1--------------------------------------------------------
pscars7 <- Prior_Setup(formula = gpm ~  c_wt+c_cyl,family=gaussian(),data =mtcars,sd=c(0.005,0.010,0.0015),n_prior=(0.01/0.99)*32)  ## 
pscars7

## ----sumlmb_out7--------------------------------------------------------------

set.seed(333)
lmb_cars7<-lmb(formula = gpm ~  c_wt+c_cyl,pfamily=dNormal(mu=pscars7$mu,Sigma=pscars7$Sigma, dispersion = pscars7$dispersion),data =mtcars)
sum_lmbcars7<-summary(lmb_cars7)
sum_lmbcars7

## ----Mt_cars_prior_full_model-------------------------------------------------
pscars8 <- Prior_Setup(formula = gpm ~  c_wt+c_cyl,family=gaussian(),data =mtcars,intercept_source = "full_model",effects_source = "full_model")  ## 
pscars8

## ----sumlmb_out8--------------------------------------------------------------

set.seed(333)
lmb_cars8<-lmb(formula = gpm ~  c_wt+c_cyl,pfamily=dNormal(mu=pscars8$mu,Sigma=pscars8$Sigma, dispersion = pscars8$dispersion),data =mtcars)
sum_lmbcars8<-summary(lmb_cars8)
sum_lmbcars8$coefficients1
sum_lmbcars8$coefficients

## ----Mt_cars_small------------------------------------------------------------
# Use Prior_Setup to illustrate prior strength
lmcars_small <- lm(formula = gpm ~  c_wt,  data = mtcars)
lmcars_small

## ----Mt_cars_Default_Prior5---------------------------------------------------
# Use Prior_Setup to illustrate prior strength
pscars9 <- Prior_Setup(  formula = gpm ~  c_wt+c_cyl,family=gaussian(),  data = mtcars,  pwt=0.01,mu=c(0.05423,0.01494,0))
pscars9

## ----sumlmb_out9--------------------------------------------------------------
set.seed(333)
lmb_cars9<-lmb(formula = gpm ~  c_wt+c_cyl,pfamily=dNormal(mu=pscars9$mu,Sigma=pscars9$Sigma, dispersion = pscars9$dispersion),data =mtcars)
sum_lmbcars9<-summary(lmb_cars9)
sum_lmbcars9$coefficients1
sum_lmbcars9$coefficients

## ----Mt_cars_Default_Prior4---------------------------------------------------
# Use Prior_Setup to illustrate prior strength
ps4 <- Prior_Setup(
  formula = gpm ~  c_wt+c_cyl,
  family=gaussian(),
  data = mtcars,
  n_prior=1
##  ,pwt = c(0.1, 0.5),     # Example vector weights
##  priorsd = c(5, 2),     # Custom SDs for each predictor
##  priorN = NULL          # Leave NULL to focus on pwt and priorsd
)
ps4

## ----Custom_prior-------------------------------------------------------------

ps_custom <- Prior_Setup(
  gpm ~ c_wt + c_cyl,
  data = mtcars,
  pwt  = 0.01,
  mu   = c(0.05423, 0.01494, 0)   # intercept, c_wt, c_cyl
)

ps_custom$mu

