# Carga de datos y librerias
rm(list=ls())
RNGkind(kind="Mersenne-Twister", normal.kind = "Inversion", sample.kind="Rejection")
set.seed(2054)
library(xgboost);library(mvtnorm);library(matrixcalc);library(plyr)
library(dplyr);library(tibble);library(purrr);library(furrr)
library(purrrlyr);library(e1071);library(MASS);library(glmnet)
library(randomForest);library(caret);library(tidymodels)
library(dials);library(tune);library(finetune)
library(ParBayesianOptimization)
#Carga de los datos
Datos <- read.table("processed.cleveland.data", sep = ",",
header = FALSE, na.strings = "?")
colnames(Datos) <- c("EDAD", "SEX", "TDT","PAR", "COL", "NAS",
"RELEC", "FCM", "AI","DST", "PST", "NVP","TAL", "Y")
# Preprocesamiento #
colSums(is.na(Datos))
Datos <- na.omit(Datos)
Datos <- Datos %> %
mutate(Y = ifelse(Y == 0, 0, 1))
Datos$SEX <- as.factor(Datos$SEX)
Datos$TDT <- as.factor(Datos$TDT)
Datos$NAS <- as.factor(Datos$NAS)
Datos$RELEC <- as.factor(Datos$RELEC)
Datos$AI <- as.factor(Datos$AI)
Datos$PST <- as.factor(Datos$PST)
Datos$NVP <- as.factor(Datos$NVP)
Datos$TAL <- as.factor(Datos$TAL)
Datos$Y <- as.factor(Datos$Y)
## Análisis Descriptivo de los datos reales para predecir una enfermedad de corazón
Datos0 <- Datos %> % filter(Y == 0)
Datos1 <- Datos %> % filter(Y == 1)
for (i in names(Datos0)) {
if (class(Datos0[[i]]) == "factor") {
a = table(Datos0[[i]])
} else {
a = summary(Datos0[[i]])
}
print(i)
print(a)
}
for (i in names(Datos1)) {
if (class(Datos1[[i]]) == "factor") {
a = table(Datos1[[i]])
} else {
a = summary(Datos1[[i]])
}
print(i)
print(a)
}
Datos_numericos <- Datos %> %
select_if(is.numeric) %> %
mutate(Y = Datos$Y)
BP_Dat<- Datos_numericos %> %
gather(key = "variable", value = "value", -Y)
### Diagramas de caja por variable numérica
ggplot(BP_Dat, aes(x = Y, y = value, fill = Y)) +
geom_boxplot() +
facet_wrap(~ variable, scales = "free", ncol = 3) +
theme_minimal() +
labs(title = "Boxplots de las variables numéricas por grupo de la variable y")+
theme(legend.position = "top")
Datos_num0 <- Datos0 %> %
dplyr::select(-SEX, -NAS, -AI,-PST,-TAL,-NVP,-Y) %> %
dplyr::mutate_all(~as.numeric(.)) %> %
as.data.frame()
Datos_num1 <- Datos1 %> %
dplyr::select(-SEX, -NAS, -AI,-PST,-TAL,-NVP, -Y) %> %
dplyr::mutate_all(~as.numeric(.)) %> %
as.data.frame()
Datos_num <- Datos %> %
dplyr::select(-SEX, -NAS, -AI,-PST,-TAL,-NVP, Y) %> %
dplyr::mutate_all(~as.numeric(.)) %> %
scale() %> %
as.data.frame() %> %
dplyr::mutate(Y = Datos$Y)
colores <- colorRampPalette(c("red", "white", "blue"))(200)
M_cor0 <- round(cor(Datos_num0),4)
M_cor1 <- round(cor(Datos_num1),4)
### Diagrama de correlación de variables en numericas y cat. nominales(NEC)
corrplot(M_cor0, method = "circle", col = colores)
### Diagrama de correlación de variables en numericas y cat. nominales(EC)
corrplot(M_cor1, method = "circle", col = colores)
### Análisis de grupos via componentes principales
pca_result <- PCA(Datos_num[-8], scale.unit = TRUE)
pca_dat <- data.frame(pca_result$ind$coord,CLASE = as.factor(Datos$CLASE))
correlaciones <- pca_result$var$coord
### Graficos Dim1/Dim2;Dim1/Dim3;Dim1/Dim4
P1 <-ggplot(pca_dat, aes(Dim.1, Dim.2, color = CLASE)) + geom_point(size=2.5)
P2 <-ggplot(pca_dat, aes(Dim.1, Dim.3, color = CLASE)) + geom_point(size=2.5)
P3 <-ggplot(pca_dat, aes(Dim.1, Dim.4, color = CLASE)) + geom_point(size=2.5)
grid.arrange(P1, P2, P3, ncol = 3)
DatosT <- Datos
# Cálculo del poder predictivo datos reales #
## Funciones Auxiliares --------------------------------------------
### Función para generar B conjuntos de Train y Test
ConjuntoEvaluacion <- function(B, por, DataP){
ParTrainTest=createDataPartition(DataP$Y, p = por,
list = FALSE,
times = B)
Train <- list(NA)
Test <- list(NA)
Id <- list(NA)
for(i in 1:B){
Train[[i]] <- DataP[ParTrainTest[,i],]
Test[[i]] <- DataP[-ParTrainTest[,i],]
Id[[i]]<- i
}
return(list(Train = Train,
Test = Test, Id=Id))
}
### Función para calcular los errores de clasificación por Y
ErroresClasificacion <- function(x, y){
Y0Train <- x[1,2]/sum(x[1,])
Y1Train <- x[2,1]/sum(x[2,])
GlobalTrain <- 1-sum(diag(x))/sum(x)
Y0Test <- y[1,2]/sum(y[1,])
Y1Test <- y[2,1]/sum(y[2,])
GlobalTest <- 1-sum(diag(y))/sum(y)
return(data.frame(Y0Train,Y1Train,GlobalTrain,Y0Test,Y1Test,GlobalTest))
}
### Función para generar resultados.
GenerarResultados <- function(DatosX,Method, workers) {
plan(strategy = multisession,
workers = workers)
Aux1 <- DatosX %> %
future_pmap(.f = get(Method),
.progress = TRUE) %> %
transpose() %> %
pmap(.f = ~ErroresClasificacion(.x,.y)) %> %
ldply(data.frame)
Aux2 <- Aux1 %> %
summarise(Y0Train = mean(Y0Train),
Y1Train = mean(Y1Train),
GlobalTrain = mean(GlobalTrain),
Y0Test = mean(Y0Test),
Y1Test = mean(Y1Test),
GlobalTest = mean(GlobalTest)) %> %
mutate_if(is.numeric, ~.*100) %> %
mutate_if(is.numeric, round, 3) %> %
add_column( Method = Method,
.before = "Y0Train")
return(list(Individual = Aux1,
Global = Aux2))
}
## Métodos ---------------------------------------------------------
### Regresión Logística
logistic <- function(Train, Test, Id) {
logit <- glm(formula = Y~., family = binomial(link=logit), data = Train)
PredTrain <- factor(ifelse(predict(object = logit,
newdata = Train,
type = "response") > .5,
1, 0),levels=levels(Train$Y))
PredTest <- factor(ifelse(predict(object = logit,
newdata = Test,
type = "response") > .5,
1, 0),levels=levels(Train$Y))
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### Regresión L. Lasso
logistic_lasso <- function(Train, Test, Id) {
XTrain <- model.matrix(Y ~ ., data = Train)[,-1]
YTrain <- Train$Y
XTest <- model.matrix(Y ~ ., data = Test)[,-1]
set.seed(Id)
lasso.tun <- cv.glmnet(x = XTrain,
y = YTrain,
nfolds = 10,
type.measure = "class",
gamma = 0,
relax = FALSE,
family = "binomial",
nlambda = 100)
PredTrain <- factor(ifelse(predict(object = lasso.tun,
newx = XTrain,
type = "response",
s = "lambda.min") > .5,
1, 0),levels=levels(Train$Y))
PredTest <- factor(ifelse(predict(object = lasso.tun,
newx = XTest,
type = "response",
s = "lambda.min") > .5,
1, 0),levels=levels(Train$Y))
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### Regresión L. Lasso (2)
logistic2_lasso <- function(Train, Test, Id) {
XTrain <- model.matrix(Y ~ .ˆ2, data = Train)[,-1]
YTrain <- Train$Y
XTest <- model.matrix(Y ~ .ˆ2, data = Test)[,-1]
set.seed(Id)
lasso.tun <- cv.glmnet(x = XTrain,
y = YTrain,
nfolds = 10,
type.measure = "class",
gamma = 0,
relax = FALSE,
family = "binomial",
nlambda = 100)
PredTrain <- factor(ifelse(predict(object = lasso.tun,
newx = XTrain,
type = "response",
s = "lambda.min") > .5,
1, 0),levels=levels(Train$Y))
PredTest <- factor(ifelse(predict(object = lasso.tun,
newx = XTest,
type = "response",
s = "lambda.min") > .5,
1, 0),levels=levels(Train$Y))
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### Bosque aleatorio
RandomForestTuneOOB <- function(Train, Test, Id) {
mallamtry <- seq(1,10,1)
mallantree <- c(100, 500, 1000)
mallanodesize <- c(1,10,15)
oobRFi = function(i, mallaj, k) {
j = max(mallaj)
forest = randomForest(Y ~ .,
mtry = i,
ntree = j,
nodesize = k,
data = Train)
return(forest$err.rate[mallaj]) #da los errores oob al considerar de 1 a ntree
}
tunoob = function(mallamtry, mallantree, mallanodesize){
nmtry=length(mallamtry)
nntree=length(mallantree)
nnodesize=length(mallanodesize)
iterk=0
tunRF1=matrix(NA, 4, nrow=nmtry*nntree*nnodesize)
for(jmt in 1:nmtry){
for(jns in 1:nnodesize){
iterk=iterk+1
rowinf=(nntree)*(iterk-1)+1
rowsup=(nntree)*(iterk-1)+nntree
tunRF1[rowinf:rowsup,4]=oobRFi(mallamtry[jmt], mallantree, mallanodesize[jns])
tunRF1[rowinf:rowsup,1]=mallamtry[jmt]
tunRF1[rowinf:rowsup,2]=mallantree
tunRF1[rowinf:rowsup,3]=mallanodesize[jns]
}
}
return(tunRF1[which.min(tunRF1[,4]),])
}
set.seed(Id)
OOB <- tunoob(mallamtry, mallantree, mallanodesize)
RF <- randomForest(formula = Y ~ .,
data = Train,
mtry = OOB[1],
ntree = OOB[2],
nodesize = OOB[3])

PredTrain <- predict(object = RF,
newdata = Train,
type = "class")
PredTest <- predict(object = RF,
newdata = Test,
type = "class")
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### XGBoost con tuneo de relleno espacial
XGBtunetinyBSF <- function(Train, Test, Id ) {
vb_train <- Train
vb_test <- Test
xgb_spec <- boost_tree(
trees = tune(),
tree_depth = tune(), min_n = tune(),
loss_reduction = tune(),
mtry = tune(),
learn_rate = tune()
) %> %
set_engine("xgboost",nthread = 4,
scale_pos_weight = tune(), penalty_L1 =tune()) %> %
set_mode("classification")
param_set <- parameters(
trees(range=c(50,1000)),
tree_depth(range = c(4, 8)),
min_n(range = c(2, 20)),
loss_reduction(range = c(0, 0.2)),
mtry(range = c(2,11)),
learn_rate(range = c(0.01, 0.3)),
scale_pos_weight(range = c(.5,3)),
penalty_L1(range=c(0,1),trans = NULL)
)
xgb_grid <- grid_space_filling(param_set, size = 350)
bb_rec <- recipe(Y ~ ., data = Train) %> %
step_normalize(all_numeric_predictors()) %> %
step_dummy(all_nominal_predictors())
xgb_wf <- workflow() %> %
add_recipe(bb_rec) %> %
add_model(xgb_spec)
vb_folds <- vfold_cv(vb_train, v=6 ,strata = Y)
set.seed(Id)
xgb_res <-tune_grid(
xgb_wf,
resamples = vb_folds,
grid = xgb_grid,
control = control_grid(save_pred = TRUE)
)
best_accuracy <- select_best(xgb_res, metric = "accuracy")
best_accuracy
XGB <- finalize_workflow(
xgb_wf,
best_accuracy
)
XGB_fit <- fit(XGB, data = vb_train )
PredTrain <- predict(object = XGB_fit,
new_data = vb_train,
type = "class")$.pred_class
PredTest <- predict(object = XGB_fit,
new_data = vb_test,
type = "class")$.pred_class
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### XGBoost con busqueda en malla aleatoria
XGBtunetinyBA <- function(Train, Test, Id ) {
vb_train <- Train
vb_test <- Test
xgb_spec <- boost_tree(
trees = tune(),
tree_depth = tune(), min_n = tune(),
loss_reduction = tune(),
mtry = tune(),
learn_rate = tune()
) %> %
set_engine("xgboost",nthread = 4,
scale_pos_weight = tune(), penalty_L1 =tune()) %> %
set_mode("classification")
param_set <- parameters(trees(range=c(50,1000)),
tree_depth(range = c(4, 8)),min_n(range = c(2, 20)),
loss_reduction(range = c(0, 0.2)),mtry(range = c(2,11)),
learn_rate(range = c(0.01, 0.3)),scale_pos_weight(range = c(.5,3)),
penalty_L1(range=c(0,1),trans = NULL))
set.seed(Id)
xgb_grid <- grid_random(param_set, size = 350)
xgb_wf <- workflow() %> %
add_formula(Y ~ .) %> %
add_model(xgb_spec)
vb_folds <- vfold_cv(vb_train, v=6 ,strata = Y)
set.seed(Id)
xgb_res <-tune_grid(
xgb_wf,
resamples = vb_folds,
grid = xgb_grid,
control = control_grid(save_pred = TRUE)
)
best_accuracy <- select_best(xgb_res, metric = "accuracy")
XGB <- finalize_workflow(
xgb_wf,
best_accuracy
)
XGB_fit <- fit(XGB, data = vb_train )
PredTrain <- predict(object = XGB_fit,
new_data = vb_train,
type = "class")$.pred_class
PredTest <- predict(object = XGB_fit,
new_data = vb_test,
type = "class")$.pred_class
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### XGBoost con busqueda por optimización bayesiana
XGBtuneCV <- function(Train, Test, Id) {
DM_TR <- model.matrix(Y~.-1,data=Train)
DM_TE <- model.matrix(Y~.-1,data=Test)
TR_LB <- as.factor(Train$Y)
obj_func <- function(eta, max_depth, min_child_weight, subsample, lambda, alpha) {
param <- list(
eta = eta,
max_depth = max_depth,min_child_weight = min_child_weight,
subsample = subsample,lambda = lambda,alpha = alpha,
booster = "gbtree",objective = "binary:logistic",
eval_metric = "logloss")
xgbcv <- xgb.cv(params = param,
data = DM_TR,label = as.numeric(TR_LB)-1,
nround = 50,nfold = 2,prediction = TRUE,
early_stopping_rounds = 5,verbose = 0,maximize = F)
lst <- list(
Score = -min(xgbcv$evaluation_log$test_logloss_mean),
nrounds = xgbcv$best_iteration
)
return(lst)
}
bounds <- list(eta = c(0.001, 0.2),max_depth = c(1L, 10L),
min_child_weight = c(1, 50),subsample = c(0.1, 1),
lambda = c(1, 10),alpha = c(1, 10))
set.seed(Id)
bayes_out <- bayesOpt(FUN = obj_func, bounds = bounds, initPoints = 30,
iters.n = 4)
data.frame(getBestPars(bayes_out))
opt_params <- append(list(booster = "gbtree",
objective = "binary:logistic",
eval_metric = "logloss"),
getBestPars(bayes_out))
xgbcv <- xgb.cv(params = opt_params,
data = DM_TR,label = as.numeric(TR_LB)-1,
nround = 100,nfold = 6,prediction = TRUE,
early_stopping_rounds = 5,verbose = 0, maximize = F)
nrounds = xgbcv$best_iteration
mdl <- xgboost(data = DM_TR, label = as.numeric(TR_LB)-1,
params = opt_params, maximize = F,
early_stopping_rounds = 5, nrounds = nrounds,
verbose = 0)
PredTrain_prob <- predict(mdl, DM_TR)
PredTrain <- ifelse(PredTrain_prob > 0.5, 1, 0)
PredTest_prob <- predict(mdl, DM_TE)
PredTest <- ifelse(PredTest_prob> 0.5, 1, 0)
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### XGBoost con optimización por métodos de carrera
XGBtuneRM <- function(Train, Test, Id) {
xgb_spec <- boost_tree(trees = tune(),tree_depth = tune(),
min_n = tune(),loss_reduction = tune(),
mtry = tune(),learn_rate = tune()
) %> %
set_engine("xgboost",nthread = 4,
scale_pos_weight = tune(), alpha = tune()) %> %
set_mode("classification")
bb_rec <- recipe(Y ~ ., data = Train) %> %
step_normalize(all_numeric_predictors()) %> %
step_dummy(all_nominal_predictors())
xgb_wf <- workflow() %> %
add_recipe(bb_rec) %> %
add_model(xgb_spec)
set.seed(Id)
bb_folds <- Train %> %
vfold_cv(v = 6, strata = Y)
eval_metrics <- metric_set(mn_log_loss)
set.seed(Id)
xgb_rs <- tune_race_anova(
object = xgb_wf,resamples = bb_folds,grid = 100,
metrics = metric_set(accuracy),
control = control_race(verbose_elim = TRUE)
)
best_accuracy <- select_best(xgb_rs, metric = "accuracy")
XGB <- finalize_workflow(
xgb_wf,best_accuracy)
XGB_fit <- fit(XGB, data = Train )
PredTrain <- predict(object = XGB_fit,new_data = Train,
type = "class")$.pred_class
PredTest <- predict(object = XGB_fit, new_data = Test,
type = "class")$.pred_class
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
## Corridas métodos -----------------------------------------------------
set.seed(2054)
Datos <- ConjuntoEvaluacion(B=100, por=.8,DataP=DatosT)
M1 <- GenerarResultados(DatosX = Datos,
Method = "logistic",
M2 <- GenerarResultados(DatosX = Datos,
Method = "logistic_lasso",
workers = availableCores())
M3 <- GenerarResultados(DatosX = Datos,
Method = "logistic2_lasso",
workers = availableCores())
M4 <- GenerarResultados(DatosX = Datos,
Method = "RandomForestTuneOOB",
workers = availableCores())
M5<- GenerarResultados(DatosX = Datos,
Method = "XGBtunetinyBSF",
workers = availableCores())
M6<- GenerarResultados(DatosX = Datos,
Method = "XGBtunetinyBA",
workers = availableCores())
M7<- GenerarResultados(DatosX = Datos,
Method = "XGBtuneCV",
workers = availableCores())
M8<- GenerarResultados(DatosX = Datos,
Method = "XGBtuneRM",
workers = availableCores())
# Cálculo del poder predictivo datos simulados
rm(list = ls())
gc()
## Definición de random number generator (RNG) RNGkind(kind="Mersenne-Twister", normal.kind = "Inversion") Para versiones de R mayores o iguales a 3.6 (por reproducibilidad con versiones anteriores a 3.6 usar "Rounding)
RNGkind(kind="Mersenne-Twister", normal.kind = "Inversion", sample.kind="Rejection")
set.seed(12345)
library(xgboost);library(mvtnorm);library(matrixcalc)
library(plyr); library(dplyr); library(tibble)
library(purrr); library(furrr); library(purrrlyr)
library(e1071); library(MASS); library(glmnet)
library(randomForest);library(caret);library(tidymodels)
library(dials); library(tune); library(finetune)
library(ParBayesianOptimization)
### Código para generar las muestras de una Gaussiana condicional
{
Probs.cbin=function(rho, p1){
p11=rho*(p1-(p1)ˆ2)+ (p1)ˆ2
p01=p1-p11
p10=p01
p00=1-p11-2*p01
return(c(p00,p01,p10,p11))
}
Krhoa=function(rho,a, p)
{
K=matrix(0,p,p)
K[1,1]=a
K[1,2]=-rho
K[p,p]=1
K[p,p-1]=-rho
for(i in 2:(p-1)){
K[i,i]=1+rhoˆ2
K[i,i-1]=-rho
K[i,i+1]=-rho
}
K=K*(1/(1-rhoˆ2))
return(K)
}
SimPath <- function(nsim, rho, p.1, as, hs, nCat, nCont){
prob=Probs.cbin(rho, p.1)
Datos=as.data.frame(matrix(0, nrow=nsim, ncol=nCat+nCont))
Datos[,1]=rbinom(nsim,1,p.1)
for(jk in 2:nCat){
Datos[,jk]=rbinom(nsim,1, prob[2]/(1-p.1)*(Datos[,jk-1]==0)+prob[4]/(p.1)*(Datos[,j}
indexi41=(Datos[,nCat]==1)
sigmai41=solve(Krhoa(rho,as[2], nCont))
hi41=rep(0,nCont)
hi41[1]=hs[2]
mui41=sigmai41 %* % hi41
Datos[indexi41,(nCat+1):(nCat+nCont)]=rmvnorm(sum(indexi41),
mui41,sigmai41)
sigmai40=solve(Krhoa(rho,as[1], nCont))
hi40=rep(0,nCont)
hi40[1]=hs[1]
mui40=sigmai40 %* % hi40
Datos[(indexi41==0),(nCat+1):(nCat+nCont)]=rmvnorm(nsim-sum(indexi41), mui40,sigmreturn(Datos)
}
SimPathFun <- function(nsimC, rhoC, p.1C, asC,hsC, nCat, nCont){
Datos0=SimPath(nsimC[1], rhoC[1], p.1C[1], asC[1:2], hsC[1:2], nCat, nCont)
Datos0$Y=0
Datos1=SimPath(nsimC[2], rhoC[2], p.1C[2], asC[3:4], hsC[3:4], nCat, nCont)
Datos1$Y=1
Datos=rbind(Datos0, Datos1)
Datos[,1:nCat] <- lapply(Datos[,1:nCat], factor)
Datos$Y=factor(Datos$Y)
return(Datos)
}
}
{
nsimC <- c(50000,50000)
scalh <- 1
rhoC <- c(.2,-.2)
p.1C <- c(.5,.4)
asC <- c(2,1, 1,2)
hsC <- c(scalh*(asC[1]-rhoC[1]ˆ2)/(1-rhoC[1]ˆ2),0, 0,-scalh*(asC[4]-rhoC[2]ˆ2)/(1-rhoC[nCat <- 4
nCont <- 6
argsSim <- list(nsimC, rhoC, p.1C, asC,hsC , nCat, nCont)
}
## Funciones Auxiliares ----------------------------------------------------
## Función para generar B conjuntos de Train (con nsim simulaciones por Y) y B conjuntos de Test (son 1000 simulaciones por Y). 
ConjuntoEvaluacion <- function(B, nsim){
nsimtest <- c(1000,1000)
nsimtrain <- c(nsim,nsim)
argssimfuntest <- list(nsimtest, rhoC, p.1C, asC, hsC, nCat, nCont)
argssimfuntrain <- list(nsimtrain, rhoC, p.1C, asC, hsC, nCat, nCont)
Train <- list(NA)
Test <- list(NA)
Id <- list(NA)
for(i in 1:B){
set.seed(i
Train[[i]] <- do.call(SimPathFun, argssimfuntrain)
Test[[i]] <- do.call(SimPathFun, argssimfuntest)
Id[[i]]<- i
}
return(list(Train = Train,
Test = Test, Id=Id))
}
## Función para calcular los errores de clasificación por Y y globales (para Train y Test).
ErroresClasificacion <- function(x, y){
Y0Train <- x[1,2]/sum(x[1,])
Y1Train <- x[2,1]/sum(x[2,])
GlobalTrain <- 1-sum(diag(x))/sum(x)
Y0Test <- y[1,2]/sum(y[1,])
Y1Test <- y[2,1]/sum(y[2,])
GlobalTest <- 1-sum(diag(y))/sum(y)
return(data.frame(Y0Train,Y1Train,GlobalTrain,Y0Test,Y1Test,GlobalTest))
}
## Función para generar resultados.
GenerarResultados <- function(B, X, Method, workers) {
DatosX=Datos[[X]]
plan(strategy = multisession,
workers = workers)
Aux1 <- DatosX %> %
future_pmap(.f = get(Method),
.progress = TRUE) %> %
transpose() %> %
pmap(.f = ~ErroresClasificacion(.x,.y)) %> %
ldply(data.frame)
Aux2 <- Aux1 %> %
summarise(Y0Train = mean(Y0Train),
Y1Train = mean(Y1Train),
GlobalTrain = mean(GlobalTrain),
Y0Test = mean(Y0Test),
Y1Test = mean(Y1Test),
GlobalTest = mean(GlobalTest)) %> %
mutate_if(is.numeric, ~.*100) %> %
mutate_if(is.numeric, round, 3) %> %
add_column(nsim = nsim[X],
Method = Method,
.before = "Y0Train")
return(list(Individual = Aux1,
Global = Aux2))
}
### Métodos --------------------------------
### Regresión Logística
logistic <- function(Train, Test, Id) {
logit <- glm(formula = Y~., family = binomial(link=logit), data = Train)
PredTrain <- factor(ifelse(predict(object = logit,
newdata = Train,
type = "response") > .5,
1, 0),levels=levels(Train$Y))
PredTest <- factor(ifelse(predict(object = logit,
newdata = Test,
type = "response") > .5,
1, 0),levels=levels(Train$Y))
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### Regresión L. Lasso
logistic_lasso <- function(Train, Test, Id) {
XTrain <- model.matrix(Y ~ ., data = Train)[,-1]
YTrain <- Train$Y
XTest <- model.matrix(Y ~ ., data = Test)[,-1]
set.seed(Id)
lasso.tun <- cv.glmnet(x = XTrain,
y = YTrain,
nfolds = 10,
type.measure = "class",
gamma = 0,
relax = FALSE,
family = "binomial",
nlambda = 100)
PredTrain <- factor(ifelse(predict(object = lasso.tun,
newx = XTrain,
type = "response",
s = "lambda.min") > .5,
1, 0),levels=levels(Train$Y))
PredTest <- factor(ifelse(predict(object = lasso.tun,
newx = XTest,
type = "response",
s = "lambda.min") > .5,
1, 0),levels=levels(Train$Y))
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### Regresión L. Lasso (2)
logistic2_lasso <- function(Train, Test, Id) {
XTrain <- model.matrix(Y ~ .ˆ2, data = Train)[,-1]
YTrain <- Train$Y
XTest <- model.matrix(Y ~ .ˆ2, data = Test)[,-1]
set.seed(Id)
lasso.tun <- cv.glmnet(x = XTrain,
y = YTrain,
nfolds = 10,
type.measure = "class",
gamma = 0,
relax = FALSE,
family = "binomial",
nlambda = 100)
PredTrain <- factor(ifelse(predict(object = lasso.tun,
newx = XTrain,
type = "response",
s = "lambda.min") > .5,
1, 0),levels=levels(Train$Y))
PredTest <- factor(ifelse(predict(object = lasso.tun,
newx = XTest,
type = "response",
s = "lambda.min") > .5,
1, 0),levels=levels(Train$Y))
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### Modelo Bosque aleatorio
RandomForestTuneOOB <- function(Train, Test, Id) {
mallamtry <- seq(1,10,1)
mallantree <- c(100, 500, 1000)
mallanodesize <- c(1)
oobRFi = function(i, mallaj, k) {
j = max(mallaj)
forest = randomForest(Y ~ .,
mtry = i,
ntree = j,
nodesize = k,
data = Train)
return(forest$err.rate[mallaj])
}
tunoob = function(mallamtry, mallantree, mallanodesize){
nmtry=length(mallamtry)
nntree=length(mallantree)
nnodesize=length(mallanodesize)
iterk=0
tunRF1=matrix(NA, 4, nrow=nmtry*nntree*nnodesize)
for(jmt in 1:nmtry){
for(jns in 1:nnodesize){
iterk=iterk+1
rowinf=(nntree)*(iterk-1)+1
rowsup=(nntree)*(iterk-1)+nntree
tunRF1[rowinf:rowsup,4]=oobRFi(mallamtry[jmt], mallantree, mallanodesize[jns])
tunRF1[rowinf:rowsup,1]=mallamtry[jmt]
tunRF1[rowinf:rowsup,2]=mallantree
tunRF1[rowinf:rowsup,3]=mallanodesize[jns]
}
}
return(tunRF1[which.min(tunRF1[,4]),])
}
set.seed(Id)
OOB <- tunoob(mallamtry, mallantree, mallanodesize)
RF <- randomForest(formula = Y ~ .,
data = Train,
mtry = OOB[1],
ntree = OOB[2],
nodesize = OOB[3])
PredTrain <- predict(object = RF,
newdata = Train,
type = "class")
PredTest <- predict(object = RF,
newdata = Test,
type = "class")
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### XGBoost con busqueda por rellenado espacial
XGBtunetinyBSF <- function(Train, Test, Id ) {
vb_train <- Train
vb_test <- Test
xgb_spec <- boost_tree(
trees = tune(),
tree_depth = tune(), min_n = tune(),
loss_reduction = tune(),
mtry = tune(),
learn_rate = tune()
) %> %
set_engine("xgboost",nthread = 4,
scale_pos_weight = tune(), penalty_L1 =tune()) %> %
set_mode("classification")
param_set <- parameters(
trees(range=c(50,1000)),
tree_depth(range = c(4, 8)),
min_n(range = c(2, 20)),
loss_reduction(range = c(0, 0.2)),
mtry(range = c(2,11)),
learn_rate(range = c(0.01, 0.3)),
scale_pos_weight(range = c(.5,3)),
penalty_L1(range=c(0,1),trans = NULL)
)
xgb_grid <- grid_space_filling(param_set, size = 350)
xgb_wf <- workflow() %> %
add_formula(Y ~ .) %> %
add_model(xgb_spec)
vb_folds <- vfold_cv(vb_train, v=6 ,strata = Y)
set.seed(Id)
xgb_res <-tune_grid(
xgb_wf,
resamples = vb_folds,
grid = xgb_grid,
control = control_grid(save_pred = TRUE)
)
best_accuracy <- select_best(xgb_res, metric = "accuracy")
best_accuracy
XGB <- finalize_workflow(
xgb_wf,
best_accuracy
)
XGB_fit <- fit(XGB, data = vb_train )
PredTrain <- predict(object = XGB_fit,
new_data = vb_train,
type = "class")$.pred_class
PredTest <- predict(object = XGB_fit,
new_data = vb_test,
type = "class")$.pred_class
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### XGBoost con busqueda en malla aleatoria
XGBtunetinyBA <- function(Train, Test, Id ) {
vb_train <- Train
vb_test <- Test
xgb_spec <- boost_tree(
trees = tune(),
tree_depth = tune(), min_n = tune(),
loss_reduction = tune(),
mtry = tune(),
learn_rate = tune()
) %> %
set_engine("xgboost",nthread = 4,
scale_pos_weight = tune(), penalty_L1 =tune()) %> %
set_mode("classification")
param_set <- parameters(
trees(range=c(50,1000)),
tree_depth(range = c(4, 8)),
min_n(range = c(2, 20)),
loss_reduction(range = c(0, 0.2)),
mtry(range = c(2,11)),
learn_rate(range = c(0.01, 0.3)),
scale_pos_weight(range = c(.5,3)),
penalty_L1(range=c(0,1)))
set.seed(Id)
xgb_grid <- grid_random(param_set, size = 10)
xgb_wf <- workflow() %> %
add_formula(Y ~ .) %> %
add_model(xgb_spec)
vb_folds <- vfold_cv(vb_train, v=6 ,strata = Y)
set.seed(Id)
xgb_res <-tune_grid(
xgb_wf,
resamples = vb_folds,
grid = xgb_grid,
control = control_grid(save_pred = TRUE)
)
best_accuracy <- select_best(xgb_res, metric = "accuracy")
XGB <- finalize_workflow(
xgb_wf,
best_accuracy
)
XGB_fit <- fit(XGB, data = vb_train )
PredTrain <- predict(object = XGB_fit,
new_data = vb_train,
type = "class")$.pred_class
PredTest <- predict(object = XGB_fit,
new_data = vb_test,
type = "class")$.pred_class
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### XGBoost con busqueda por optimización bayesiana
XGBtuneCV <- function(Train, Test, Id) {
DM_TR <- model.matrix(Y~.-1,data=Train)
DM_TE <- model.matrix(Y~.-1,data=Test)
TR_LB <- as.factor(Train$Y)
obj_func <- function(eta, max_depth, min_child_weight, subsample, lambda, alpha) {
set.seed(Id)
param <- list(
eta = eta,
max_depth = max_depth,
min_child_weight = min_child_weight,
subsample = subsample,
lambda = lambda,
alpha = alpha,
booster = "gbtree",
objective = "binary:logistic",
eval_metric = "logloss"
)
set.seed(Id)
xgbcv <- xgb.cv(params = param,
data = DM_TR,
label = as.numeric(TR_LB) - 1,
nround = 50,
nfold = 2,
prediction = TRUE,
early_stopping_rounds = 5,
verbose = 0,
maximize = F)
logloss_mean <- xgbcv$evaluation_log$test_logloss_mean
if (sd(logloss_mean) == 0) {
return(list(Score = Inf, nrounds = 0))
}
lst <- list(
Score = -min(logloss_mean),
nrounds = xgbcv$best_iteration
)
return(lst)
}
bounds <- list(eta = c(0.001, 0.2),max_depth = c(1L, 10L),
min_child_weight = c(1, 50),subsample = c(0.1, 1),
lambda = c(1, 10),alpha = c(1, 10))
bounds <- list(eta = c(0.01, 0.3),max_depth = c(3L, 10L),
min_child_weight = c(1, 10),subsample = c(0.5, 1.0),lambda = c(0, 10), alpha = )
set.seed(Id)
bayes_out <- bayesOpt(FUN = obj_func, bounds = bounds, initPoints = 10,
iters.n = 3)
data.frame(getBestPars(bayes_out))
opt_params <- append(list(booster = "gbtree",
objective = "binary:logistic",
eval_metric = "logloss"),
getBestPars(bayes_out))
set.seed(Id)
xgbcv <- xgb.cv(params = opt_params,
data = DM_TR,label = as.numeric(TR_LB)-1,nround = 100,
nfold = 6,prediction = TRUE,early_stopping_rounds = 5,
verbose = 0,maximize = F)
nrounds = xgbcv$best_iteration
mdl<- xgboost(data = DM_TR, label = as.numeric(TR_LB)-1,
params = opt_params, maximize = F,
early_stopping_rounds = 5, nrounds = nrounds,
verbose = 0)
PredTrain_prob <- predict(mdl, DM_TR)
PredTrain <- ifelse(PredTrain_prob > 0.5, 1, 0)
PredTest_prob <- predict(mdl, DM_TE)
PredTest <- ifelse(PredTest_prob> 0.5, 1, 0)
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### XGBoost con optimización de hiperparámetros usando métodos de carrera

XGBtuneRM <- function(Train, Test, Id) {
xgb_spec <- boost_tree(
trees = tune(),
tree_depth = tune(), min_n = tune(),
loss_reduction = tune(),
mtry = tune(),
learn_rate = tune()
) %> %
set_engine("xgboost",nthread = 4,
scale_pos_weight = tune(), alpha = tune()) %> %
set_mode("classification")
bb_rec <- recipe(Y ~ ., data = Train) %> %
step_normalize(all_numeric_predictors()) %> %
step_dummy(all_nominal_predictors())
xgb_wf <- workflow() %> %
add_recipe(bb_rec) %> %
add_model(xgb_spec)
set.seed(Id)
bb_folds <- Train %> %
vfold_cv(v = 6, strata = Y)
eval_metrics <- metric_set(mn_log_loss)
set.seed(Id)
xgb_rs <- tune_race_anova(
object = xgb_wf,
resamples = bb_folds,
grid = 100,
metrics = metric_set(accuracy),
control = control_race(verbose_elim = TRUE) )
best_accuracy <- select_best(xgb_rs, metric = "accuracy")
best_accuracy
XGB <- finalize_workflow(
xgb_wf,
best_accuracy
)
XGB_fit <- fit(XGB, data = Train )
PredTrain <- predict(object = XGB_fit,
new_data = Train,
type = "class")$.pred_class
PredTest <- predict(object = XGB_fit,
new_data = Test,
type = "class")$.pred_class
return(list(TrainGlobal = table(Train$Y, PredTrain),
TestGlobal = table(Test$Y, PredTest)))
}
### Corridas métodos --------------------------------------------------------
nsim <- c(50,100,1000)
Datos <- lapply(X = nsim,
FUN = ConjuntoEvaluacion,
B = 100)
M1 <- lapply(X = c(1:length(nsim)),
FUN = GenerarResultados,
B = 100,
Method = "logistic",
workers = availableCores())
tictoc::tic()
M2 <- lapply(X = c(1:length(nsim)),
FUN = GenerarResultados,
B = 100,
Method = "logistic_lasso",
workers = availableCores())
M3 <- lapply(X = c(1:length(nsim)),
FUN = GenerarResultados,
B = 100,
Method = "logistic2_lasso",
workers = availableCores())
M4 <- lapply(X = c(1:length(nsim)),
FUN = GenerarResultados,
B = 100,
Method = "RandomForestTuneOOB",
workers = availableCores())
M5 <- lapply(X = c(1:length(nsim)),
FUN = GenerarResultados,
B = 100,
Method = "XGBtunetinyBSF",
workers = availableCores())
M6 <- lapply(X = c(1:length(nsim)),
FUN = GenerarResultados,
B = 100,
Method = "XGBtunetinyBA",
workers = availableCores())
M7 <- lapply(X = c(1:length(nsim)),
FUN = GenerarResultados,
B = 100,
Method = "XGBtuneRM",
workers = availableCores())
