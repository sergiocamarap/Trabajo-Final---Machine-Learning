###############################################################################

# Programa: Trabajo Final - Machine Learning
# Hecho por: Sergio Cámara Peña e Iñigo Clemente Larramendi
# Fecha: 09/03/2022
# Versión: 1.0

# Nota: Guardado con el encoding UTF-8.

###############################################################################

Modelo <- function(){
  
  # Inicialización - Carga de paquetes.
  
  library(tidyverse)
  library(caret)
  library(corrplot)
  library(fastDummies)
  library(mlr)
  
  # Carga inicial de los datos necesarios
  X_train <- read.csv("X_train.csv", encoding = "UTF-8",na.strings=c(""," ","NA"))
  Y_train <- read.csv("Y_train.csv", encoding = "UTF-8")
  Y_train_orig <- Y_train
  
  # Tratamiento inicial de los datos.
  X_train <- X_train %>% 
    dplyr::select(-c(hº.act.cerca.sem,origen.antepasados..extranjeros.,origen.antepasados..españa.,fecha))
  
  X_train[X_train=="No lo se"] <- NA
  X_train[X_train=="No lo se, No"] <- "No"
  
  X_train$Familiar.miope.num. <- as.character(X_train$Familiar.miope.num.)
  
  # Clasificador automatico para los tipos de columnas
  N_variables <- ncol(X_train)
  indices <- seq(1,N_variables)
  colclasses <- vector(length = N_variables)
  
  for(i in indices){
    tipo <- class(X_train[,i])
    tama <- length(unique(X_train[,i]))
    if (tipo == "character") {
      colclasses[i] <- "Categorica"
    }else{
      colclasses[i] <- "Numerica"
    }
  }
  
  Tipo_columnas <- data.frame(colnames(X_train),colclasses)
  colnames(Tipo_columnas) <- c("Variable","Tipo")
  
  # Imputar valores numéricos con la media.
  Indices_col_con_NA_num <- which((colSums(is.na(X_train)) > 0) & (colclasses == "Numerica"))
  
  for (i in Indices_col_con_NA_num) {
    X_train[is.na(X_train[,i]),i] <- mean(X_train[,i], na.rm = TRUE)
  }
  
  # Imputar valores categóricos con la moda.
  getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  
  Indices_col_con_NA_cat <- which((colSums(is.na(X_train)) > 0) & (colclasses == "Categorica"))
  
  for (i in Indices_col_con_NA_cat) {
    X_train[is.na(X_train[,i]),i] <- getmode(X_train[,i])
  }
  
  X_train <- X_train[complete.cases(X_train),] #Los valores que aun asi no se hayan imputado por algún error, son retirados.
  
  # Normalización
  min_max_norm <- function(x) {
    (x - min(x)) / (max(x) - min(x))
  }
  
  Indices_col_name <- which((colclasses == "Numerica") & (Tipo_columnas$Variable != "edad"))
  
  for (i in Indices_col_name) {
    X_train[,i] <- as.data.frame(lapply(X_train[i], min_max_norm))
  }
  
  # Feature Selection - Retirada de features numéricas redundantes
  col_numericas <- which(colclasses == "Numerica")
  
  set.seed(1032)
  
  correlationMatrix <- cor(X_train[,col_numericas])
  
  highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.9)
  
  #corrplot(correlationMatrix)
  
  X_train <- X_train %>% 
    dplyr::select(-highlyCorrelated)
  
  # Generación de Dummy variables
  categoric_variables <-  colnames(X_train[sapply(X_train, class)== "character"])
  categoric_variables <- categoric_variables[1:length(categoric_variables)]
  
  X_train <- dummy_cols(X_train, select_columns = categoric_variables)
  
  X_train <- X_train %>% 
    dplyr::select(-categoric_variables)
  
  colnames(X_train) <- make.names(colnames(X_train),unique = TRUE)
  
  Nombres_respuesta <- colnames(Y_train)
  Y_train <- dummy_cols(Y_train)
  
  Y_train <-  Y_train %>% 
    dplyr::select(-Nombres_respuesta)
  
  contador <- length(Y_train)
  
  for (i in 1:contador) {
    Y_train[,i] <- as.logical(Y_train[,i])
  }
  
  Train_merge <- cbind(X_train,Y_train)
  
  # Multilabel Classification Model
  labels = colnames(Y_train)
  
  Task_data = makeMultilabelTask(id = "multi", data = Train_merge, target = labels)
  
  set.seed(1234)
  lrn.rfsrc = makeMultilabelClassifierChainsWrapper("classif.rpart")
  
  Separador <- round(dim(Train_merge)[1]*0.8)
  
  mod = mlr::train(lrn.rfsrc, Task_data, subset = 1:Separador)
  
  return(mod)
}
############################# FIN DEL PROGRAMA #################################
