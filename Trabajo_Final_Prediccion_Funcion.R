###############################################################################

# Programa: Trabajo Final - Machine Learning
# Hecho por: Sergio Cámara Peña e Iñigo Clemente Larramendi
# Fecha: 09/03/2022
# Versión: 1.0

# Nota: Guardado con el encoding UTF-8.

###############################################################################

Prediccion <- function(x_test){
  
  # Referencia al modelo
  
  mod <- Modelo()
  # Tratamiento inicial de los datos.
  x_test <- x_test %>% 
    dplyr::select(-c(hº.act.cerca.sem,origen.antepasados..extranjeros.,origen.antepasados..españa.,fecha))
  
  x_test[x_test=="No lo se"] <- NA
  x_test[x_test=="No lo se, No"] <- "No"
  
  x_test$Familiar.miope.num. <- as.character(x_test$Familiar.miope.num.)
  
  # Clasificador automatico para los tipos de columnas
  N_variables <- ncol(x_test)
  indices <- seq(1,N_variables)
  colclasses <- vector(length = N_variables)
  
  for(i in indices){
    tipo <- class(x_test[,i])
    tama <- length(unique(x_test[,i]))
    if (tipo == "character") {
      colclasses[i] <- "Categorica"
    }else{
      colclasses[i] <- "Numerica"
    }
  }
  
  Tipo_columnas <- data.frame(colnames(x_test),colclasses)
  colnames(Tipo_columnas) <- c("Variable","Tipo")
  
  # Imputar valores numéricos con la media.
  Indices_col_con_NA_num <- which((colSums(is.na(x_test)) > 0) & (colclasses == "Numerica"))
  
  for (i in Indices_col_con_NA_num) {
    x_test[is.na(x_test[,i]),i] <- mean(x_test[,i], na.rm = TRUE)
  }
  
  # Imputar valores categóricos con la moda.
  getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  
  Indices_col_con_NA_cat <- which((colSums(is.na(x_test)) > 0) & (colclasses == "Categorica"))
  
  for (i in Indices_col_con_NA_cat) {
    x_test[is.na(x_test[,i]),i] <- getmode(x_test[,i])
  }
  
  x_test <- x_test[complete.cases(x_test),] #Los valores que aun asi no se hayan imputado por algún error, son retirados.
  
  # Normalización
  min_max_norm <- function(x) {
    (x - min(x)) / (max(x) - min(x))
  }
  
  Indices_col_name <- which((colclasses == "Numerica") & (Tipo_columnas$Variable != "edad"))
  
  for (i in Indices_col_name) {
    x_test[,i] <- as.data.frame(lapply(x_test[i], min_max_norm))
  }
  
  # Feature Selection - Retirada de features numéricas redundantes
  col_numericas <- which(colclasses == "Numerica")
  
  set.seed(1032)
  
  correlationMatrix <- cor(x_test[,col_numericas])
  
  highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.9)
  
  x_test <- x_test %>% 
    dplyr::select(-highlyCorrelated)
  
  # Generación de Dummy variables
  categoric_variables <-  colnames(x_test[sapply(x_test, class)== "character"])
  categoric_variables <- categoric_variables[1:length(categoric_variables)]
  
  x_test <- dummy_cols(x_test, select_columns = categoric_variables)
  
  x_test <- x_test %>% 
    dplyr::select(-categoric_variables)
  
  colnames(x_test) <- make.names(colnames(x_test),unique = TRUE)
  
  # Prediccion
  
  newdata.pred = predict(mod, newdata = x_test)
  
  Response <- newdata.pred$data
  Longitud <- length(Response)
  Response <- Response[,(Longitud-10):Longitud]
  
  contador2 <- 1:length(Response)
  
  for (i2 in contador2) {
    Response[,i2] <- as.integer(Response[,i2])
  }
  
  MultChoiceCondense<-function(vars,indata){
    tempvar<-matrix(NA,ncol=1,nrow=length(indata[,1]))
    dat<-indata[,vars]
    for (i in 1:length(vars)){
      for (j in 1:length(indata[,1])){
        if (dat[j,i]==1) tempvar[j]=i
      }
    }
    return(tempvar)
  }
  
  # Predicciones en formato Y_train
  Resultados <- Response
  Resultados$M <- MultChoiceCondense(c("response.M_NO","response.M_SI"),Response)
  Resultados$MM <- MultChoiceCondense(c("response.MM_NO","response.MM_SI"),Response)
  Resultados$Combo <- MultChoiceCondense(c("response.Combo_C","response.Combo_M","response.Combo_MM"),Response)
  Resultados$DCombo <- MultChoiceCondense(c("response.DCombo_C","response.DCombo_M1","response.DCombo_M2","response.DCombo_MM"),Response)
  
  Resultados <- Resultados %>% 
    dplyr::select(c(M,MM,Combo,DCombo))
  
  Resultados$M <- factor(Resultados$M,levels = c(1,2),labels = c("NO","SI"))
  Resultados$MM <- factor(Resultados$MM,levels = c(1,2),labels = c("NO","SI"))
  Resultados$Combo <- factor(Resultados$Combo,levels = c(1:3),labels = c("C","M","MM"))
  Resultados$DCombo <- factor(Resultados$DCombo,levels = c(1:4),labels = c("C","M1","M2","MM"))
  
  return(Resultados)
}
############################# FIN DEL PROGRAMA #################################