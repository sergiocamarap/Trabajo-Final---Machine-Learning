###############################################################################

# Programa: Trabajo Final - Machine Learning
# Hecho por: Sergio Cámara Peña e Iñigo Clemente Larramendi
# Fecha: 09/03/2022
# Versión: 1.0

# Nota: Guardado con el encoding UTF-8.

###############################################################################

setwd("C:/Users/Sergio/Desktop/Master_en_Metodos_Computacionales/Segundo_Semestre/Aprendizaje_automatico(Machine_Learning)/Trabajo_Final")

source("Trabajo_Final_Modelo.R",encoding = "UTF-8")
source("Trabajo_Final_Prediccion_Funcion.R",encoding = "UTF-8")

x_test <- read.csv("X_train.csv", encoding = "UTF-8",na.strings=c(""," ","NA"))
y_test <- Prediccion(x_test)
write.csv(y_test,"y_test_predicted.csv", row.names = FALSE)

############################# FIN DEL PROGRAMA #################################
