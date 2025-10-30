import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

datos = {
    "Nombre":["Pau","Sergio","Leonard","Lluc","Samuel","Sofia","Abdrahim","Mehdi","Mathew","Hamza"],
    "Horas dormidas":[7,8,7,7,5,5,4,8,5,1],
    "Tiempo de uso de movil en una hora":[5,4,40,20,25,10,15,10,3,2],
    "numero de pausas de 5 min": [2,1,1,3,2,5,3,2,6,10],
    "Horas de deporte semanales": [2,6,1,8,8,3,1,10,3,10],
    "Nota examen":[8,4,2,6,4,0,1,3,4,0],
}
df = pd.DataFrame(datos)
print("Dades introduïdes manualment: /n",df)
print("mitjana de les notes",df["Nota examen"].mean())
print("Máxim i mínim de les notes:", df["Nota examen"].max(), "i" , df["Nota examen"].min())
#Grafica nota y horas dormidas:
x = df[["Horas dormidas"]]
y = df["Nota examen"]
model = LinearRegression()
model.fit(x, y)
plt.scatter(df[["Horas dormidas"]], df["Nota examen"], color='blue', label='Dadesr reals')

x_range = np.linspace(df["Horas dormidas"].min(), df["Horas dormidas"].max(), 100).reshape(-1,1)
y_pred = model.predict(x_range)

plt.plot(x_range, y_pred, color = 'red', label = 'Recta de regresion')
plt.xlabel("Horas dormidas")
plt.ylabel("Nota examen")
plt.title("Relacion horas de estudio vs nota media")
plt.legend()
plt.show()