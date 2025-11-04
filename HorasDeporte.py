import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

datos = {
    "Nombre":["Pau","Sergio","Leonard","Lluc","Samuel","Sofia","Abdrahim","Mehdi","Mathew","Hamza"],
    "HorasDeporte": [2,6,1,8,8,3,1,10,3,10],
    "Nota":[8,4,2,6,4,0,1,3,4,0],
}

df = pd.DataFrame(datos)
print("Datos introducidos manualmente: \n", df)

print("\nMedia de las notas :", df["Nota"].mean())
print("Maximo y minimo de las notas:",df["Nota"].max(), df["Nota"].min())

x = df[["HorasDeporte"]]
y = df["Nota"]

model = LinearRegression()
model.fit(x, y)

horas_nuevas2 = [[10.22]]
prediccion2 = model.predict(horas_nuevas2)
print(f"\nPrediccion de nota media por 2 horas estudio/semana: {prediccion2[0] : .2f}")

#Grafico con recta de regresion
plt.scatter(df["HorasDeporte"], df["Nota"], color = 'blue', label = 'Datos reales')

#Crear valores para la linea de regresion

x_range = np.linspace(df["HorasDeporte"].min(), df["HorasDeporte"].max(), 100).reshape(-1, 1)
y_pred = model.predict(x_range)

plt.plot(x_range, y_pred, color = 'red', label = 'Recta de regresion')
plt.xlabel("HorasDeporte")
plt.ylabel("Nota")
plt.title("Relacion nota y horas de deporte")
plt.legend()
plt.show()