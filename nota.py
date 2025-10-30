import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dades = {
    "Nom":["Benji","Godfrey","Pau","Lluc","Celvin","Sergio","Leo","Mehdi"],
    "Hores":[4,4,6.5,4,4,0.5,0.5,3],
    "Nota":[6.05, 5.05,8.1,6.55,4.1,4.3,2.6,2.7]
}
df = pd.DataFrame(dades)
print("Dades introduïdes manualment: /n",df)
print("mitjana de les notes",df["Nota"].mean())
print("Máxim i mínim de les notes:", df["Nota"].max(), "i" , df["Nota"].min())

x = df[["Hores"]]
y = df["Nota"]
model = LinearRegression()
model.fit(x, y)
Hores_abde = [[0]]
Prediccio_abde = model.predict(Hores_abde)
print(f"/nPredicció nota de abderra: {Prediccio_abde[0]:.2f}")

plt.scatter(df["Hores"], df["Nota"], color='blue', label='Dadesr reals')

x_range = np.linspace(df["Hores"].min(), df["Hores"].max(), 100).reshape(-1,1)
y_pred = model.predict(x_range)

plt.plot(x_range, y_pred, color = 'red', label = 'Recta de regresion')
plt.xlabel("Hores")
plt.ylabel("Nota")
plt.title("Relacion horas de estudio vs nota media")
plt.legend()
plt.show()