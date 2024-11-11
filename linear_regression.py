import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

time_study = np.array([33, 47, 33, 59, 26, 5, 41, 18, 53, 9, 30, 55, 22]).reshape(-1,1)
scores = np.array([23, 3, 27, 58, 16, 49, 7, 35, 21, 50, 12, 39, 24]).reshape(-1,1)

#sao números gerados aleatoriamente usando GPT

model = LinearRegression()
model.fit(time_study, scores)

plt.scatter(time_study,scores)
plt.plot(np.linspace(0,70,100), model.predict(np.linspace(0,70,100).reshape(-1,1)), 'r')
plt.ylim(0,100)
plt.show()