import matplotlib.pyplot as plt
import numpy as np

predictions = np.array([95.16129032258065 , 90.32258064516128,96.7741935483871,98.38709677419355,96.7741935483871 ])
classifier = np.array(["KNN","Random \nForest","Support \nVector \nMachine","Logistic \nRegression","Decision \nTree"])

plt.barh(classifier,predictions,color="orange")
plt.show()
