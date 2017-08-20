#!/usr/bin/env python3
import matplotlib.pyplot as plt

sizes = list(range(50, 801, 50))
accuracies = [0.9977619047619047, 
              0.9992857142857143,
              0.9996666666666667, 
              0.9996666666666667, 
              0.9996666666666667,
              0.9996666666666667,
              0.9996666666666667,
              0.9996666666666667, 
              0.9997142857142857,
              0.9997142857142857,
              0.9997142857142857,
              0.9997142857142857,
              0.9997142857142857,
              0.9997142857142857,
              0.9997142857142857,
              0.9997142857142857]

plt.plot(sizes, accuracies, 'g.-')
plt.title('Growth of Accuracy over Training Size per Language')
plt.xlabel('# of Training Documents per Language')
plt.ylabel('Accuracy')
plt.show()
