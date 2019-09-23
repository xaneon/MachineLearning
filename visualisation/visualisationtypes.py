from matplotlib import pyplot as plt
import numpy as np
from numpy.random import randint, randn

num = 42
years = [randint(1950, 2010) for i in range(num)]
gdp = [round(randint(300, 15_000) + randn(1)[0], 2)
       for i in range(num)]

plt.plot(sorted(years), sorted(gdp),
         color="green", marker="o", linestyle="solid")
plt.title("Nominal GDP")
plt.ylabel("Billions of $")
plt.savefig("lineplot.png")
