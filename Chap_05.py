import os
import pandas as pd
import numpy as np
import wradlib.ipol as ipol
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from pykrige.uk import UniversalKriging

os.getcwd()
os.chdir("C:/Users/johan/OneDrive/Dokumente/GitHub/mhbil/data")

# Data
df_vil = pd.read_excel("villages.xls")
X = df_vil["AD"]
E_vil = df_vil["id"]
df_meg = pd.read_csv("meg_dw.csv", sep=';')

df_vil_compact = pd.concat([X, E_vil], axis=1)  # not necessary?
df_meg_nodes = pd.read_csv("voronoi_nodes_radius.csv", sep=',')

# 5.1 Regression ===============================
# https://towardsdatascience.com/linear-regression-in-6-lines-of-python-5e1d0cd05b8d

# Pearson Stats
cor = pearsonr(X, E_vil)
print(cor)

# Simple Linear Regression

X_np = np.array(df_vil["AD"]).reshape((-1, 1))  # Is required to be 2D
E_vil_np = np.array(df_vil["id"])

lm = LinearRegression()
lm.fit(X_np, E_vil_np)
E_vil_pred = lm.predict(X_np)

plt.scatter(X_np, E_vil_np, color="Aqua")
plt.plot(X_np, E_vil_pred, color="blue")

# 5.1.2 Linear Models T1_vil

plt.scatter(X_np, E_vil_np, color="Black")
plt.plot([df_vil.iat[0, 2], df_vil.iat[12, 2]], [df_vil.iat[0, 0], df_vil.iat[12, 0]], color="red")
years = np.arange(1259, 1350, step=1)
y = 1 + 0.003 * (years-1259) ** 2

plt.plot(years, y, color="blue")

# Goodness of fit test / Chi Square

E = E_vil
T1_vil = 1 + (12/91)*(X - 1259)
T2_vil = 1 + 0.003 * (X-1259)**2

sum(((E-T1_vil)**2)/T1_vil)
# 1.955191
sum(((E-T2_vil)**2)/T2_vil)
# 12.12744

# T3_vil model / c=0

yr = X - 1259
yr = np.array(yr).reshape((-1, 1))
E = np.array(E).reshape((-1, 1))

regr = LinearRegression()
regr.fit(yr, E)
T3_vil = regr.predict(yr)

# Regression coefficient
reg_coeff = regr.coef_
mean_squared_error(E, yr)

plt.scatter(X - 1259, E_vil,  color='black')
plt.plot(yr, T3_vil, color='black', linewidth=1)
plt.xlabel('X-1259')
plt.ylabel('E_vil')

# Fitted values -> predicted values
print(T3_vil)
fT3_vil = T3_vil
sum(((E-fT3_vil)**2)/fT3_vil)
# array([1.12807617])

# R2
print(r2_score(E, T3_vil))

# More quadratic approach
regr.fit(yr**2, E)
T4_vil = regr.predict(yr**2)
reg_coeff_T4 = regr.coef_

fT4_vil = T4_vil
sum((E-fT4_vil)**2/fT4_vil)

# Polynomial approach
yr_poly = PolynomialFeatures(degree=4).fit_transform(yr)
regr.fit(yr_poly, E)
T5_vil = regr.predict(yr_poly)
reg_coeff_poly = regr.coef_

fT5_vil = T5_vil
sum((E-fT5_vil)**2/fT5_vil)
print(r2_score(E, T5_vil)) # Good fit -> Plot

plt.scatter(X - 1259, E_vil,  color='black')
plt.plot(yr, T5_vil, color='black', linewidth=1)
plt.xlabel('X-1259')
plt.ylabel('E_vil')

# 5.1.3 Model Choice, Overfitting and Decomposition ===============================

# läuft immer noch nicht
chi = np.arange(1, 21).reshape(-1, 1)

with np.nditer(chi) as it:
    for i in it:
        regs = LinearRegression()
        yr_poly_for = PolynomialFeatures(degree=i).fit_transform(yr)
        regs.fit(yr_poly_for, E)
        T6_vil = regs.predict(yr_poly_for)
        chi[i] = sum(((E - T6_vil)**2)/T6_vil)

with np.nditer(chi) as it:
    for i in it:
        print(i)

plt.plot(chi, color='black', linewidth=1)

# lalala
chi_2 = list(range(1, 21))
regs = LinearRegression()
yr_poly_for = PolynomialFeatures(degree=12).fit_transform(yr)
regs.fit(yr_poly_for, E)
T6_vil = regs.predict(yr_poly_for)
chis= sum(((E - T6_vil)**2)/T6_vil)


# 5.2.2 Inverse Distance Weighting ===============================
df_voronoi_nodes = pd.read_csv("voronoi_nodes_radius.csv")
np_voro = np.asarray(df_voronoi_nodes)

x = np.array(df_meg['x'])
y = np.array(df_meg['y'])

# Define the borders
xmin = min(x)
xmax = max(x)
ymin = min(y)
ymax = max(y)

# Source Points / Megalith Points
src = np.vstack(([x], [y])).transpose()
vals = df_meg['id']  # Values = ID

# Create meshgrid and target points
xtrg = np.linspace(src[:, 0].min(), src[:, 0].max(), 200)
ytrg = np.linspace(src[:, 1].min(), src[:, 1].max(), 200)
trg = np.meshgrid(xtrg, ytrg)
trg = np.vstack((trg[0].ravel(), trg[1].ravel())).T

# Plot Definition


def gridplot(interpolated, title=""):
    plt.pcolormesh(xtrg, ytrg, interpolated.reshape((len(xtrg), len(ytrg))))
    plt.axis("tight")
    plt.scatter(src[:, 0], src[:, 1], facecolor="None", s=50, marker='s')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")


# Interpolation IDW
idw = ipol.Idw(src, trg)
gridplot(idw(vals.ravel()), "IDW")

# Other approach for IDW
ip_near = ipol.Nearest(src, trg)
maxdist = trg[1, 0] - trg[0, 0]
result_near = ip_near(vals.ravel(), maxdist=maxdist)

# 5.2.3 Kriging ===============================
df_meg_nodes_np = np.array(df_meg_nodes)
gridx = np.arange(0.0, 5.5, 0.5)
gridy = np.arange(0.0, 5.5, 0.5)

# Ordninary Kriging
ok = ipol.OrdinaryKriging(src, trg)
gridplot(ok(vals.ravel()), "Ordinary Kriging")

# Universal Kriging
UK = UniversalKriging(df_meg_nodes_np[:, 0], df_meg_nodes_np[:, 4], df_meg_nodes_np[:, 1],
                      variogram_model='linear', drift_terms=['regional_linear'])
z, ss = UK.execute('grid', xtrg, ytrg)

# Variogram KDE Sample Points
from skgstat import Variogram

data = np.array(df_meg_nodes_np[:, [5, 6, 7]])
coordinates = np.array(data[:, [0, 1]])
values = np.array(data[:, 2])

V_exp = Variogram(coordinates, values, model="exponential")
V_exp.plot()

V_gaussian = Variogram(coordinates, values, model="gaussian")
V_gaussian.plot()

V_matern = Variogram(coordinates, values, model="matern")
V_matern.plot()

V_stable = Variogram(coordinates, values, model="stable")
V_stable.plot()

