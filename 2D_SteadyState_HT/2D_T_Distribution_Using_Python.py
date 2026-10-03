# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 08:07:37 2026

@author: ER
"""
#Let's import some modules first
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

#%% Second, define plate dimensions and temperature values of its boundaries
L_plate=2 #m
W_plate=1 #m


T_left=25 #degC
T_right=T_left; T_bottom=T_left
T_top=100 #degC

if not (L_plate>0 and W_plate>0):
    raise ValueError ("Plate dimensions must be positive")

# '''
# Can you put here a raise ValueError condition if T_left!=T_bottom!=T_right?
# '''

#%% Declare the symbolic variables
x, y, L, W= sp.symbols("x y L W",positive=True,real=True)

pi=sp.pi

T1,T2=sp.symbols("T1 T2", real=True)

N=sp.symbols("N",integer=True, positive=True)
k=sp.symbols("k",integer=True, nonnegative=True)
n=sp.symbols("n", integer=True, odd=True)


# n=2*k+1 #this is to make odd Fourier index, e.g., k=0,1,2,... > n=1,3,5,...
n_relation=sp.Eq(n,2*k+1)

print("\nRelation between n and k:\n")
sp.pprint(n_relation)  #pprint stands for pretty print
#%% Defining theta term

inside_sum=1/n*(sp.sin(n*pi*x/L))*sp.sinh(n*pi*y/L)/sp.sinh(n*pi*W/L)

#Let's check we type things correct
print("\n\nTerm inside the summation:\n")
sp.pprint(inside_sum)

#%% Replace n by 2*k + 1 to generate only odd terms
odd_term = inside_sum.subs(n, 2*k + 1)

print("\nTerm after applying n = 2*k + 1:\n")
sp.pprint(odd_term)

#%% Construct the finite summation
theta_sum = sp.Sum(odd_term, (k, 0, N - 1))

print("\nFinite summation using N odd terms:\n")
sp.pprint(theta_sum)

#%% Construct T(x,y) using N odd terms
T_xy = T1 + (T2 - T1)*(4/pi)*theta_sum

print("\nTemperature expression, T(x,y):\n")
sp.pprint(T_xy)

#%% Substitute the plate dimensions and boundary temperatures
T_xy_plate = T_xy.subs({
    T1: T_left,
    T2: T_top,
    L: L_plate,
    W: W_plate
})

print("\nTemperature expression for the specified plate:\n")
sp.pprint(T_xy_plate)

#%% Use the first N odd terms
no_terms=80
T_xy_N = T_xy_plate.subs(N, no_terms).doit()

print(f"\nTemperature expression using {no_terms} odd terms:\n")
sp.pprint(T_xy_N)

#%% Evaluate temperature at one point
T_at_point = T_xy_N.subs({
    x: 0.8,
    y: 0.4
}).evalf()

print(f"\nTemperature at (x,y) using {no_terms} terms = (0.8,0.4) m:\n")
sp.pprint(T_at_point)

#%% Convert the symbolic expression to a numerical function
T_function = sp.lambdify( 
    (x, y),
    T_xy_N,
    modules="numpy"
)

'''
Note that lambdify converts symbolic expressions into highly efficient 
numerical functions that can be evaluated rapidly using external libraries,
e.g., NumPy or SciPy. 
The function takes three primary arguments:lambdify(variables, expression, modules)
'''

print("\nThe numerical temperature function is ready.")

# Verify it at the selected point
print("\nTemperature from the numerical function:")
print(T_function(0.8, 0.4))

#%% Create the coordinate grid
Nx = 101
Ny = 101

x_values = np.linspace(0, L_plate, Nx)
y_values = np.linspace(0, W_plate, Ny)

X, Y = np.meshgrid(x_values, y_values)

print("\nGrid dimensions:")
print("X shape =", X.shape)
print("Y shape =", Y.shape)

#%% Evaluate the temperature across the plate
T_grid = T_function(X, Y)

print("\nTemperature-grid dimensions:")
print("T_grid shape =", T_grid.shape)

print("\nMinimum calculated temperature:")
print(np.min(T_grid))

print("\nMaximum calculated temperature:")
print(np.max(T_grid))

#%% Plot the temperature distribution
plt.figure(figsize=(10, 5))

temperature_plot = plt.contourf(
    X,
    Y,
    T_grid,
    levels=100,
    cmap="jet"
)

plt.colorbar(
    temperature_plot,
    label="Temperature [°C]"
)

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"2D Steady-State Temperature Distribution — N = {no_terms}")

plt.axis("equal")
plt.xlim(0, L_plate)
plt.ylim(0, W_plate)

plt.tight_layout()
plt.show()