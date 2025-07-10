import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(f, df, x0, epsilon=0.001, delta_x=0.01, max_iter=100):
    x = x0
    pasos = [x]
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-8:
            print("Derivada muy cercana a cero. Método detenido.")
            break
        x_new = x - fx / dfx
        pasos.append(x_new)
        if abs(x_new - x) < epsilon:
            break
        x = x_new
    return x, pasos

def volumen_caja(L):
    return 200*L - 60*L**2 + 4*L**3

def d_volumen_caja(L):
    return 200 - 120*L + 12*L**2

L_opt, _ = newton_raphson(volumen_caja, d_volumen_caja, x0=1.0)

L_vals = np.linspace(0, 5, 1000)
V_vals = volumen_caja(L_vals)

plt.figure(figsize=(8, 5))
plt.plot(L_vals, V_vals, 'b-', label='Volumen')
plt.plot(L_opt, volumen_caja(L_opt), 'go', label='Máximo')
plt.title("Volumen de la caja")
plt.xlabel("Lado (cm)")
plt.ylabel("Volumen (cm³)")
plt.grid(True)
plt.legend()
plt.show()

def area_lata(r):
    return 2 * np.pi * r**2 + (400 / r)

def d_area_lata(r):
    return 4 * np.pi * r - 400 / r**2

r_opt, _ = newton_raphson(area_lata, d_area_lata, x0=1.0)

r_vals = np.linspace(0.1, 5, 1000)
A_vals = area_lata(r_vals)

plt.figure(figsize=(8, 5))
plt.plot(r_vals, A_vals, 'b-', label='Área')
plt.plot(r_opt, area_lata(r_opt), 'go', label='Mínimo')
plt.title("Área de la lata")
plt.xlabel("Radio (cm)")
plt.ylabel("Área (cm²)")
plt.grid(True)
plt.legend()
plt.show()

def f1(x):
    return x**2 + 54/x

def df1(x):
    return 2*x - 54/x**2

x1, _ = newton_raphson(f1, df1, x0=2.0)
x_vals1 = np.linspace(0.1, 10, 1000)
y_vals1 = f1(x_vals1)

plt.figure(figsize=(8, 5))
plt.plot(x_vals1, y_vals1, 'b-', label='f(x)')
plt.plot(x1, f1(x1), 'go', label='Extremo')
plt.title("f(x) = x² + 54/x")
plt.grid(True)
plt.legend()
plt.show()

def f2(x):
    return x**3 + 2*x - 3

def df2(x):
    return 3*x**2 + 2

x2, _ = newton_raphson(f2, df2, x0=1.0)
x_vals2 = np.linspace(0, 5, 1000)
y_vals2 = f2(x_vals2)

plt.figure(figsize=(8, 5))
plt.plot(x_vals2, y_vals2, 'b-', label='f(x)')
plt.axhline(0, color='gray', linestyle='--')
plt.plot(x2, f2(x2), 'go', label='Raíz')
plt.title("f(x) = x³ + 2x − 3")
plt.grid(True)
plt.legend()
plt.show()

def f3(x):
    return x**4 + x**2 - 33

def df3(x):
    return 4*x**3 + 2*x

x3, _ = newton_raphson(f3, df3, x0=2.0)
x_vals3 = np.linspace(-2.5, 2.5, 1000)
y_vals3 = f3(x_vals3)

plt.figure(figsize=(8, 5))
plt.plot(x_vals3, y_vals3, 'b-', label='f(x)')
plt.plot(x3, f3(x3), 'go', label='Extremo')
plt.title("f(x) = x⁴ + x² − 33")
plt.grid(True)
plt.legend()
plt.show()

def f4(x):
    return 3*x**4 - 8*x**3 - 6*x**2 + 12*x

def df4(x):
    return 12*x**3 - 24*x**2 - 12*x + 12

x4, _ = newton_raphson(f4, df4, x0=2.0)
x_vals4 = np.linspace(-1.5, 3, 1000)
y_vals4 = f4(x_vals4)

plt.figure(figsize=(8, 5))
plt.plot(x_vals4, y_vals4, 'b-', label='f(x)')
plt.plot(x4, f4(x4), 'go', label='Extremo')
plt.title("f(x) = 3x⁴ − 8x³ − 6x² + 12x")
plt.grid(True)
plt.legend()
plt.show()
