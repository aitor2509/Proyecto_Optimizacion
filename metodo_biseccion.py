import numpy as np
import matplotlib.pyplot as plt

def bounding_phase(f, x0, delta=0.1, max_iter=100):
    k = 0
    if f(x0 - abs(delta)) >= f(x0) and f(x0) <= f(x0 + abs(delta)):
        return (x0 - abs(delta), x0 + abs(delta))
    if f(x0 - abs(delta)) < f(x0):
        delta = -abs(delta)
    else:
        delta = abs(delta)
    x_prev = x0
    f_prev = f(x_prev)
    x_curr = x0 + delta
    f_curr = f(x_curr)
    k = 1
    while f_curr < f_prev and k < max_iter:
        delta *= 2
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_prev + delta
        f_curr = f(x_curr)
        k += 1
    if x_prev < x_curr:
        return (x_prev - delta/2, x_curr)
    else:
        return (x_curr, x_prev - delta/2)


def biseccion_min(f, a, b, epsilon=0.001):
    puntos = []
    if f(a) == f(b):
      return (a+b)/2, f((a+b)/2), puntos
    
    while (b - a) > epsilon:
        xm = (a + b) / 2
        x1 = a + (xm - a) / 2
        x2 = b - (b - xm) / 2
        puntos.append((x1, x2))
        if f(x1) < f(xm): b = xm
        elif f(x2) < f(xm): a = xm
        else: a = x1; b = x2
    
    xmin = (a + b) / 2
    return xmin, f(xmin), puntos

def graficar_biseccion(f, x_range, puntos, xmin, titulo):
    # ... (código de graficación que devuelve la figura)
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = np.vectorize(f)(x_vals)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, y_vals, 'b-', label=f'f(x) = {titulo}', linewidth=2)
    if puntos:
        p_x1, p_x2 = zip(*puntos)
        ax.plot(p_x1, np.vectorize(f)(p_x1), 'ro', label='Puntos x1', alpha=0.5)
        ax.plot(p_x2, np.vectorize(f)(p_x2), 'mx', label='Puntos x2', alpha=0.5)
    ax.plot(xmin, f(xmin), 'g*', markersize=15, label=f'Mínimo en x ≈ {xmin:.4f}')
    ax.set_title(f"Método de Bisección para '{titulo}'")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.grid(True); ax.legend()
    return fig