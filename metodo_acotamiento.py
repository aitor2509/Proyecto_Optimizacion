import numpy as np
import matplotlib.pyplot as plt

def bounding_phase_method(func, a, b, delta):
    x0 = (a + b) / 2
    points = [x0]
    f0 = func(x0)
    
    if func(x0 - delta) >= f0 <= func(x0 + delta):
        return [x0 - delta, x0 + delta], points
    elif func(x0 - delta) < f0:
        direction = -1
    else:
        direction = 1
        
    k = 1
    x_prev = x0
    x_curr = x0 + direction * delta
    points.append(x_curr)

    while func(x_curr) < func(x_prev):
        x_prev = x_curr
        x_curr = x_prev + direction * (2**k) * delta
        if x_curr < a or x_curr > b: # Si nos salimos del dominio, detenemos
            return (x_prev, b) if direction == 1 else (a, x_prev), points
        points.append(x_curr)
        k += 1
    
    if direction == 1:
        return (x0, x_curr), points
    else:
        return (x_curr, x0), points


def plot_acotamiento(func, a, b, points, intervalo, titulo):
    x = np.linspace(a, b, 1000)
    y = np.vectorize(func)(x)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, label=f'f(x) = {titulo}', color='blue')
    
    y_points = [func(p) for p in points]
    ax.scatter(points, y_points, color='red', label='Puntos Visitados', zorder=5)
    
    ax.axvspan(intervalo[0], intervalo[1], alpha=0.3, color='green', label=f'Intervalo Final: [{intervalo[0]:.3f}, {intervalo[1]:.3f}]')

    ax.set_title(f"Fase de Acotamiento para '{titulo}'")
    ax.set_xlabel('x'); ax.set_ylabel('f(x)'); ax.legend(); ax.grid(True)
    return fig