import numpy as np
import matplotlib.pyplot as plt

def busqueda_unidireccional(func, x_t, s_t, paso_alpha=0.1, max_iter=100):
    """
    Realiza una búsqueda unidireccional para encontrar el mínimo a lo largo de una línea.
    """
    historial = []
    alpha_actual = 0.0
    
    x_actual = x_t + alpha_actual * s_t
    valor_min_objetivo = func(x_actual)
    x_optimo = x_actual.copy()
    alpha_optimo = alpha_actual
    historial.append((alpha_actual, valor_min_objetivo))

    for i in range(max_iter):
        siguiente_alpha = (i + 1) * paso_alpha
        siguiente_x = x_t + siguiente_alpha * s_t
        siguiente_valor = func(siguiente_x)
        historial.append((siguiente_alpha, siguiente_valor))
        if siguiente_valor < valor_min_objetivo:
            valor_min_objetivo = siguiente_valor
            alpha_optimo = siguiente_alpha
            x_optimo = siguiente_x.copy()
        else:
            break
            
    for i in range(max_iter):
        siguiente_alpha = -(i + 1) * paso_alpha
        siguiente_x = x_t + siguiente_alpha * s_t
        siguiente_valor = func(siguiente_x)
        historial.append((siguiente_alpha, siguiente_valor))
        if siguiente_valor < valor_min_objetivo:
            valor_min_objetivo = siguiente_valor
            alpha_optimo = siguiente_alpha
            x_optimo = siguiente_x.copy()
        else:
            break

    historial.sort(key=lambda item: item[0])
    return x_optimo, historial, alpha_optimo

def plot_busqueda_unidireccional(func, x_inicial, s_direccion, historial, alpha_opt, x_opt):
    """
    Genera la gráfica del mapa de contornos.
    Ahora devuelve una sola figura de matplotlib.
    """
    alphas = [h[0] for h in historial]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    puntos_linea_x = [x_inicial[0] + alpha * s_direccion[0] for alpha in alphas]
    puntos_linea_y = [x_inicial[1] + alpha * s_direccion[1] for alpha in alphas]

    x_min, x_max = min(puntos_linea_x) - 1, max(puntos_linea_x) + 1
    y_min, y_max = min(puntos_linea_y) - 1, max(puntos_linea_y) + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    Z = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            Z[i, j] = func(np.array([xx[i, j], yy[i, j]]))

    ax.contourf(xx, yy, Z, levels=50, cmap='viridis', alpha=0.8)
    
    ax.plot(puntos_linea_x, puntos_linea_y, marker='x', linestyle='-', color='white', alpha=0.7, label='Busqueda Unidireccional')
    
    ax.plot(x_inicial[0], x_inicial[1], 'go', markersize=10, label='Punto de Partida')
    ax.plot(x_opt[0], x_opt[1], 'y*', markersize=15, label=f'Punto Óptimo')
    ax.quiver(x_inicial[0], x_inicial[1], s_direccion[0], s_direccion[1], angles='xy', scale_units='xy', scale=1, color='cyan', width=0.007, label='Dirección de Búsqueda')

    ax.set_title('Ruta de Búsqueda en Espacio 2D')
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.axis('equal'); ax.grid(True); ax.legend()

    return fig