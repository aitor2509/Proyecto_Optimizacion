import numpy as np
import matplotlib.pyplot as plt
import os

# Create a directory to save the graphs
save_dir = "optimization_graphs"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Definición de las funciones
def func1(x):  # Función de la Lata
    r = x
    h = x
    return 2 * np.pi * r**2 + 2 * np.pi * r * h

def func2(x):  # Función de la Caja
    return 200*x - 60*x**2 + 4*x**3

def func3(x):  # f(x) = x^3+54/x
    return x**3 + 54/x

def func4(x):  # f(x) = x^3 + 2x - 3
    return x**3 + 2*x - 3

def func5(x):  # f(x) = x^4 + x^2 - 33
    return x**4 + x**2 - 33

def func6(x):  # f(x) = 3x^4 - 8x^3 - 6x^2 + 12x
    return 3*x**4 - 8*x**3 - 6*x**2 + 12*x

# Función para el método de intervalo por la mitad
def intervalo_mitad(func, a, b, tol=0.001, max_iter=100):
    iterations = []
    
    # Salvar los límites iniciales
    a_init, b_init = a, b
    
    for i in range(max_iter):
        c = (a + b) / 2
        d = (c + b) / 2
        e = (a + c) / 2
        
        fc = func(c)
        fd = func(d)
        fe = func(e)
        
        iterations.append((a, c, b, fc))
        
        if fd < fc:
            a = c
            c = d
        elif fe < fc:
            b = c
            c = e
        else:
            a = e
            b = d
        
        if abs(b - a) < tol:
            break
    
    # Encontrar el punto mínimo
    x_min = (a + b) / 2
    f_min = func(x_min)
    
    return x_min, f_min, iterations, a_init, b_init

# Función para graficar y guardar resultados
def plot_results(func, x_min, f_min, iterations, a_init, b_init, title, precision, func_name):
    x = np.linspace(a_init, b_init, 1000)
    y = [func(xi) for xi in x]
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'b-')
    plt.plot(x_min, f_min, 'ro', markersize=8)
    
    # Graficar los puntos visitados en cada iteración
    for i, (a, c, b, fc) in enumerate(iterations):
        plt.plot(c, fc, 'go', markersize=6, alpha=0.5)
        plt.text(c, fc, f"{i+1}", fontsize=8)
    
    plt.title(f"{title} - Método de Intervalo por la Mitad (Precisión: {precision})")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    
    filename = f"{save_dir}/{func_name}_precision_{precision}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  
    
    print(f"Gráfica guardada en: {filename}")

def run_all_functions(precision):
    print(f"\nPrecisión: {precision}")
    
    x_min, f_min, iterations, a_init, b_init = intervalo_mitad(func1, 0.1, 10, tol=precision)
    print(f"Función de la Lata - Mínimo: x = {x_min:.6f}, f(x) = {f_min:.6f}, Iteraciones: {len(iterations)}")
    plot_results(func1, x_min, f_min, iterations, a_init, b_init, "Función de la Lata", precision, "lata")
    
    x_min, f_min, iterations, a_init, b_init = intervalo_mitad(func2, 0.1, 5, tol=precision)
    print(f"Función de la Caja - Máximo: x = {x_min:.6f}, f(x) = {f_min:.6f}, Iteraciones: {len(iterations)}")
    plot_results(func2, x_min, f_min, iterations, a_init, b_init, "Función de la Caja", precision, "caja")
    
    x_min, f_min, iterations, a_init, b_init = intervalo_mitad(func3, 0.1, 10, tol=precision)
    print(f"Función 3 - Mínimo: x = {x_min:.6f}, f(x) = {f_min:.6f}, Iteraciones: {len(iterations)}")
    plot_results(func3, x_min, f_min, iterations, a_init, b_init, "f(x) = x^3+54/x", precision, "func3")
    
    x_min, f_min, iterations, a_init, b_init = intervalo_mitad(func4, 0.1, 5, tol=precision)
    print(f"Función 4 - Mínimo: x = {x_min:.6f}, f(x) = {f_min:.6f}, Iteraciones: {len(iterations)}")
    plot_results(func4, x_min, f_min, iterations, a_init, b_init, "f(x) = x^3 + 2x - 3", precision, "func4")
    
    x_min, f_min, iterations, a_init, b_init = intervalo_mitad(func5, -2.5, 2.5, tol=precision)
    print(f"Función 5 - Mínimo: x = {x_min:.6f}, f(x) = {f_min:.6f}, Iteraciones: {len(iterations)}")
    plot_results(func5, x_min, f_min, iterations, a_init, b_init, "f(x) = x^4 + x^2 - 33", precision, "func5")
    
    x_min, f_min, iterations, a_init, b_init = intervalo_mitad(func6, -1.5, 3, tol=precision)
    print(f"Función 6 - Mínimo: x = {x_min:.6f}, f(x) = {f_min:.6f}, Iteraciones: {len(iterations)}")
    plot_results(func6, x_min, f_min, iterations, a_init, b_init, "f(x) = 3x^4 - 8x^3 - 6x^2 + 12x", precision, "func6")

precisions = [0.5, 0.1, 0.01, 0.0001]

for precision in precisions:
    run_all_functions(precision)

print(f"\nTodas las gráficas han sido guardadas en el directorio: {os.path.abspath(save_dir)}")