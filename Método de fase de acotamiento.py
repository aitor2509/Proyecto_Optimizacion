import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.pi * x**2 + 500 / x

def f2(x):
    return -4 * x**3 + 60 * x**2 - 200 * x

def f3(x):
    return x**2 + 54 / x

def f4(x):
    return x**3 + 2 * x - 3

def f5(x):
    return x**4 + x**2 - 33

def f6(x):
    return 3 * x**4 - 8 * x**3 - 6 * x**2 + 12 * x

functions = {
    'Can (Lata)': (f1, 0.1, 10),
    'Box (Caja)': (f2, 2, 3),
    'f3(x) = x^2 + 54/x': (f3, 0.1, 10),
    'f4(x) = x^3 + 2x - 3': (f4, 0, 5),
    'f5(x) = x^4 + x^2 - 33': (f5, -2.5, 2.5),
    'f6(x) = 3x^4 - 8x^3 - 6x^2 + 12x': (f6, -1.5, 3)
}

def bounding_phase_method(func, a, b, delta, precision):
    x0 = (a + b) / 2
    points = [x0]
    
    f0 = func(x0)
    
    x_left = max(a, x0 - delta)
    x_right = min(b, x0 + delta)
    f_left = func(x_left)
    f_right = func(x_right)
    points.extend([x_left, x_right])
    
    if f_left < f0 and f_left <= f_right:
        direction = -1
        x = x_left
        fx = f_left
    elif f_right < f0:
        direction = 1
        x = x_right
        fx = f_right
    else:
        return [max(a, x0 - delta), min(b, x0 + delta)], points
    
    k = 1
    while True:
        x_new = x + direction * (2**k * delta)
        if x_new < a or x_new > b:
            break
        f_new = func(x_new)
        points.append(x_new)
        
        if f_new > fx:
            if direction == 1:
                return [x - 2**(k-1) * delta, x_new], points
            else:
                return [x_new, x + 2**(k-1) * delta], points
        x = x_new
        fx = f_new
        k += 1
    
    if direction == 1:
        return [x - 2**(k-1) * delta, b], points
    else:
        return [a, x + 2**(k-1) * delta], points

def plot_function_with_points(func, a, b, points, title, delta):
    x = np.linspace(a, b, 1000)
    y = [func(xi) for xi in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Function', color='blue')
    
    y_points = [func(p) for p in points]
    plt.scatter(points, y_points, color='red', label='Points Visited', zorder=5)
    for i, (xp, yp) in enumerate(zip(points, y_points)):
        plt.text(xp, yp, f'{i}', fontsize=12, color='black', verticalalignment='bottom')
    
    plt.title(f"{title} (delta={delta})")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

precisions = [0.5, 0.1, 0.01, 0.001]
for name, (func, a, b) in functions.items():
    print(f"\n=== Optimizing {name} ===")
    for precision in precisions:
        delta = precision
        interval, points = bounding_phase_method(func, a, b, delta, precision)
        print(f"Precision {precision}: Interval = {interval}, Points Visited = {points}")
        plot_function_with_points(func, a, b, points, name, delta)