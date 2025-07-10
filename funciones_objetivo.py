import numpy as np


def f1_func(x):
    if x <= 0: return float('inf')
    return 2 * np.pi * x**2 + 500 / x
def f1_deriv(x):
    if x <= 0: return float('inf')
    return 4 * np.pi * x - 500 / x**2
def f1_deriv2(x):
    if x <= 0: return float('inf')
    return 4 * np.pi + 1000 / x**3

def f2_func(x):
    return -(200*x - 60*x**2 + 4*x**3)
def f2_deriv(x):
    return -(200 - 120*x + 12*x**2)
def f2_deriv2(x):
    return -(-120 + 24*x)

def f3_func(x):
    if x == 0: return float('inf')
    return x**2 + 54 / x
def f3_deriv(x):
    if x == 0: return float('inf')
    return 2*x - 54 / x**2
def f3_deriv2(x):
    if x == 0: return float('inf')
    return 2 + 108 / x**3

def f4_func(x):
    return x**3 + 2*x - 3
def f4_deriv(x):
    return 3*x**2 + 2
def f4_deriv2(x):
    return 6*x

def f5_func(x):
    return x**4 + x**2 - 33
def f5_deriv(x):
    return 4*x**3 + 2*x
def f5_deriv2(x):
    return 12*x**2 + 2

def f6_func(x):
    return 3*x**4 - 8*x**3 - 6*x**2 + 12*x
def f6_deriv(x):
    return 12*x**3 - 24*x**2 - 12*x + 12
def f6_deriv2(x):
    return 36*x**2 - 48*x - 12


FUNCIONES = {
    "Área de Lata": {
        "func": f1_func, "deriv": f1_deriv, "deriv2": f1_deriv2, "latex": r"f(x) = 2\pi x^2 + \frac{500}{x}"
    },
    "Volumen de Caja (Maximizar)": {
        "func": f2_func, "deriv": f2_deriv, "deriv2": f2_deriv2, "latex": r"f(x) = -(200x - 60x^2 + 4x^3)"
    },
    "f(x) = x² + 54/x": {
        "func": f3_func, "deriv": f3_deriv, "deriv2": f3_deriv2, "latex": r"f(x) = x^2 + \frac{54}{x}"
    },
    "f(x) = x³ + 2x - 3": {
        "func": f4_func, "deriv": f4_deriv, "deriv2": f4_deriv2, "latex": r"f(x) = x^3 + 2x - 3"
    },
    "f(x) = x⁴ + x² - 33": {
        "func": f5_func, "deriv": f5_deriv, "deriv2": f5_deriv2, "latex": r"f(x) = x^4 + x^2 - 33"
    },
    "f(x) = 3x⁴ - 8x³ - 6x² + 12x": {
        "func": f6_func, "deriv": f6_deriv, "deriv2": f6_deriv2, "latex": r"f(x) = 3x^4 - 8x^3 - 6x^2 + 12x"
    }
}