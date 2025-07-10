import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optimización Numérica", layout="wide")

import funciones_objetivo as fo_uni
import funciones_objetivo_multivariable as fo_multi
import metodo_exhaustiva, metodo_biseccion, metodo_golden, metodo_fibonacci, metodo_newton
import metodo_secante, metodo_intervalo_mitad, metodo_acotamiento
import metodo_hooke_jeeves, metodo_nelder_mead, metodo_cauchy, metodo_newton_multi
import metodo_hill_climbing_multi, metodo_recocido_simulado_multi, metodo_caminata_aleatoria_multi
import metodo_busqueda_unidireccional

METODOS_UNI = {
    "Búsqueda Exhaustiva": (metodo_exhaustiva.busqueda_exhaustiva, metodo_exhaustiva.graficar_exhaustiva),
    "Fase de Acotamiento": (metodo_acotamiento.bounding_phase_method, metodo_acotamiento.plot_acotamiento),
    "Intervalo Mitad": (metodo_intervalo_mitad.intervalo_mitad, metodo_intervalo_mitad.plot_intervalo_mitad),
    "Bisección": (metodo_biseccion.biseccion_min, metodo_biseccion.graficar_biseccion),
    "Sección Dorada": (metodo_golden.golden_section_search, metodo_golden.plot_golden),
    "Fibonacci": (metodo_fibonacci.fibonacci_search, metodo_fibonacci.plot_fibonacci),
    "Newton-Raphson": (metodo_newton.newton_raphson_optimization, metodo_newton.plot_newton),
    "Secante": (metodo_secante.secante_min, metodo_secante.graficar_secante),
}
METODOS_MULTI = {
    "Búsqueda Unidireccional": (metodo_busqueda_unidireccional.busqueda_unidireccional, metodo_busqueda_unidireccional.plot_busqueda_unidireccional),
    "Hooke-Jeeves": (metodo_hooke_jeeves.hooke_jeeves, metodo_hooke_jeeves.plot_contour),
    "Nelder-Mead": (metodo_nelder_mead.nelder_mead, metodo_nelder_mead.plot_contour),
    "Cauchy (Descenso de Gradiente)": (metodo_cauchy.cauchy, metodo_cauchy.plot_contour),
    "Newton": (metodo_newton_multi.newton_multi, metodo_newton_multi.plot_contour),
    "Hill Climbing": (metodo_hill_climbing_multi.hill_climbing_multi, metodo_hill_climbing_multi.plot_contour),
    "Recocido Simulado": (metodo_recocido_simulado_multi.recocido_simulado_multi, metodo_recocido_simulado_multi.plot_contour),
    "Caminata Aleatoria": (metodo_caminata_aleatoria_multi.caminata_aleatoria_multi, metodo_caminata_aleatoria_multi.plot_contour),
}
DEFINICIONES_METODOS = {
    "Búsqueda Exhaustiva": "Divide un intervalo en múltiples sub-intervalos para encontrar de forma directa la región que contiene el mínimo. Es simple pero computacionalmente intensivo.",
    "Fase de Acotamiento": "Encuentra rápidamente un intervalo inicial [a, b] que garantiza contener un mínimo. Es un paso preparatorio para otros métodos más precisos.",
    "Intervalo Mitad": "Método de reducción de intervalo que lo divide en cuatro secciones, eliminando partes en cada iteración para acotar el mínimo.",
    "Bisección": "Adaptación del método de búsqueda de raíces que reduce el intervalo de incertidumbre evaluando la función en puntos intermedios.",
    "Sección Dorada": "Eficiente método de reducción de intervalo que utiliza la proporción áurea para converger hacia el mínimo, reutilizando evaluaciones de la función.",
    "Fibonacci": "Similar a la Sección Dorada, pero utiliza la secuencia de Fibonacci para determinar los puntos de prueba. Teóricamente, es el más eficiente para un número fijo de evaluaciones.",
    "Newton-Raphson": "Potente método que utiliza la primera y segunda derivada para encontrar el mínimo de forma cuadrática. Es muy rápido cerca de la solución.",
    "Secante": "Alternativa al método de Newton que aproxima la segunda derivada mediante diferencias finitas de la primera. Útil cuando la segunda derivada no está disponible.",
    "Búsqueda Unidireccional": "Este método no es un optimizador completo, sino un componente (búsqueda de línea). Dado un punto y una dirección, encuentra el mejor tamaño de paso (alpha) a lo largo de esa línea recta.",
    "Hooke-Jeeves": "Método de búsqueda directa (sin derivadas) que combina movimientos exploratorios en cada eje con 'saltos' o movimientos de patrón para acelerar la convergencia.",
    "Nelder-Mead": "Algoritmo de búsqueda directa que utiliza una figura geométrica (simplex) que se desplaza, expande y contrae en el espacio para encontrar el mínimo.",
    "Cauchy (Descenso de Gradiente)": "Algoritmo fundamental de primer orden. Se mueve iterativamente en la dirección opuesta al gradiente, que es la dirección de máximo descenso de la función.",
    "Newton": "Extensión multivariable del método de Newton. Utiliza el gradiente y la matriz Hessiana (de segundas derivadas) para una convergencia muy rápida cerca del mínimo.",
    "Hill Climbing": "Heurística estocástica simple. Comienza en un punto y se mueve a un vecino si este es mejor. Es rápido pero puede quedar atrapado en mínimos locales.",
    "Recocido Simulado": "Método estocástico avanzado que permite 'malos' movimientos con una probabilidad decreciente (controlada por una 'temperatura') para escapar de mínimos locales.",
    "Caminata Aleatoria": "Método estocástico básico que explora el espacio de búsqueda de forma aleatoria, guardando la mejor solución encontrada hasta el momento."
}


st.title("Proyecto Final de Optimización")
st.markdown("""
La **optimización** es el proceso de encontrar la mejor solución a un problema a partir de un conjunto de posibles soluciones.
En este contexto, buscamos minimizar el valor de una función objetivo ajustando sus variables de entrada mediante diversos algoritmos numéricos.
""")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuración General")
    tipo_optimizacion = st.radio("Elige el tipo de problema:", ("Univariable", "Multivariable"), key="tipo_opt")

    if tipo_optimizacion == "Univariable":
        st.header("Optimización Univariable")
        metodo_uni_sel = st.selectbox("Selecciona un método:", list(METODOS_UNI.keys()))
        func_uni_sel = st.selectbox("Selecciona una función:", list(fo_uni.FUNCIONES.keys()))
        func_info_uni = fo_uni.FUNCIONES[func_uni_sel]
        st.latex(func_info_uni["latex"])
        st.subheader("Parámetros")
        if "Búsqueda Exhaustiva" in metodo_uni_sel or "Acotamiento" in metodo_uni_sel or "Intervalo Mitad" in metodo_uni_sel or "Sección Dorada" in metodo_uni_sel or "Fibonacci" in metodo_uni_sel:
            col1, col2 = st.columns(2)
            a = col1.number_input("Límite inferior (a)", value=-5.0)
            b = col2.number_input("Límite superior (b)", value=5.0)
        if "Búsqueda Exhaustiva" in metodo_uni_sel:
            n_puntos = st.number_input("Número de puntos (n)", value=100, min_value=10)
        if "Acotamiento" in metodo_uni_sel:
            delta_uni = st.number_input("Paso inicial (δ)", value=0.1, format="%.3f")
        if "Bisección" in metodo_uni_sel:
             x0_bi = st.number_input("Punto inicial (x₀)", value=1.0)
        if "Newton-Raphson" in metodo_uni_sel or "Secante" in metodo_uni_sel:
            x0_uni = st.number_input("Punto inicial (x₀)", value=2.0)
        if "Secante" in metodo_uni_sel:
            x1_sec = st.number_input("Punto inicial 2 (x₁)", value=2.1)
        if "Bisección" in metodo_uni_sel or "Sección Dorada" in metodo_uni_sel or "Fibonacci" in metodo_uni_sel or "Newton-Raphson" in metodo_uni_sel or "Secante" in metodo_uni_sel or "Intervalo Mitad" in metodo_uni_sel:
            epsilon_uni = st.number_input("Precisión (ε)", value=0.001, format="%.5f")

    elif tipo_optimizacion == "Multivariable":
        st.header("Optimización Multivariable")
        metodo_multi_sel = st.selectbox("Selecciona un método:", list(METODOS_MULTI.keys()))
        func_multi_sel = st.selectbox("Selecciona una función:", list(fo_multi.FUNCIONES_MULTI.keys()))
        func_info_multi = fo_multi.FUNCIONES_MULTI[func_multi_sel]
        st.latex(func_info_multi["latex"])
        st.subheader("Parámetros")
        x0_default = ", ".join(map(str, func_info_multi["x0"]))
        x0_str = st.text_input("Punto de inicio (x_t)", value=x0_default)

        if metodo_multi_sel == "Búsqueda Unidireccional":
            s_t_str = st.text_input("Vector de Dirección (s_t)", value="-1.0, -1.0")
            paso_alpha = st.number_input("Paso de Alpha", value=0.1, format="%.3f")
            max_iter_multi = st.number_input("Max. Iteraciones", value=50, key="mimb")
        elif "Cauchy" in metodo_multi_sel or "Newton" in metodo_multi_sel:
            col1, col2 = st.columns(2)
            epsilon1 = col1.number_input("Epsilon 1 (grad)", value=1e-5, format="%.5f", key="e1m")
            epsilon2 = col2.number_input("Epsilon 2 (paso)", value=1e-5, format="%.5f", key="e2m")
            max_iter_multi = st.number_input("Max. Iteraciones", value=100, key="mim")
        elif "Hill Climbing" in metodo_multi_sel or "Caminata Aleatoria" in metodo_multi_sel:
            step_size = st.number_input("Tamaño de paso", value=0.1, format="%.3f")
            max_iter_multi = st.number_input("Max. Iteraciones", value=1000, key="mims")
        elif "Recocido Simulado" in metodo_multi_sel:
            temp_inicial = st.number_input("Temperatura Inicial", value=100.0)
            alpha_recocido = st.number_input("Factor de Enfriamiento (α)", value=0.99)
            max_iter_multi = st.number_input("Max. Iteraciones", value=1000, key="mima")
        else: # Hooke-Jeeves y Nelder-Mead
            epsilon_multi = st.number_input("Epsilon", value=1e-5, format="%.5f", key="em")



if tipo_optimizacion == "Univariable":
    st.header(f"Resultados para: {metodo_uni_sel}")
    st.info(DEFINICIONES_METODOS[metodo_uni_sel])
    if st.button(f"Ejecutar {metodo_uni_sel}", type="primary"):
        algoritmo_uni, plotter_uni = METODOS_UNI[metodo_uni_sel]
        func_obj_uni = func_info_uni["func"]
        with st.spinner("Calculando..."):
            try:
                if metodo_uni_sel == "Búsqueda Exhaustiva":
                    resultado, _ = algoritmo_uni(a, b, n_puntos, func_obj_uni)
                    st.metric("Intervalo del Mínimo", f"[{resultado[0]:.5f}, {resultado[1]:.5f}]")
                    fig = plotter_uni(a, b, func_obj_uni, resultado, func_uni_sel)
                elif metodo_uni_sel == "Fase de Acotamiento":
                    intervalo, puntos = algoritmo_uni(func_obj_uni, a, b, delta_uni)
                    st.metric("Intervalo de Incertidumbre", f"[{intervalo[0]:.5f}, {intervalo[1]:.5f}]")
                    fig = plotter_uni(func_obj_uni, a, b, puntos, intervalo, func_uni_sel)
                elif metodo_uni_sel == "Bisección":
                    a_b, b_b = metodo_biseccion.bounding_phase(func_obj_uni, x0=x0_bi)
                    xmin, ymin, puntos = algoritmo_uni(func_obj_uni, a_b, b_b, epsilon=epsilon_uni)
                    st.metric("Mínimo en x", f"{xmin:.6f}"); st.metric("Valor f(x)", f"{ymin:.6f}")
                    fig = plotter_uni(func_obj_uni, (a_b, b_b), puntos, xmin, func_uni_sel)
                elif metodo_uni_sel == "Newton-Raphson":
                    xmin, pasos = algoritmo_uni(func_info_uni["deriv"], func_info_uni["deriv2"], x0_uni, epsilon_uni)
                    st.metric("Mínimo en x", f"{xmin:.6f}"); st.metric("Valor f(x)", f"{func_obj_uni(xmin):.6f}")
                    fig = plotter_uni(func_obj_uni, xmin, pasos, func_uni_sel)
                elif metodo_uni_sel == "Secante":
                    xmin, puntos = algoritmo_uni(func_obj_uni, x0_uni, x1_sec, epsilon_uni)
                    st.metric("Mínimo en x", f"{xmin:.6f}"); st.metric("Valor f(x)", f"{func_obj_uni(xmin):.6f}")
                    fig = plotter_uni(func_obj_uni, (min(puntos)-1, max(puntos)+1), puntos, xmin, func_uni_sel)
                else: # Golden, Fibonacci, Intervalo Mitad
                    xmin, fmin, iterations, *_ = algoritmo_uni(func_obj_uni, a, b, tol=epsilon_uni)
                    st.metric("Mínimo en x", f"{xmin:.6f}"); st.metric("Valor f(x)", f"{fmin:.6f}")
                    fig = plotter_uni(func_obj_uni, xmin, fmin, iterations, a, b, func_uni_sel)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

elif tipo_optimizacion == "Multivariable":
    st.header(f"Resultados para: {metodo_multi_sel}")
    st.info(DEFINICIONES_METODOS[metodo_multi_sel])
    
    if st.button(f"Ejecutar {metodo_multi_sel}", type="primary"):
        algoritmo_multi, plotter_multi = METODOS_MULTI[metodo_multi_sel]
        func_obj_multi = func_info_multi["func"]
        
        try:
            x0 = np.array([float(x.strip()) for x in x0_str.split(',')])

            with st.spinner("Optimizando... por favor espera."):
                if metodo_multi_sel == "Búsqueda Unidireccional":
                    s_t = np.array([float(s.strip()) for s in s_t_str.split(',')])
                    solucion, historial, alpha_opt = algoritmo_multi(func_obj_multi, x0, s_t, paso_alpha, max_iter_multi)
                    
                    st.success("¡Búsqueda completada!")
                    col1, col2 = st.columns(2)
                    sol_str = ", ".join([f"{val:.4f}" for val in solucion])
                    col1.metric("Punto Óptimo Encontrado (x*)", f"[{sol_str}]")
                    col2.metric("Alpha Óptimo (α*)", f"{alpha_opt:.4f}")
                    
                    fig = plotter_multi(func_obj_multi, x0, s_t, historial, alpha_opt, solucion)
                    st.pyplot(fig)

                else:
                    if "Cauchy" in metodo_multi_sel or "Newton" in metodo_multi_sel:
                        solucion, historial = algoritmo_multi(func_obj_multi, x0, epsilon1, epsilon2, max_iter_multi)
                    elif "Hill Climbing" in metodo_multi_sel or "Caminata Aleatoria" in metodo_multi_sel:
                        solucion, historial = algoritmo_multi(func_obj_multi, x0, max_iter_multi, step_size)
                    elif "Recocido Simulado" in metodo_multi_sel:
                        solucion, historial = algoritmo_multi(func_obj_multi, x0, temp_inicial, alpha_recocido, max_iter_multi)
                    elif "Nelder-Mead" in metodo_multi_sel:
                        solucion, historial = algoritmo_multi(func_obj_multi, x0, epsilon=epsilon_multi)
                    else: # Hooke-Jeeves
                        delta_hj = np.full(len(x0), 0.5); alpha_hj = 2.0
                        solucion, historial = algoritmo_multi(func_obj_multi, x0, delta_hj, alpha_hj, epsilon_multi)

                    st.success("¡Optimización completada!")
                    col1, col2 = st.columns(2)
                    sol_str = ", ".join([f"{val:.4f}" for val in solucion])
                    col1.metric("Solución encontrada (x*)", f"[{sol_str}]")
                    col2.metric("Valor de la función f(x*)", f"{func_obj_multi(solucion):.6f}")

                    fig, ax = plt.subplots(figsize=(10, 8))
                    hist_array = np.array(historial)
                    x_min_p = min(hist_array[:, 0].min(), solucion[0]) - 1
                    x_max_p = max(hist_array[:, 0].max(), solucion[0]) + 1
                    y_min_p = min(hist_array[:, 1].min(), solucion[1]) - 1
                    y_max_p = max(hist_array[:, 1].max(), solucion[1]) + 1
                    xx, yy = np.meshgrid(np.linspace(x_min_p, x_max_p, 150), np.linspace(y_min_p, y_max_p, 150))
                    Z = np.zeros_like(xx)
                    for i in range(xx.shape[0]):
                        for j in range(xx.shape[1]):
                            Z[i, j] = func_obj_multi(np.array([xx[i, j], yy[i, j]]))

                    ax.contourf(xx, yy, Z, levels=50, cmap='viridis', alpha=0.8)
                    ax.plot(hist_array[:, 0], hist_array[:, 1], 'r-o', markersize=3, label='Camino')
                    ax.plot(hist_array[0, 0], hist_array[0, 1], 'go', markersize=10, label='Inicio')
                    ax.plot(hist_array[-1, 0], hist_array[-1, 1], 'y*', markersize=15, label='Fin')
                    ax.set_title(f"{metodo_multi_sel} para {func_multi_sel}"); ax.set_xlabel('x1'); ax.set_ylabel('x2')
                    ax.legend(); ax.grid(True)
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Ocurrió un error durante la ejecución: {e}")