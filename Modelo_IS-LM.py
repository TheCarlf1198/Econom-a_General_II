import sympy as sp
from sympy import symbols, Eq, solve, latex, expand
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Modelo IS-LM")

st.markdown("### **La curva IS (Investment - Savings)**")

st.markdown("#### **1. Variables endógenas**")

st.markdown(r"""
$Y$   : ingreso  
$C$   : consumo  
$Y_d$ : ingreso disponible  
$Imp$ : impuestos  
$I$   : inversión  
$G$   : gasto público  
$r$   : tasa de interés
""")

st.markdown("#### **2. Variables exógenas y parámetros**")

st.markdown(r"""
$C_0$ : consumo autónomo  
$c_1$ : propensión marginal al consumo  
$C_2$ : sensibilidad del consumo a la tasa de interés  
$I_0$ : inversión autónoma  
$I_1$ : sensibilidad de la inversión a la tasa de interés  
$G_0$ : gasto autónomo  
$G_1$ : sensibilidad del gasto público al ingreso  
$T$   : impuesto fijo  
$t$   : tasa de impuesto a la renta
""")

# Definición de las variables endógenas y exógenas
Y, C, Yd, Imp, I, G, r = symbols('Y C Yd Imp I G r') # Variables endógenas
C0, c1, C2, I0, I1, G0, G1, T, t = symbols('C0 c1 C2 I0 I1 G0 G1 T t') # Variables exógenas

st.markdown("#### **3. Ecuaciones**")

# Definición de las ecuaciones que resuelven el modelo
eq_Y = Eq(Y, C + I + G)
eq_C = Eq(C, C0 + c1*Yd + C2*r)
eq_Yd = Eq(Yd, Y - T - t*Y)
eq_Imp = Eq(Imp, T + t*Y)
eq_I = Eq(I, I0 + I1*r)
eq_G = Eq(G, G0 + G1*Y)

modelo = [eq_Y, eq_C, eq_Yd, eq_Imp, eq_I, eq_G]

st.markdown("**1. Condición de equilibrio en el mercado de bienes:**")
st.latex(r"Y = C + I + G")

st.markdown("**2. Función del consumo:**")
st.latex(r"C = C_0 + c_1 Y_d + C_2 r")

st.markdown("**3. Función del ingreso disponible:**")
st.latex(r"Y_d = Y - T - tY")

st.markdown("**4. Impuestos:**")
st.latex(r"Imp = T + tY")

st.markdown("**5. Función de la inversión:**")
st.latex(r"I = I_0 + I_1 r")

st.markdown("**6. Función del gasto público:**")
st.latex(r"G = G_0 + G_1 Y")

#Solución paramétrica del modelo
solucion = solve(modelo, [Y, C, Yd, Imp, I, G])
IS_expr = solucion[Y]

st.markdown("#### **4. Calibración**")

col1, col2, col3 = st.columns(3)

with col1:
    C0_v = st.number_input(r"$C_0$", value=100.0)
    C2_v = st.number_input(r"$C_2$", value=0.0)
    I0_v = st.number_input(r"$I_0$", value=25.0)

with col2: 
    I1_v = st.number_input(r"$I_1$", value=-1.0)
    T_v  = st.number_input(r"$T$", value=50.0)
    G0_v = st.number_input(r"$G_0$", value=45.0)
    
with col3:
    G1_v = st.number_input(r"$G_1$", value=0.0)
    t_v  = st.slider(r"$t$", 0.0, 1.0, 0.0)
    c1_v = st.slider(r"$c_1$", 0.0, 1.0, 0.5)

# Almacena los valores atribuidos
datos_1 = {
    C0: C0_v, c1: c1_v, C2: C2_v,
    I0: I0_v, I1: I1_v,
    T: T_v, t: t_v,
    G0: G0_v, G1: G1_v
}

st.markdown("Reemplazando los valores en las ecuaciones anteriores:")

st.latex(latex(eq_C.subs(datos_1)))
st.latex(latex(eq_Yd.subs(datos_1)))
st.latex(latex(eq_Imp.subs(datos_1)))
st.latex(latex(eq_I.subs(datos_1)))
st.latex(latex(eq_G.subs(datos_1)))

C_total = eq_C.subs(
    Yd,
    eq_Yd.rhs.subs(Imp, eq_Imp.rhs)
)

st.markdown("Lo cual se reduce a:")

st.latex(latex(C_total.subs(datos_1)))
st.latex(latex(eq_I.subs(datos_1)))
st.latex(latex(eq_G.subs(datos_1)))

st.markdown("#### **5. La IS**")

st.markdown("Reemplazando las ecuaciones en la condición de equilibrio en el mercado de bienes y operando se obtiene la IS:")

Y_bienes = eq_Y.subs({
    C: eq_C.rhs.subs(Yd, eq_Yd.rhs),
    I: eq_I.rhs,
    G: eq_G.rhs
})

st.latex(latex(Y_bienes.subs(datos_1)))


# Ecuación simbólica para mostrarla en LaTeX
y_is = Eq(Y, IS_expr.subs(datos_1))
st.latex(latex(y_is))

# Expresión funcional
y_is = IS_expr.subs(datos_1)

st.header("La curva LM (Liquidity - Money)")

st.subheader("1. Plantear variables endógenas")

st.markdown(r"""
- $Y$: ingreso
- $r$: tasa de interés
""")

st.subheader("2. Plantear variables exógenas y parámetros")

M, P, a, b, c = symbols('M P a b c')

st.markdown(r"""
- $M$: oferta de dinero
- $P$: nivel de precios
- $a$: demanda autonóma de dinero
- $b$: sensibilidad de la demanda de dinero al ingreso
- $c$: sensibilidad de la demanda de dinero a la tasa de interés
""")

LM_expr = (M/P - a - c*r)/b

st.subheader("3. Definición del equilibrio monetario")

st.latex(r"\frac{M}{P} = a + bY + cr")

st.subheader("4. Calibración")

M_v = st.number_input("M", value=100.0)
P_v = st.number_input("P", value=1.0)
a_v = st.number_input("a", value=0.0)
b_v = st.number_input("b", value=0.5)
c_v = st.number_input("c", value=-2.0)

datos_2 = {
    M: M_v, P: P_v,
    a: a_v, b: b_v, c: c_v
}

st.subheader("5. Equilibrio ($L^d=M$)")

y_lm = Eq(Y, LM_expr.subs(datos_2))
st.latex(latex(y_lm))
y_lm = LM_expr.subs(datos_2)

st.header("Equilibrio general")

eq_IS_LM = Eq(y_is, y_lm)
st.latex(latex(eq_IS_LM))

r_e = solve(eq_IS_LM, r)[0]
st.latex(latex(r_e))

Y_e = y_is.subs(r, r_e)
st.latex(latex(Y_e))

st.header("Gráficamente")

Y_r = np.linspace(0, float(Y_e)*2, 100)

LM_r = (-datos_2[b]*Y_r + datos_2[M]/datos_2[P] + datos_2[a]) / datos_2[c]

A = float(datos_1[C0] + datos_1[I0] + datos_1[G0] - datos_1[T]*datos_1[c1])
m = float(1 + datos_1[t]*datos_1[c1] - datos_1[c1])
IS_r = A - m*Y_r

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(Y_r, LM_r, label="LM", color="red")
ax.plot(Y_r, IS_r, label="IS", color="blue")
ax.set_xlabel("Y")
ax.set_ylabel("r")
ax.grid(True)
ax.legend()
st.pyplot(fig)

activar_shock_IS = st.checkbox("Aplicar shocks en IS")

if activar_shock_IS:
    params_IS = st.multiselect("Selecciona parámetros IS", ["C0","I0","G0","T","c1","t"])
    datos_shock = datos_1.copy()
    for p in params_IS:
        v = st.number_input(f"Shock en {p}", value=0.0, key=f"shock_IS_{p}")
        if p == "C0": datos_shock[C0] += v
        elif p == "I0": datos_shock[I0] += v
        elif p == "G0": datos_shock[G0] += v
        elif p == "T": datos_shock[T] += v
        elif p == "c1": datos_shock[c1] += v
        elif p == "t": datos_shock[t] += v

    A_s = float(datos_shock[C0] + datos_shock[I0] + datos_shock[G0] - datos_shock[T]*datos_shock[c1])
    m_s = float(1 + datos_shock[t]*datos_shock[c1] - datos_shock[c1])
    IS_r_shocked = A_s - m_s*Y_r

activar_shock_LM = st.checkbox("Aplicar shocks en LM")

if activar_shock_LM:
    params_LM = st.multiselect("Selecciona parámetros LM", ["M","P","a","b","c"])
    datos_LM_shock = datos_2.copy()
    for p in params_LM:
        v = st.number_input(f"Shock en {p}", value=0.0, key=f"shock_LM_{p}")
        if p == "M": datos_LM_shock[M] += v
        elif p == "P": datos_LM_shock[P] += v
        elif p == "a": datos_LM_shock[a] += v
        elif p == "b": datos_LM_shock[b] += v
        elif p == "c": datos_LM_shock[c] += v

    LM_r_shocked = (
        -datos_LM_shock[b]*Y_r
        + datos_LM_shock[M]/datos_LM_shock[P]
        + datos_LM_shock[a]
    ) / datos_LM_shock[c]

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(Y_r, IS_r, label="IS original", color="blue")
ax.plot(Y_r, LM_r, label="LM original", color="red")

if activar_shock_IS:
    ax.plot(Y_r, IS_r_shocked, label="IS con shock", linestyle="--")

if activar_shock_LM:
    ax.plot(Y_r, LM_r_shocked, label="LM con shock", linestyle="--")

ax.set_xlabel("Y")
ax.set_ylabel("r")
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.header("Ecuaciones luego de los shocks")

if activar_shock_IS:
    st.latex(latex(Eq(Y, IS_expr.subs(datos_shock))))

if activar_shock_LM:
    st.latex(latex(Eq(Y, LM_expr.subs(datos_LM_shock))))

st.header("Equilibrio general")

eq_IS_LM_2 = Eq(IS_expr.subs(datos_shock), LM_expr.subs(datos_LM_shock))
st.latex(latex(eq_IS_LM_2))

r_e_2 = solve(eq_IS_LM_2, r)[0]
st.latex(latex(r_e_2))

Y_e_2 = y_lm.subs(r, r_e_2)
st.latex(latex(Y_e_2))



