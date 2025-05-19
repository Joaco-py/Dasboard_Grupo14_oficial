# dashboard_tarea.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates

# 1. Configuración básica de la página
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.title("📊 Dashboard Supermarket Sales")
# Configuración simple para los gráficos
sns.set_style("whitegrid")

# 2. Carga de datos (con cache para mejorar rendimiento)
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    return df

df = load_data("data.csv")

# 3. Sidebar - filtros interactivos
st.sidebar.header("Filtros")

# Filtro de Sucursal (igual que antes)
branch_opt = ['Todas'] + list(df['Branch'].unique())
branch = st.sidebar.selectbox("Seleccione Sucursal", branch_opt)

# USAR ESTA VERSIÓN:
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

date_range = st.sidebar.date_input(
    "Rango de Fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Convertir las fechas de datetime.date a pd.Timestamp para filtrar
start_date = pd.to_datetime(date_range[0])
end_date   = pd.to_datetime(date_range[1])


# Filtro de Producto
products = st.sidebar.multiselect(
    "Líneas de Producto",
    options=df['Product line'].unique(),
    default=list(df['Product line'].unique())
)

# 4. Aplicar todos los filtros al DataFrame
df_filtered = df.copy()

# Filtrar por sucursal
if branch != 'Todas':
    df_filtered = df_filtered[df_filtered['Branch'] == branch]

# Filtrar por fecha
df_filtered = df_filtered[
    (df_filtered['Date'] >= start_date) &
    (df_filtered['Date'] <= end_date)
]

# Filtrar por producto
df_filtered = df_filtered[df_filtered['Product line'].isin(products)]

# Ahora df_filtered ya contiene la versión filtrada de tus datos
st.write(f"Mostrando {len(df_filtered)} registros desde {start_date.date()} hasta {end_date.date()} en la sucursal “{branch}”")

# #######################################################
# # SECCIÓN DE MÉTRICAS (PRIMERA FILA)
# #######################################################

# KPIs rápidos
st.markdown("---")
st.subheader("📌 KPIs Resumidos")
col1, col2, col3 = st.columns(3)
col1.metric("Ventas Totales", f"${df_filtered['Total'].sum():,.0f}")
col2.metric("Ticket Promedio", f"${df_filtered['Total'].mean():.2f}")
col3.metric("Rating Medio", f"{df_filtered['Rating'].mean():.2f}")

#########################################################
# SECCIÓN DE GRÁFICOS (SEGUNDA FILA)
#########################################################

# Segunda fila: Evolución de las Ventas Totales e Ingresos por Línea de Producto —
col1_f1, col2_f1 = st.columns((2))

# 5. Sección: Evolución de las Ventas Totales

# 5.1 Función para verificación autónoma y parseo de fechas a DD-MM-YYYY (Ajustamos fechas de data.csv)
def parse_date(x):
    if pd.isna(x) or not isinstance(x, str):
        return pd.NaT
    s = x.strip()
    if '-' in s:
        d, m, y = s.split('-')
        return pd.to_datetime(f"{d}-{m}-{y}", format='%d-%m-%Y', errors='coerce')
    if '/' in s:
        a, b, y = s.split('/')
        ia, ib = int(a), int(b)
        if ia > 12:
            return pd.to_datetime(f"{ia}-{ib}-{y}", format='%d-%m-%Y', errors='coerce')
        if ib > 12:
            return pd.to_datetime(f"{ib}-{ia}-{y}", format='%d-%m-%Y', errors='coerce')
        return pd.to_datetime(f"{ia}-{ib}-{y}", format='%d-%m-%Y', errors='coerce')
    return pd.to_datetime(s, infer_datetime_format=True, errors='coerce')

# 5.2. Función para limpiar números con separadores de miles
def clean_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    mask = s.str.count(r'\.') > 1
    cleaned = np.where(mask, s.str.replace('.', '', regex=False), s)
    return pd.to_numeric(cleaned, errors='coerce')
df = pd.read_csv('data.csv', dtype=str)

# 5.3 Parseo de Date a datetime64
df['Date'] = df['Date'].apply(parse_date)

df['Time'] = df['Time'].astype(str)

# 5.4. Columnas categóricas
for col in ['Branch','City','Customer type','Gender','Product line','Payment']:
    df[col] = df[col].astype('category')

# 5.5. Otros tipos básicos
df['Invoice ID'] = df['Invoice ID'].astype(str)
df['Quantity']   = pd.to_numeric(df['Quantity'], downcast='integer', errors='coerce')
for col in ['Unit price','Tax 5%','cogs','Rating']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
for col in ['Total','gross margin percentage','gross income']:
    df[col] = clean_numeric(df[col])

with col1_f1: 
    st.header("1️⃣ Evolución de las Ventas Totales")

    # 5.6 Agrupar por mes
    sales_monthly = (
        df.groupby(pd.Grouper(key='Date', freq='M'))['Total']
        .sum()
        .reset_index()
    )

    fig1, ax = plt.subplots(figsize=(10,7))
    sns.lineplot(
        data=sales_monthly,
        x='Date', y='Total',
        marker='o', color='orange', ax=ax, linewidth=2, markersize=6
    )

    # 5.7 formatear ticks: uno por mes con etiqueta corta
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.tick_params(axis='x', rotation=45)

    ax.set_title('Evolución Mensual de Ventas Totales', weight='bold')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Ventas Totales')
    fig1.tight_layout()

    # 5.8 Mostrar en Streamlit
    st.pyplot(fig1)

# 6. Sección: Ingresos por Línea de Producto

with col2_f1:

    st.header("2️⃣ Ingresos por Línea de Producto")
      # 6.1. Agrupar por línea de producto y sumar 'Total'
    revenue_by_product = (
        df_filtered.groupby('Product line')['Total']
                   .sum()
                   .sort_values(ascending=False)
    )

    # 6.2. Crear figura y eje
    fig2, ax2 = plt.subplots(figsize=(6, 2.7))

    # 6.3. Graficar el barplot en ese eje
    sns.barplot(
        x=revenue_by_product.index,
        y=revenue_by_product.values,
        color='orange',
        edgecolor='black',
        ax=ax2
    )

    # 6.4. Ajustes de estilo
    ax2.set_title('Ingresos Totales por Línea de Producto')
    ax2.set_xlabel('Línea de Producto')
    ax2.set_ylabel('Ingresos Totales')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    # 6.5. Renderizar en Streamlit
    st.pyplot(fig2)

# Tercera fila: Distribución de ratings & Gastos por tipo de cliente —
col1_f2, col2_f2 = st.columns((2))

# 7. Sección: Distribución de Ratings
with col1_f2:
    st.header("3️⃣ Distribución de Calificaciones de Clientes")
    
    # 7.1. Crear figura y eje con un tamaño ajustado
    fig3, ax3 = plt.subplots(figsize=(6, 3))

    # 7.2. Dibujar histograma + KDE sobre ax3
    sns.histplot(
        df_filtered['Rating'],    # usa df_filtered en tu dashboard
        bins=10,
        kde=True,
        color='orange',
        edgecolor='black',
        ax=ax3
    )

    # 7.3. Títulos y etiquetas
    ax3.set_title('Distribución de Calificaciones')
    ax3.set_xlabel('Rating')
    ax3.set_ylabel('Frecuencia')

    # 7.4. Ajustar layout para que no se encimen las etiquetas
    fig3.tight_layout()

    # 7.5. Renderizar en Streamlit
    st.pyplot(fig3)


# 8. Sección: Gasto por Tipo de Cliente (Boxplot)
with col2_f2:
    st.header("4️⃣ Gasto Total por Tipo de Cliente")
    
    # 8.1. Definir figura y eje con un tamaño ajustado
    fig4, ax4 = plt.subplots(figsize=(6, 3))

    # 8.2. Crear el boxplot sobre ax4
    colors = ['#FF69B4', '#1f77b4']  # dos tonos de naranja
    sns.boxplot(
        x='Customer type',
        y='Total',
        data=df_filtered,    # usa el df filtrado para que respete tus filtros
        palette=colors,
        ax=ax4
    )

    # 8.3. Títulos y etiquetas
    ax4.set_title('Gasto Total por Tipo de Cliente')
    ax4.set_xlabel('Tipo de Cliente')
    ax4.set_ylabel('Gasto Total')

    # 8.4. Ajustar el layout
    fig4.tight_layout()

    # 8.5. Renderizar en Streamlit
    st.pyplot(fig4)

# Cuarta fila: Composición del Ingreso Bruto por Sucursal y Línea de Producto & Precio Unitario Vs Rating
col1_f3, col2_f3 = st.columns((2))

# 9. Sección: Distribución de Ratings
with col1_f3:
    st.header("5️⃣ Composición del Ingreso Bruto por sucursal")

    # 9.1. Configurar estilo general
    sns.set_style("whitegrid")                     # fondo con cuadrícula sutil
    plt.rcParams.update({                          # ajustes de fuente y ejes
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # 9.2. Agrupar y pivotar para obtener ingreso bruto por Branch y Product line
    pivot_income = (
        df
        .groupby(['Branch', 'Product line'])['gross income']
        .sum()
        .unstack(fill_value=0)
    )

    # 9.3. Crear figura y eje
    fig5, ax = plt.subplots(figsize=(12, 7))

    # 9.4. Dibujar barras apiladas con paleta suave
    pivot_income.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        edgecolor='white',
        linewidth=0.8,
        colormap='tab20c'    # paleta discreta agradable
    )

    # 9.5. Anotar totales encima de cada barra
    totals = pivot_income.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(
            i,
            total + totals.max() * 0.01,      # un poquito por encima
            f'{total:,.0f}',                  # formateo con separador de miles
            ha='center',
            va='bottom',
            weight='bold'
        )

    # 9.6. Ajustes de etiquetas y leyenda
    ax.set_title('Composición del Ingreso Bruto por Sucursal y Línea de Producto', weight='bold')
    ax.set_xlabel('Sucursal (Branch)')
    ax.set_ylabel('Ingreso Bruto Total')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    leg = ax.legend(
        title='Línea de Producto',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0
    )
    leg.get_title().set_weight('bold')

    # 9.7. Finalizar
    fig5.tight_layout()
    st.pyplot(fig5)

# 10. Sección: Precio Unitario Vs Rating
with col2_f3:
    st.header("6️⃣ Precio Unitario Vs Rating")
    
    # 10.1. Crear figura y eje
    fig6, ax6 = plt.subplots(figsize=(12, 7))
    
    # Scatter plot Precio unitario vs Rating
    sns.scatterplot(data=df, x='Unit price', y='Rating', alpha=0.6)
    
    # 10.2. Ajustes de estilo
    ax6.set_title('Precio Unitario vs Rating', fontsize=18, fontweight='bold')
    ax6.set_xlabel('Precio Unitario')
    ax6.set_ylabel('Rating del Cliente')
    ax6.grid(True, linestyle='--', alpha=0.4)

    
    # 10.3. Mostrar en Streamlit
    fig6.tight_layout()
    st.pyplot(fig6)

# 11. Sección: Superficie 3D interactiva de transacciones por Mes y Hora del día

import calendar
import plotly.graph_objects as go

st.header("7️⃣ Superficie 3D de Transacciones por Mes y Hora")

# 11.1. Preparar datos
df_temp = df.copy()
df_temp['MonthNum'] = df_temp['Date'].dt.month
df_temp['HourInt']  = df_temp['Time'].astype(str).str.split(':').str[0].astype(int)
pivot = df_temp.groupby(['MonthNum','HourInt']).size().unstack(fill_value=0)

months = pivot.index.values
hours  = pivot.columns.values
Z       = pivot.values

# 11.2 Crear superficie 3D interactiva
fig3d = go.Figure(
    data=go.Surface(
        x=months,
        y=hours,
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title='Transacciones')
    )
)

# 11.3 Ajustar ejes, etiquetas y aspecto igualado
fig3d.update_layout(
    title='Transacciones por Mes y Hora del día',
    scene=dict(
        xaxis=dict(
            title='Mes',
            tickmode='array',
            tickvals=list(months),
            ticktext=[calendar.month_abbr[m] for m in months]
        ),
        yaxis=dict(title='Hora del día'),
        zaxis=dict(title='Número de transacciones'),
        aspectmode='cube'    # mantiene misma escala en X, Y y Z
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)

# 11.4. Mostrar en Streamlit
st.plotly_chart(fig3d, use_container_width=True)


# 12. Pie de página simple

st.markdown("---")
st.caption("Dashboard Supermarket Sales | Datos: data.csv")

# FIN

