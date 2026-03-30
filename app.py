import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

from optimizers import beale_numpy, run

# --- Surface mesh (precomputed, log1p-compressed) ---
_lin = np.linspace(-5, 5, 80)  # 80×80 grid — fast enough for smooth surface
X, Y = np.meshgrid(_lin, _lin)
Z = np.log1p(beale_numpy(X, Y))  # log1p compression — never use raw Beale

# Precompute surface trace once — it never changes between callbacks
SURFACE = go.Surface(
    x=_lin, y=_lin, z=Z,
    colorscale='Viridis', opacity=0.7,
    showscale=False, name='Beale (log1p)',
    hoverinfo='skip',
)

# --- Trace colors (fixed) ---
COLORS = {
    'sgd': '#EF4444',
    'momentum': '#3B82F6',
    'nesterov': '#10B981',
    'adam': '#F59E0B',
}

LAYOUT = go.Layout(
    scene=dict(
        xaxis_title='w₁',
        yaxis_title='w₂',
        zaxis_title='log(J+1)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        aspectmode='cube',
    ),
    paper_bgcolor='#0f172a',
    plot_bgcolor='#0f172a',
    font=dict(color='white'),
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(x=0.01, y=0.99),
)


def build_figure(lr, mu, b1, b2, n, selected):
    """Build the complete 3D figure with surface and optimizer trajectories."""
    traces = [SURFACE]
    for opt in (selected or []):
        xs, ys, js = run(opt, lr, mu, b1, b2, n)
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=js,
            mode='lines+markers',
            line=dict(color=COLORS[opt], width=4),
            marker=dict(size=[2] * (len(xs) - 1) + [8]),  # large final marker
            name=opt.upper(),
        ))
    return go.Figure(data=traces, layout=LAYOUT)


# Build initial figure at import time so the page loads with a graph immediately
DEFAULTS = dict(lr=0.01, mu=0.9, b1=0.9, b2=0.999, n=200,
                selected=['sgd', 'momentum', 'nesterov', 'adam'])
INITIAL_FIG = build_figure(**DEFAULTS)

# --- Dash app ---
app = dash.Dash(__name__)


# --- Reusable style helpers ---
INFO_STYLE = {
    'fontSize': '11px',
    'color': '#94a3b8',
    'marginBottom': '14px',
    'lineHeight': '1.5',
    'borderLeft': '2px solid #334155',
    'paddingLeft': '8px',
}

SECTION_STYLE = {
    'marginBottom': '4px',
    'fontSize': '12px',
    'fontWeight': '600',
    'color': '#e2e8f0',
    'letterSpacing': '0.05em',
    'textTransform': 'uppercase',
}

DIVIDER_STYLE = {
    'borderTop': '1px solid #1e293b',
    'margin': '14px 0',
}

def info(text):
    return html.P(text, style=INFO_STYLE)

def section(label):
    return html.P(label, style=SECTION_STYLE)

def divider():
    return html.Hr(style=DIVIDER_STYLE)


app.layout = html.Div(style={'display': 'flex', 'backgroundColor': '#0f172a',
                              'height': '100vh', 'color': 'white',
                              'fontFamily': 'sans-serif'}, children=[
    # Left panel — sliders & checklist (25%)
    html.Div(style={'width': '25%', 'padding': '20px', 'overflowY': 'auto'}, children=[
        html.H2("Optimizer Physics", style={'marginBottom': '4px'}),
        info("Visualización 3D de cómo distintos optimizadores descienden la función de Beale, "
             "un benchmark clásico con un mínimo global en (3, 0.5). La superficie muestra "
             "log(J+1) para comprimir el rango dinámico y poder apreciar la topología."),

        divider(),
        section("⚙️ Hiperparámetros globales"),

        html.Label("Learning Rate (lr)"),
        info("Tamaño del paso en cada iteración. Valores altos convergen rápido pero pueden "
             "oscilar o divergir; valores bajos son estables pero lentos."),
        dcc.Slider(id='lr', min=0.001, max=0.1, step=0.001, value=0.01,
                   marks=None, tooltip={"placement": "bottom"}),

        html.Label("Momentum (μ)"),
        info("Fracción de la velocidad anterior que se conserva en cada paso (SGD y Nesterov). "
             "μ = 0 es SGD puro; cerca de 1 acumula mucho impulso y puede sobrepasar el mínimo."),
        dcc.Slider(id='mu', min=0.0, max=0.99, step=0.01, value=0.9,
                   marks=None, tooltip={"placement": "bottom"}),

        html.Label("β₁ (ADAM)"),
        info("Media móvil exponencial del gradiente en ADAM. Controla el 'impulso' del primer "
             "momento. Valor típico: 0.9."),
        dcc.Slider(id='beta1', min=0.8, max=0.99, step=0.01, value=0.9,
                   marks=None, tooltip={"placement": "bottom"}),

        html.Label("β₂ (ADAM)"),
        info("Media móvil exponencial del gradiente al cuadrado. Escala adaptativamente el lr "
             "por coordenada. Valor típico: 0.999."),
        dcc.Slider(id='beta2', min=0.9, max=0.999, step=0.001, value=0.999,
                   marks=None, tooltip={"placement": "bottom"}),

        html.Label("Iterations"),
        info("Número de pasos de optimización. Más iteraciones permiten ver si los optimizadores "
             "convergen al mínimo global o quedan atrapados."),
        dcc.Slider(id='iters', min=50, max=500, step=50, value=200,
                   marks=None, tooltip={"placement": "bottom"}),

        divider(),
        section("🔬 Optimizadores"),
        info("Todos parten del mismo punto (-4.5, -4.5). Los gradientes se recortan a norma ≤ 10 "
             "para evitar explosión en las zonas de alta curvatura de Beale."),

        dcc.Checklist(
            id='opts',
            options=[
                {'label': ' SGD — Descenso de gradiente puro, sin memoria.', 'value': 'sgd'},
                {'label': ' Momentum — SGD + inercia acumulada.', 'value': 'momentum'},
                {'label': ' Nesterov — Momentum con corrección anticipada.', 'value': 'nesterov'},
                {'label': ' ADAM — Lr adaptativo por parámetro.', 'value': 'adam'},
            ],
            value=['sgd', 'momentum', 'nesterov', 'adam'],
            style={'marginTop': '8px', 'lineHeight': '2'},
            labelStyle={'color': 'white', 'fontSize': '12px'},
        ),

        divider(),
        html.P("★ El marcador grande al final de cada trayectoria indica la posición final "
               "del optimizador. El mínimo global está en (3, 0.5).",
               style={**INFO_STYLE, 'borderColor': '#f59e0b', 'color': '#fcd34d'}),
    ]),

    # Right panel — 3D plot (75%)
    html.Div(style={'width': '75%', 'height': '100vh'}, children=[
        dcc.Graph(id='plot', figure=INITIAL_FIG, style={'height': '90vh'}),
    ]),
])


@app.callback(
    Output('plot', 'figure'),
    [Input('lr', 'value'), Input('mu', 'value'),
     Input('beta1', 'value'), Input('beta2', 'value'),
     Input('iters', 'value'), Input('opts', 'value')],
    prevent_initial_call=True,
)
def update(lr, mu, b1, b2, n, selected):
    return build_figure(lr, mu, b1, b2, n, selected)


if __name__ == '__main__':
    app.run(debug=True)
