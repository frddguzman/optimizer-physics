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

app.layout = html.Div(style={'display': 'flex', 'backgroundColor': '#0f172a',
                              'height': '100vh', 'color': 'white',
                              'fontFamily': 'sans-serif'}, children=[
    # Left panel — sliders & checklist (25%)
    html.Div(style={'width': '25%', 'padding': '20px', 'overflowY': 'auto'}, children=[
        html.H2("Optimizer Physics"),

        html.Label("Learning Rate (lr)"),
        dcc.Slider(id='lr', min=0.001, max=0.1, step=0.001, value=0.01,
                   marks=None, tooltip={"placement": "bottom"}),

        html.Label("Momentum (μ)"),
        dcc.Slider(id='mu', min=0.0, max=0.99, step=0.01, value=0.9,
                   marks=None, tooltip={"placement": "bottom"}),

        html.Label("β₁ (ADAM)"),
        dcc.Slider(id='beta1', min=0.8, max=0.99, step=0.01, value=0.9,
                   marks=None, tooltip={"placement": "bottom"}),

        html.Label("β₂ (ADAM)"),
        dcc.Slider(id='beta2', min=0.9, max=0.999, step=0.001, value=0.999,
                   marks=None, tooltip={"placement": "bottom"}),

        html.Label("Iterations"),
        dcc.Slider(id='iters', min=50, max=500, step=50, value=200,
                   marks=None, tooltip={"placement": "bottom"}),

        html.Label("Optimizers"),
        dcc.Checklist(
            id='opts',
            options=[
                {'label': ' SGD', 'value': 'sgd'},
                {'label': ' Momentum', 'value': 'momentum'},
                {'label': ' Nesterov', 'value': 'nesterov'},
                {'label': ' ADAM', 'value': 'adam'},
            ],
            value=['sgd', 'momentum', 'nesterov', 'adam'],
            style={'marginTop': '8px'},
            labelStyle={'color': 'white'},
        ),
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
