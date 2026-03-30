# Optimizer Physics

Interactive 3D visualization of how gradient-based optimizers navigate a non-convex loss landscape.

Built for the **Modelos Conexionistas** contest (Ciencia de Datos, UV).

## What it does

The app renders [Beale's function](https://en.wikipedia.org/wiki/Test_functions_for_optimization) as a 3D surface and traces the paths that four different optimizers take from the same starting point `(-4.5, -4.5)` toward the global minimum at `(3, 0.5)`:

- **SGD** -- stalls near the saddle region, barely makes progress
- **Momentum** -- overshoots and oscillates, but eventually advances
- **Nesterov** -- similar to Momentum but with tighter turns and faster convergence
- **ADAM** -- adaptive step sizes let it ignore the steep gradients and curve directly toward the minimum

All optimization is done with `torch.optim` and autograd on a 2D parameter tensor -- no neural network, no dataset, just pure optimizer kinematics on a mathematical surface.

## Demo

The surface z-values are `log(1 + J)` compressed so the extreme peaks at the domain edges don't blow out the colorscale. Trajectories are rendered as colored 3D paths lying directly on the surface.

### Controls

| Slider | What it changes |
|--------|----------------|
| Learning Rate | Step size for all optimizers |
| Momentum (mu) | Momentum coefficient for SGD+Momentum and Nesterov |
| beta1, beta2 | ADAM exponential decay rates |
| Iterations | Number of optimization steps (50--500) |

The checklist toggles which optimizers are shown. All updates are reactive -- drag a slider and the trajectories recompute instantly.

## Run

```bash
git clone https://github.com/frddguzman/optimizer-physics.git
cd optimizer-physics
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8050.

## Stack

- **Dash** -- layout and reactive callbacks
- **Plotly** -- 3D surface and scatter traces
- **PyTorch** -- `torch.optim` optimizers with autograd gradients
- **NumPy** -- surface mesh computation
