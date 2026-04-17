# PACE-MAPP Visualizer — How It Works

A high-level guide for Python users who want to understand, use, or adapt the visualizer.

---

## What Is This?

The PACE-MAPP visualizer is an **interactive web app** built with [Plotly Dash](https://dash.plotly.com/). It runs entirely in Python — no JavaScript required. You launch it from the command line, it starts a local web server, and you interact with it in a browser.

```bash
python plotPACEMAPP_plotly_newFormat.py --dir testData/
```

Dash is essentially a glue layer between two well-known Python libraries:

- **Plotly** — generates the interactive figures
- **Flask** — serves the app in the browser

You write everything in Python. Dash translates your Python layout code into HTML/CSS and wires up interactivity automatically.

---

## The Three Pillars

The entire app is organized around three concepts. Understanding these three things is enough to read, modify, or extend any part of the code.

```
┌─────────────────────────────────────────────────────────┐
│                        BROWSER                          │
│                                                         │
│   ┌─────────────┐    user action    ┌───────────────┐   │
│   │   LAYOUT    │ ───────────────▶  │   CALLBACKS   │   │
│   │  (what you  │                   │  (what happens│   │
│   │   see)      │ ◀───────────────  │   next)       │   │
│   └─────────────┘    updated UI     └───────┬───────┘   │
│                                             │           │
└─────────────────────────────────────────────┼───────────┘
                                              │ calls
                                    ┌─────────▼──────────┐
                                    │  HELPER FUNCTIONS  │
                                    │  (data reading,    │
                                    │   filtering,       │
                                    │   figure creation) │
                                    └────────────────────┘
```

---

## Pillar 1 — Layout (What You See)

**Location in code:** inside `run_app()`, starting around line 5846 — the `app.layout = html.Div([...])` block.

The layout describes every visual element on the page: dropdowns, sliders, plots, labels, containers. It is defined **once at startup** and does not change — only the *contents* of elements are updated later by callbacks.

Dash provides HTML building blocks as Python objects:

| Dash component | What it renders |
|---|---|
| `html.Div(...)` | A box/container (like a `<div>`) |
| `html.Label(...)` | A text label |
| `dcc.Dropdown(...)` | A dropdown menu |
| `dcc.Slider(...)` | A slider |
| `dcc.Graph(...)` | A Plotly figure |
| `dcc.Store(...)` | Invisible data store (no visual) |

Every interactive element is given a unique **`id`** string. This ID is the address callbacks use to read from or write to that element.

**Example — a dropdown in the layout:**
```python
dcc.Dropdown(
    id='file-selector',          # unique ID — callbacks refer to this
    options=file_options,        # list of {label, value} dicts
    value=None,                  # currently selected value
    placeholder="Select a file"
)
```

The layout is purely declarative, it just says *what exists*, not *what it does*. Behavior lives entirely in callbacks.

---

## Pillar 2 — Callbacks (What Happens)

**Location in code:** also inside `run_app()`, starting around line 7027. There are ~35 callbacks total.

A **callback** is a Python function that Dash calls automatically whenever a specified UI element changes. You decorate it with `@app.callback(...)` to tell Dash:

- **Output** — which element(s) to update, and which property to set (e.g., `'figure'`, `'options'`, `'style'`)
- **Input** — which element(s) trigger this function when they change
- **State** — which element(s) to read passively (don't trigger, just provide current value)

**Minimal skeleton:**
```python
@app.callback(
    Output('my-plot', 'figure'),       # → update this element's figure
    Input('file-selector', 'value'),   # ← triggered when this changes
    Input('cost-slider', 'value'),     # ← also triggered when this changes
)
def update_plot(selected_file, max_cost):
    data = get_cached_data(selected_file)
    fig = create_scatter_plot(data, max_cost=max_cost)
    return fig                         # returned value goes to Output
```

When the user picks a new file or moves the cost slider, Dash calls `update_plot()` automatically and swaps in the new figure, no manual event handling needed.

**Key callbacks in this app:**

| Callback function | Triggered by | Updates |
|---|---|---|
| `update_plot_type_options` | File selection | Available tabs in the plot-type dropdown |
| `update_scatter_plot` | File, tab, property, cost | Main scatter map + intensity/DoLP plots |
| `update_aod_time_plots` | File, tab, property, cost, HSRL selector | Property vs Time plots |
| `handle_scatter_click` | Click on scatter map | Properties table for selected pixel |
| `handle_time_plot_click` | Click on time plot | Properties table for selected time point |
| `apply_cost_filter` | "Apply" button on cost slider | Stored cost value used by all plot callbacks |

**Inputs vs State (the key distinction):**

- `Input(...)` — triggers the callback when it changes
- `State(...)` — read-only; does not trigger the callback, just provides the current value

This matters when you want to read a value (e.g., which file is loaded) without re-running the callback every time that value changes.

---

## Pillar 3 — Helper Functions (Data & Figures)

**Location in code:** the top ~4000 lines of the file, before `run_app()`.

Callbacks are kept thin because they mostly orchestrate. The actual work of reading files, filtering data, and building figures is delegated to standalone helper functions. This makes the code testable and reusable.

### Data reading & caching

| Function | Purpose |
|---|---|
| `scan_directory_for_files(dir)` | Finds all `.h5` / `.nc` files in the data directory |
| `load_retrieval_file(path)` | Reads a retrieval file into a flat `data_dict` |
| `get_cached_data(path)` | Wrapper around `load_retrieval_file` — caches result so re-selecting a file is instant |
| `read_hsrl_file(path)` | Reads HSRL2 lidar HDF5 files (separate cache) |
| `detect_file_format(path)` | Distinguishes between RSP, HARP2, SPEX, and PACE-MAPP format variants |

All file data ends up in a **`data_dict`** — a plain Python dictionary where keys are variable names (e.g., `'optical_depth_total_532'`, `'latitude'`, `'rsp_time'`) and values are NumPy arrays. All downstream functions operate on this dictionary.

### Filtering

| Function | Purpose |
|---|---|
| `filter_by_cost(data_dict, max_cost)` | Masks out pixels whose cost function value exceeds the threshold |
| `filter_by_intensity_threshold(...)` | Masks out RSP pixels that fail an intensity residual check |
| `apply_threshold_if_needed(data_dict, params)` | Wrapper that applies threshold only if params are set |

### Figure creation

Each plot type has a dedicated creation function that takes a `data_dict` and returns a Plotly `fig` object. Callbacks call these and pass the result straight to a `dcc.Graph`.

| Function | Plot it creates |
|---|---|
| `create_scatter_plot_only(...)` | Main geospatial scatter map |
| `create_property_vs_time_plot(...)` | Property vs time (RSP flight track) |
| `create_aod_total_plot(...)` | AOD vs wavelength for a clicked pixel |
| `create_angular_combined_plot(...)` | Intensity + DoLP vs scattering angle |
| `create_property_histogram(...)` | Histogram of any retrieval property |
| `create_residual_plot(...)` | Fit residuals for a clicked pixel |
| `create_properties_table_compact(...)` | HTML table of all properties at a clicked pixel |

---

## Data Flow — End to End

Here is what happens from the moment a user selects a file to seeing a plot:

```
1. User picks a file from the dropdown
        │
        ▼
2. `get_cached_data()` reads the file (or returns cached result)
   → produces data_dict  {var_name: np.array, ...}
        │
        ▼
3. A plot callback fires (e.g., update_scatter_plot)
   → calls filter functions to apply cost / threshold masks
   → calls a figure-creation function (e.g., create_scatter_plot_only)
   → figure-creation function builds a Plotly go.Figure with go.Scatter traces
        │
        ▼
4. Callback returns the figure → Dash updates dcc.Graph in the browser
        │
        ▼
5. User clicks a point on the scatter map
   → handle_scatter_click callback fires
   → reads original pixel index from customdata attached to the trace
   → calls create_properties_table_compact() with that index
   → returns HTML table → displayed below the plot
```

---

## Adapting the Visualizer

The most common customizations and where to make them:

| What you want to change | Where to look |
|---|---|
| Add a new plot type | Write a `create_xyz_plot(data_dict, ...)` helper, add a tab option in `update_plot_type_options`, add a `dcc.Graph` to the layout, wire up a new callback |
| Add a new dropdown control | Add `dcc.Dropdown(id='my-control', ...)` to the layout, add `Input('my-control', 'value')` to the relevant callback |
| Support a new file format | Add a format detection branch in `detect_file_format()`, add a reading branch in `load_retrieval_file()` |
| Change what appears in hover tooltips | Find the `hovertemplate=` argument in the relevant `fig.add_trace(go.Scatter(...))` call |
| Change plot colors or styling | Find the `marker=dict(...)` / `line=dict(...)` arguments in the figure-creation function |

---

## File Structure

```
plotPACEMAPP_plotly_newFormat.py  ← the entire app (single file)
testData/                         ← sample retrieval files (.h5, .nc)
assets/                           ← static files (CSS, images) auto-served by Dash
README.md                         ← setup and usage instructions
```

The app is intentionally a single file to make it easy to share and modify without a package structure. For larger adaptations, the helper functions (Pillar 3) are good candidates to split into separate modules.
