# PACE-MAPP Aerosol Properties Interactive Visualization Tool

## Overview

This Python application creates an interactive web-based visualization tool for
exploring atmospheric aerosol properties from PACE (Plankton, Aerosol, Cloud,
ocean Ecosystem). PACE intstruments are HARP2 (polarimeter), SPEXone
(polarimeter), and OCI and the Microphysical Aerosol Properties from
Polarimetry (MAPP) retrieval framework [(Stamnes et
al., 2023)](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2023.1174672/full).

The tool displays three interconnected plots that help users analyze the spatial
distribution of aerosol properties and the "quality" of the retrieval, or how
well we modeled actual measurements of Intensity and Degreee of Linear
Polarization (DoLP)

## What This Tool Does

The application creates a dashboard with three main components:

1. **Geographic Map** (left): Shows aerosol properties overlaid on a map
2. **Intensity vs. Viewing Angle Plot** (middle): Displays how light intensity
varies with viewing angle
3. **Degree of Linear Polarization (DoLP) vs. Viewing Angle Plot** (right):
Shows polarization measurements

When you click on a point on the map, the other two plots update to show the
measured/modeled Intensity/DoLP for that specific location.

## Key Packages Used

- **Dash**: A Python web framework for building interactive web applications
- **Plotly**: Creates interactive plots and visualizations  
- **h5py**: Reads HDF5 scientific data files
- **NumPy**: Handles numerical computations and array operations

## Setting Up Virtual Environment and Installation Requirements

Before running this code, you need to create a virtual environment:
```bash
python3 -m venv myenv
```

activate the environment:
```bash
source myenv/bin/activate
```

install the required Python packages from requirements.txt
```bash
pip install -r requirements.txt
```

## How to Run the Application

The application is run from the command line with a **directory path**
containing your data files:

```bash
python plotPACEMAPP_plotly.py --directory /path/to/your/data/files
```

The directory should contain `.h5` or `.nc` files with PACE-MAPP retrieval
results.

## Code Structure Overview

### Main Components

The code is organized into several key sections:

#### 1. **Imports and Configuration** (Lines ~1-20)
```python
import numpy as np
import h5py
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
```
These import necessary libraries for data handling, plotting, and web interface
creation.

#### 2. **Utility Functions** (Lines ~25-100)
- `scan_directory_for_files()`: Finds all .nc of .h5 data files in a directory
- `find_nearest_point()`: Locates the closest data point to clicked coordinates  
- `determine_retrieval_scenario()`: Identifies which instruments were used in
  the retrieval

#### 3. **Data Reading Functions** (Lines ~100-400)
- `read_hdf5_variables()`: Loads data from HDF5 files into Python dictionaries
- `filter_by_cost()`: Removes poor-quality data points based on cost function values
- `get_channel_intensity_dolp_vza()`: Extracts optical measurements for specific locations

#### 4. **Plotting Functions** (Lines ~400-800)
- `create_scatter_plot_only()`: Creates the geographic map with aerosol
  properties
- `create_combined_intensity_dolp_plots()`: Creates the intensity and dolp plots
- `create_export_figure()`: Prepares a plot for .png export

#### 5. **Dash App Layout** (Lines ~800-900)
Defines the HTML structure and styling for the web interface, including:
- Dropdown menus for selecting retrieval file and aerosol properties
- Three-column layout for the plots
- Controls for cost filtering data

#### 6. **Callback Functions** (Lines ~900-1200)
These functions handle user interactions:
- When you select a different aerosol property, the map updates
- When you click a point on the map, the Intenstiy/DoLP plots populate
- When you change the cost filter, pixels not meeting this criteria are removed

#### 7. **Main Function** (Lines ~1200-1300)
- `run_app()`: Sets up and launches the web application
- Command-line argument parsing for specifying data directory

## Understanding the Data

### Atmospheric Science Context

**Aerosols** are tiny particles suspended in the atmosphere (like dust, smoke,
or sea salt). We are interested in their properties because they:
- Affect climate by scattering and absorbing sunlight
- Impact air quality and human health
- Interacts with clouds

**PACE** the instruments measure the intensity and polarization of light
at the top of the atmosphere at multiple wavelengths and viewing angles.

**PACE-MAPP** this refers to the retrieval algorithm that infers aerosol
properties from the above measurements.

### Key Data Variables

The tool displays various aerosol properties:

- **Aerosol Optical Depth (AOD)**: The extinction of light due to aerosols
(higher AOD = more extinction)
- **Single Scattering Albedo**: How much light aerosols scatter vs. absorb
- **Asymmetry Parameter**: Describes the directional scattering pattern
- **Refractive Index**: Optical properties of the aerosol material
- **Size Distribution**: Range of particle sizes present

### Quality Control

The **cost function** indicates how well the retrieval algorithm performed:
- Lower values = higher confidence in the results
- Higher values = more uncertainty
- The tool allows filtering out high-cost (low-quality) retrievals

## How the Interface Works

### Three-Panel Layout

1. **Left Panel (25% width)**: Controls and information
   - File selector dropdown
   - Aerosol property selector  
   - Cost function filter slider
   - Clicked point information
   - Properties table

2. **Middle Panel (31% width)**: Geographic scatter map
   - Each point represents a measurement location
   - Colors indicate aerosol property values
   - Click points to see detailed measurements

3. **Right Panel (42% width)**: Intensity and DoLP plots
   - Shows how measurements vary with viewing angle
   - Updates when you click on map points
   - Helps validate retrieval quality

### Interactive Features

- **Click on map points**: Updates the intensity/DoLP plots for that location
- **Change aerosol property**: Updates the map colors and data display
- **Adjust cost filter**: Hides low-quality data points
- **Export functionality**: Save plots as images

## Key Programming Concepts

### Dash Callbacks

Callbacks are functions that automatically run when users interact with the interface:

```python
@app.callback(
    Output('aerosol-plot', 'figure'),  # What gets updated
    Input('property-selector', 'value')  # What triggers the update
)
def update_plot(selected_property):
    # Function that runs when property changes
    return new_figure
```

### Data Handling Patterns

The code uses several important patterns:

1. **Dictionary-based data storage**: All arrays stored in `data_dict`
2. **2D array indexing**: Geographic data stored as `[latitude_index, longitude_index]`
3. **Masking for quality control**: Using boolean arrays to filter data
4. **Flattening for plotting**: Converting 2D geographic grids to 1D arrays

### Error Handling

The code includes extensive error checking:
- File existence validation
- Data shape verification  
- Graceful handling of missing data
- Debug print statements for troubleshooting

## Common Modifications You Might Make

### Adding New Aerosol Properties

To display additional variables from the data files:

1. **Find the variable name** in the HDF5 file structure
2. **Add it to the reading function** in `read_hdf5_variables()`
3. **Update the dropdown options** in `create_dropdown_options()`

### Changing the Map Display

To modify how the geographic data appears:

1. **Edit `create_scatter_plot_only()`** function
2. **Change color scales** by modifying the `colorscale` parameter
3. **Adjust point sizes** by changing the `size` parameter

### Adding New Plot Types

To create additional visualization panels:

1. **Create a new plotting function** following the pattern of existing ones
2. **Add the plot to the layout** in the app layout section
3. **Create callbacks** to handle user interactions

### Modifying Data Filtering

To change how data quality filtering works:

1. **Edit `filter_by_cost()`** function
2. **Adjust threshold values** for what constitutes "good" data
3. **Add new filtering criteria** beyond just cost function

## Debugging Tips

### Common Issues and Solutions

1. **"No module named" errors**: Install missing packages with pip
2. **File not found**: Check that your data directory path is correct
3. **Empty plots**: Verify your data files contain the expected variables
4. **Slow performance**: Reduce the number of data points or increase filtering

### Debug Mode

The code includes a `debug` variable at the top. Set `debug = 2` for verbose output that shows:
- Data loading progress
- Array shapes and sizes
- Processing steps
- Error details

### Using Print Statements

Add print statements to understand data flow:

```python
print(f"Data shape: {data_array.shape}")
print(f"Min/max values: {np.min(data_array):.3f} / {np.max(data_array):.3f}")
print(f"Number of valid points: {np.sum(np.isfinite(data_array))}")
```

## Extending/modifying the Code

### Ideas for Student Involvement/Addition

1. **Add statistical analysis**: Calculate correlations between different aerosol properties
2. **Implement data export**: Allow downloading filtered datasets as CSV files  
3. **Create time series plots**: If you have multiple files, show how properties change over time
4. **Add machine learning**: Use aerosol properties to classify different aerosol types
5. **Improve visualization**: Add 3D plots, animation, or additional map layers
6. **Performance optimization**: Implement data caching or more efficient plotting

### Best Practices for Development

1. **Test with small datasets first**: Use a subset of data while developing
2. **Comment your code extensively**: Explain what each section does
3. **Use version control**: Keep track of your changes with git
4. **Validate your results**: Compare outputs with known good data
5. **Handle edge cases**: What happens with missing data or unusual values?

## File Structure Summary

The main file `plotPACEMAPP_plotly.py` contains:

- **Lines 1-50**: Imports, constants, and utility functions
- **Lines 50-500**: Data reading and processing functions  
- **Lines 500-800**: Plotting and visualization functions
- **Lines 800-1000**: Dash app layout definition
- **Lines 1000-1200**: Interactive callback functions
- **Lines 1200+**: Main application setup and command-line interface

Understanding this structure will help you navigate the code and make targeted modifications for your specific needs.

## Getting Help

If you encounter issues:

1. **Check the debug output**: Set `debug = 2` and look for error messages
2. **Verify your data files**: Make sure they contain the expected variables
3. **Test with different datasets**: Some files might have different structures
4. **Read the error messages carefully**: They often indicate exactly what's wrong
5. **Use online resources**: Dash and Plotly have excellent documentation
