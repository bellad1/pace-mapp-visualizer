import numpy as np
import h5py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import colorsys
import argparse
import glob
import os
import base64
import traceback


# Global Variables
PLOT_WIDTH = 1550  # in pixels
debug = 1
verbose = 1
default_cost = 0.5
_data_cache = {}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_cached_data(file_path):
    """
    Get data from cache or load if not cached
    """
    if file_path not in _data_cache:
        print(f"Loading and caching data for: {file_path}")

        # Use wrapper function that detects format and routes to correct reader
        file_format = detect_file_format(file_path)
        print(f"Detected file format: {file_format}")

        data_dict, sorted_vars, display_names, var_metadata = load_retrieval_file(file_path)

        _data_cache[file_path] = {
            'data_dict': data_dict,
            'sorted_variables': sorted_vars,
            'display_names': display_names,
            'variable_metadata': var_metadata
        }
    else:
        print(f"Using cached data for: {file_path}")

    return _data_cache[file_path]


def clear_data_cache():
    """
    Clear cache if needed
    """
    global _data_cache
    _data_cache.clear()


def scan_directory_for_files(directory_path):
    """
    Scan specified directory for specified .h5 and .nc files

    Args:
        directory_path: (str) path to directory containing retrieval files

    Returns:
        all_files (list): A list of file paths
    """
    # Make sure directory exists
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")

    # Find all .h5 and .nc files
    h5_files = glob.glob(os.path.join(directory_path, "*.h5"))
    nc_files = glob.glob(os.path.join(directory_path, "*.nc"))

    # Combine and sort alphabetically
    all_files = sorted(h5_files + nc_files)

    if not all_files:
        raise ValueError(f"No .h5 or .nc files found in {directory_path}")

    return all_files


def find_nearest_point(lats, lons, target_lat, target_lon):
    """
    Find the index of the measurement closest to target lat/lon

    Args:
        lats:
        lons:
        target_lat:
        target_lon:

    Returns:
        closest_idx: Index of closest point
    """
    # Calculate distance:
    # Start with simplified euclidian method. May need to be improved (e.g.,
    # haversine formula?)
    distances = np.sqrt((lats - target_lat)**2 + (lons - target_lon)**2)

    # Find index of min distance
    closest_idx = np.argmin(distances)

    return closest_idx


def determine_retrieval_scenario(file_path):
    """
    **NOTE**: This is currently not being used (or only by the export function).
    Likely this and the export function should be updated

    Determine which instruments were used in the retrieval

    Args:
        file_path: full path to the retrieval file

    Returns:
        value:
    """
    has_oci = "OCI" in file_path
    has_harp = "HARP" in file_path
    has_spex = "SPEX" in file_path

    if has_spex and has_harp and not has_oci:
        return 1  # "Scenario 1: SPEX and HARP (no OCI)"
    elif has_spex and has_harp and has_oci:
        return 2  # "Scenario 2: SPEX, HARP, and OCI"
    elif has_harp and not has_spex and not has_oci:
        return 3  # "Scenario 3: HARP only (no OCI or SPEX)"
    elif has_spex and not has_harp and not has_oci:
        return 4  # "Scenario 4: SPEX only (no OCI or HARP)
    else:
        raise ValueError(f"Unsupported instrument combination in filename: {file_path}")


def get_distance_warning(distance_km, time_diff_minutes=None):
    """
    Generate color-coded warning based on spatial and temporal distance

    Returns:
        dict with 'color', 'icon', and 'text' keys
    """
    if distance_km < 10:
        spatial_warning = {
            "color": "#27ae60",
            "icon": "✓",
            "text": "Good spatial match"
        }
    elif distance_km < 50:
        spatial_warning = {
            "color": "#f39c12",
            "icon": "⚠",
            "text": "Moderate spatial difference"
        }
    else:
        spatial_warning = {
            "color": "#e74c3c",
            "icon": "⚠",
            "text": "Large spatial difference - use caution"
        }

    return spatial_warning


def extract_timestamp_from_filename(filepath):
    """
    Extract timestamp from filename if available
    Returns formatted string or 'Unknown'
    """
    import re
    from datetime import datetime

    filename = os.path.basename(filepath)

    # Try to extract timestamp pattern like: 20240704T161336
    match = re.search(r'(\d{8})T(\d{6})', filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)

        try:
            dt = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
            result = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
            return result
        except Exception as e:
            print(f"DEBUG: Error parsing datetime: {e}")

    return "Cannot parse time/date from file name"


def detect_file_format(file_path):
    """
    Detect whether the file is RSP or HARP2 format based on filename.

    Args:
        file_path: (str) path to the retrieval file

    Returns:
        str: 'RSP' or 'HARP2'
    """
    filename = os.path.basename(file_path).lower()

    if 'rsp' in filename:
        return 'RSP'
    else:
        return 'HARP2'


def get_reference_wavelength(filtered_data):
    """
    Find the best reference wavelength for displaying aerosol properties.
    Prefers wavelengths close to 550-556 nm range (green channel).

    Args:
        filtered_data: Dictionary containing data arrays

    Returns:
        str: Reference wavelength (e.g., '556', '555', or closest available)
    """
    # Preferred wavelengths in order of preference (green channel region)
    preferred_wavelengths = ['556', '555', '550', '553', '560']

    # Check if any preferred wavelength exists in the data
    for wl in preferred_wavelengths:
        # Check if any property with this wavelength exists
        test_key = f'optical_depth_fine_{wl}'
        if test_key in filtered_data:
            return wl

    # If no preferred wavelength found, search for any wavelength in the data
    # by checking optical_depth properties
    wavelengths_found = set()
    for key in filtered_data.keys():
        if 'optical_depth_' in key and '_' in key:
            parts = key.split('_')
            if len(parts) >= 3 and parts[-1].isdigit():
                wavelengths_found.add(parts[-1])

    if wavelengths_found:
        # Convert to integers, sort, and pick the middle one (or closest to 550)
        wl_ints = sorted([int(w) for w in wavelengths_found])
        # Find closest to 550 nm
        closest_wl = min(wl_ints, key=lambda x: abs(x - 550))
        return str(closest_wl)

    # Fallback: return '556' (may not exist, but at least won't crash)
    return '556'


def get_available_modes(filtered_data):
    """
    Detect which aerosol modes are available in the dataset.

    Args:
        filtered_data: Dictionary containing data arrays

    Returns:
        list: List of available mode names (e.g., ['fine', 'dust'])
    """
    # Check if available_modes is already stored in data_dict
    if 'available_modes' in filtered_data:
        return filtered_data['available_modes']

    # Otherwise, detect modes by checking for optical_depth variables
    all_modes = ['fine', 'coarse', 'dust', 'sea_salt']
    available_modes = []

    for mode in all_modes:
        # Check multiple properties to confirm mode exists
        test_keys = [
            f'optical_depth_{mode}',  # No wavelength suffix
            f'reff_{mode}',
            f'veff_{mode}',
        ]
        # Also check with common wavelengths
        for wl in ['556', '555', '550']:
            test_keys.append(f'optical_depth_{mode}_{wl}')

        # If any test key exists, this mode is available
        if any(key in filtered_data for key in test_keys):
            available_modes.append(mode)

    return available_modes if available_modes else ['fine', 'coarse', 'dust']


def get_mode_colors():
    """
    Return standard colors for aerosol modes.

    Returns:
        dict: Mapping of mode names to hex colors
    """
    return {
        'fine': '#3498db',      # Blue
        'coarse': '#e74c3c',    # Red
        'dust': '#f39c12',      # Orange
        'sea_salt': '#2ecc71'   # Green
    }


def create_properties_table_compact(filtered_data, selected_row, selected_col, selected_property):
    """
    Create a compact table showing properties by mode with metadata header
    """
    # Helper function to extract scalar from potentially multi-dimensional data
    def extract_scalar(value):
        """Extract first scalar value from potentially nested array"""
        if value is None:
            return None
        # Keep extracting first element until we get a scalar
        while hasattr(value, '__len__') and not isinstance(value, str):
            if len(value) == 0:
                return None
            value = value.flat[0] if hasattr(value, 'flat') else value[0]
        # Convert numpy types to Python types
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # Extract metadata values
    try:
        # Determine data dimensionality from original_shape
        original_shape = filtered_data.get('original_shape', None)
        is_1d = original_shape is not None and len(original_shape) == 1

        # Handle SZA (Solar Zenith Angle)
        sza = filtered_data.get('sza', None)
        if sza is not None:
            if is_1d:
                sza_val = extract_scalar(sza[selected_row])
            else:
                sza_val = extract_scalar(sza[selected_row, selected_col])
            # Convert from cosine to degrees
            sza_val = np.degrees(np.arccos(sza_val)) if sza_val is not None and np.isfinite(sza_val) else None
        else:
            sza_val = None

        # Handle RAA (Relative Azimuth Angle)
        raa = filtered_data.get('raa', None)
        if raa is not None:
            if is_1d:
                raa_val = extract_scalar(raa[selected_row])
            else:
                raa_val = extract_scalar(raa[selected_row, selected_col])
            raa_val = raa_val if raa_val is not None and np.isfinite(raa_val) else None
        else:
            raa_val = None

        # Handle cost function
        cost = filtered_data.get('cost_function', None)
        if cost is not None:
            if is_1d:
                cost_val = extract_scalar(cost[selected_row])
            else:
                cost_val = extract_scalar(cost[selected_row, selected_col])
        else:
            cost_val = None

        # Get selected property value
        selected_val = None
        if selected_property in filtered_data:
            prop_data = filtered_data[selected_property]
            if is_1d:
                selected_val = extract_scalar(prop_data[selected_row])
            else:
                selected_val = extract_scalar(prop_data[selected_row, selected_col])
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        import traceback
        traceback.print_exc()
        sza_val = raa_val = cost_val = selected_val = None

    # Create metadata section
    metadata_items = []

    if sza_val is not None and np.isfinite(sza_val):
        metadata_items.append(
            html.Div([
                html.Strong("SZA: ", style={'fontSize': '12px', 'textAlign': 'left'}),
                html.Span(f"{sza_val:.2f}°", style={'fontSize': '12px', 'fontFamily': 'monospace'})
            ], style={'marginBottom': '3px'})
        )

    if raa_val is not None and np.isfinite(raa_val):
        metadata_items.append(
            html.Div([
                html.Strong("RAA: ", style={'fontSize': '12px'}),
                html.Span(f"{raa_val:.2f}°", style={'fontSize': '12px', 'fontFamily': 'monospace'})
            ], style={'marginBottom': '3px'})
        )

    if selected_property and selected_val is not None and np.isfinite(selected_val):
        # Shorten property name if too long
        display_prop = selected_property.replace('_', ' ').title()
        if len(display_prop) > 25:
            display_prop = display_prop[:22] + "..."

        metadata_items.append(
            html.Div([
                html.Strong("Selected: ", style={'fontSize': '12px'}),
                html.Span(f"{display_prop} = {selected_val:.3f}",
                          style={'fontSize': '12px', 'fontFamily': 'monospace'})
            ], style={'marginBottom': '3px'})
        )

    if cost_val is not None and np.isfinite(cost_val):
        metadata_items.append(
            html.Div([
                html.Strong("Cost: ", style={'fontSize': '12px'}),
                html.Span(f"{cost_val:.3f}", style={'fontSize': '12px', 'fontFamily': 'monospace'})
            ], style={'marginBottom': '3px'})
        )

    metadata_section = html.Div(metadata_items, style={
        'padding': '8px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '4px',
        'marginBottom': '10px',
        'border': '1px solid #e0e0e0',
        'textAlign': 'left'
    }) if metadata_items else None

    # Get reference wavelength and available modes dynamically
    ref_wl = get_reference_wavelength(filtered_data)
    modes = get_available_modes(filtered_data)
    mode_colors = get_mode_colors()

    # Define the properties and their display names (same as before)
    properties_config = [
        ('optical_depth', 'Optical Depth', ref_wl),
        ('ssa', 'Single Scattering Albedo', ref_wl),
        ('real', 'Real Refractive Index', ref_wl),
        ('imag', 'Imaginary Refractive Index', ref_wl),
        ('asymmetry', 'Asymmetry Parameter', ref_wl),
        ('cross_section', 'Cross Section', ref_wl),
        ('number_concentration', 'Number Concentration', ref_wl),
        ('reff', 'Effective Radius', ''),
        ('veff', 'Effective Variance', ''),
    ]

    # Create table header
    header = html.Tr([
        html.Th(f"Property (* {ref_wl} nm)", style={
            'textAlign': 'left',
            'padding': '6px',
            'borderBottom': '2px solid #34495e',
            'fontSize': '14px',
            'fontWeight': 'bold'
        }),
        *[html.Th(mode.title(), style={
            'textAlign': 'center',
            'padding': '6px',
            'borderBottom': '2px solid #34495e',
            'color': mode_colors.get(mode, '#95a5a6'),
            'fontWeight': 'bold',
            'fontSize': '14px'
        }) for mode in modes]
    ])

    # Create table rows (same as before)
    table_rows = [header]

    for prop_base, prop_display, wavelength in properties_config:
        mode_values = {}
        has_data = False

        for mode in modes:
            if wavelength:
                prop_key = f"{prop_base}_{mode}_{wavelength}"
            else:
                prop_key = f"{prop_base}_{mode}"

            if prop_key in filtered_data:
                try:
                    # Handle both 1D (RSP) and 2D (HARP2) indexing
                    prop_data = filtered_data[prop_key]
                    if prop_data.ndim == 1:
                        # 1D data (RSP): use selected_row as index
                        value = prop_data[selected_row]
                    elif prop_data.ndim == 2:
                        # 2D data (HARP2): use [row, col] indexing
                        value = prop_data[selected_row, selected_col]
                    else:
                        value = np.nan

                    if np.isfinite(value):
                        mode_values[mode] = f"{value:.3f}"
                        has_data = True
                    else:
                        mode_values[mode] = "N/A"
                except:
                    mode_values[mode] = "N/A"
            else:
                mode_values[mode] = "-"

        if has_data and any(val not in ["-", "N/A"] for val in mode_values.values()):
            # Use * for reference wavelength properties, show wavelength for others
            if wavelength == ref_wl:
                property_label = f"{prop_display}*"
            elif wavelength:
                property_label = f"{prop_display} ({wavelength}nm)"
            else:
                property_label = prop_display

            is_selected = any(
                f"{prop_base}_{mode}_{wavelength}" == selected_property or
                f"{prop_base}_{mode}" == selected_property
                for mode in modes
            )

            row = html.Tr([
                html.Td(property_label, style={
                    'padding': '6px',
                    'borderBottom': '1px solid #ecf0f1',
                    'fontWeight': 'bold' if is_selected else '500',
                    'fontSize': '12px',
                    'backgroundColor': '#e8f5e9' if is_selected else 'transparent'
                }),
                *[html.Td(mode_values.get(mode, "-"), style={
                    'textAlign': 'center',
                    'padding': '6px',
                    'borderBottom': '1px solid #ecf0f1',
                    'color': mode_colors.get(mode, '#95a5a6') if mode_values.get(mode, "-") not in ["-", "N/A"] else '#95a5a6',
                    'fontSize': '12px',
                    'fontFamily': 'monospace',
                    'backgroundColor': '#e8f5e9' if is_selected else 'transparent'
                }) for mode in modes]
            ])
            table_rows.append(row)

    if len(table_rows) == 1:  # Only header
        table_section = html.P("No properties available", style={'fontSize': '12px', 'color': '#999'})
    else:
        table_section = html.Div([
            html.Table(table_rows, style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'fontSize': '12px'
            })
        ], style={
            'maxHeight': '200px',
            'overflowY': 'auto',
            'border': '1px solid #ddd',
            'borderRadius': '4px',
            'backgroundColor': 'white',
            'padding': '4px'
        })

    # Combine metadata and table
    return html.Div([
        metadata_section,
        table_section
    ])


def create_time_point_properties_table(data_dict, time_index, source_plot, file_path):
    """
    Create a properties table for a clicked time point in RSP data.
    Similar to create_properties_table_compact but adapted for 1D time-series data.

    Args:
        data_dict: Dictionary containing RSP data arrays
        time_index: Index in the time array (original, not filtered)
        source_plot: 'single', 'plot-1', or 'plot-2'
        file_path: Path to the data file

    Returns:
        html.Div: Formatted properties table
    """
    try:
        # Get available modes and reference wavelength
        modes = get_available_modes(data_dict)
        ref_wl = get_reference_wavelength(data_dict)
        mode_colors = get_mode_colors()

        # Properties configuration (same as scatter plot version)
        properties_config = [
            ('optical_depth', 'Optical Depth', ref_wl),
            ('ssa', 'Single Scattering Albedo', ref_wl),
            ('real', 'Real Refractive Index', ref_wl),
            ('imag', 'Imaginary Refractive Index', ref_wl),
            ('asymmetry', 'Asymmetry Parameter', ref_wl),
            ('cross_section', 'Cross Section', ref_wl),
            ('number_concentration', 'Number Concentration', ref_wl),
            ('reff', 'Effective Radius', ''),
            ('veff', 'Effective Variance', ''),
        ]

        # Create table header
        header = html.Tr([
            html.Th(f"Property (* {ref_wl} nm)", style={
                'textAlign': 'left',
                'padding': '6px',
                'borderBottom': '2px solid #34495e',
                'fontSize': '14px',
                'fontWeight': 'bold'
            }),
            *[html.Th(mode.title(), style={
                'textAlign': 'center',
                'padding': '6px',
                'borderBottom': '2px solid #34495e',
                'color': mode_colors.get(mode, '#95a5a6'),
                'fontWeight': 'bold',
                'fontSize': '14px'
            }) for mode in modes]
        ])

        # Create table rows
        table_rows = [header]

        for prop_base, prop_display, wavelength in properties_config:
            mode_values = {}
            has_data = False

            for mode in modes:
                if wavelength:
                    prop_key = f"{prop_base}_{mode}_{wavelength}"
                else:
                    prop_key = f"{prop_base}_{mode}"

                if prop_key in data_dict:
                    try:
                        # RSP data is 1D, so direct indexing
                        value = data_dict[prop_key][time_index]

                        if np.isfinite(value):
                            mode_values[mode] = f"{value:.3f}"
                            has_data = True
                        else:
                            mode_values[mode] = "N/A"
                    except:
                        mode_values[mode] = "N/A"
                else:
                    mode_values[mode] = "-"

            if has_data and any(val not in ["-", "N/A"] for val in mode_values.values()):
                # Property label
                if wavelength == ref_wl:
                    property_label = f"{prop_display}*"
                elif wavelength:
                    property_label = f"{prop_display} ({wavelength}nm)"
                else:
                    property_label = prop_display

                row = html.Tr([
                    html.Td(property_label, style={
                        'padding': '6px',
                        'borderBottom': '1px solid #ecf0f1',
                        'fontWeight': '500',
                        'fontSize': '12px'
                    }),
                    *[html.Td(mode_values.get(mode, "-"), style={
                        'textAlign': 'center',
                        'padding': '6px',
                        'borderBottom': '1px solid #ecf0f1',
                        'color': mode_colors.get(mode, '#95a5a6') if mode_values.get(mode, "-") not in ["-", "N/A"] else '#95a5a6',
                        'fontSize': '12px',
                        'fontFamily': 'monospace'
                    }) for mode in modes]
                ])
                table_rows.append(row)

        if len(table_rows) == 1:  # Only header
            return html.P("No properties available", style={'fontSize': '12px', 'color': '#999', 'textAlign': 'center'})

        # Create table with file identifier for multi-file mode
        file_label = ""
        if source_plot in ['plot-1', 'plot-2']:
            file_num = '1' if source_plot == 'plot-1' else '2'
            filename = os.path.basename(file_path) if file_path else f"File {file_num}"
            file_label = html.Div([
                html.Strong(f"File: "), filename
            ], style={
                'marginBottom': '10px',
                'fontSize': '12px',
                'textAlign': 'center',
                'color': '#7f8c8d'
            })

        return html.Div([
            file_label,
            html.Div([
                html.Table(table_rows, style={
                    'width': '100%',
                    'borderCollapse': 'collapse',
                    'fontSize': '12px'
                })
            ], style={
                'maxHeight': '300px',
                'overflowY': 'auto',
                'border': '1px solid #ddd',
                'borderRadius': '4px',
                'backgroundColor': 'white',
                'padding': '4px'
            })
        ])

    except Exception as e:
        print(f"Error creating time point properties table: {e}")
        import traceback
        traceback.print_exc()
        return html.P(f"Error creating table: {str(e)}", style={'color': 'red', 'fontSize': '12px'})


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================
def get_wavelength_mapping_pace():
    """
    Returns wavelength-instrument mapping for PACE files (HARP/SPEX/OCI).
    Modify this to add new PACE instruments/wavelengths.

    Returns:
        wavelength_mapping (list): containing each measurement wavelength,
            corresponding instrument, and number of views.
    """
    wavelength_mapping = [
        (556, 'SPEX', 5),
        (413, 'SPEX', 5),
        (440, 'HARP', 10),
        (470, 'SPEX', 5),
        (533, 'SPEX', 5),
        (550, 'HARP', 10),
        (665, 'HARP', 60),
        (866, 'HARP', 10),
        (1038, 'OCI', 1),
        (1618, 'OCI', 1),
        (2130, 'OCI', 1),
        (2258, 'OCI', 1),
        (437, 'SPEX', 5),
        (668, 'SPEX', 5)
    ]

    return wavelength_mapping


def get_wavelength_mapping_rsp():
    """
    Returns wavelength-instrument mapping for RSP files.
    Modify this to add new RSP wavelengths.

    RSP has 504 viewing angles total, shared across all wavelengths.
    Each wavelength has 126 viewing angles (504 / 4 wavelengths).

    Returns:
        wavelength_mapping (list): containing each measurement wavelength,
            corresponding instrument, and number of views.
    """
    num_angles = 126
    # wavelength_mapping = [
    #     (555, 'RSP', 126),
    #     (410, 'RSP', 126),
    #     (469, 'RSP', 126),
    #     (670, 'RSP', 126),
    #     (864, 'RSP', 126)
    # ]
    wavelength_mapping = [
        (555, 'RSP', num_angles),
        (410, 'RSP', num_angles),
        (469, 'RSP', num_angles),
        (670, 'RSP', num_angles),
        (864, 'RSP', num_angles)
    ]

    return wavelength_mapping


def get_wavelength_instrument_mapping(file_format='HARP2'):
    """
    Dispatcher function that returns the appropriate wavelength mapping
    based on file format.

    Args:
        file_format (str): File format type ('HARP2', 'RSP', etc.)

    Returns:
        wavelength_mapping (list): format-specific wavelength mapping
    """
    if file_format == 'RSP':
        return get_wavelength_mapping_rsp()
    else:  # Default to PACE/HARP2
        return get_wavelength_mapping_pace()


def build_channel_ranges(wavelength_mapping, output_channels_order=None):
    """
    Dynamically build channel ranges based on wavelength instrument mapping.
    Preserves order from output_channels/wavelengths variable in file
    (otherwise uses mapping order).

    Args:
        wavelength_mapping: list of tuples (wavelength, instrument, n_vza)
        output_channels_order: Array of wavelengths in the order they appear
            in retrieval file

    Returns:
        channel_ranges (dict): mapping wavelength strings to
            (start_idx, end_idx) tuples
        metadata (dict): mapping with metadata about total angles for
            intensity/dolp
    """
    # Create a lookup dict for the wavelength mapping
    wavelength_dict = {wl: (instrument, n_vza) for wl, instrument, n_vza in wavelength_mapping}

    if output_channels_order is not None:
        # Use order from output channels
        ordered_wavelengths = []
        for wl in output_channels_order:
            wl_int = int(wl)
            if wl_int in wavelength_dict:
                instrument, n_vza = wavelength_dict[wl_int]
                ordered_wavelengths.append((wl_int, instrument, n_vza))
            else:
                print("Warning: Wavelength {} from output_channels/wavelengths is not found in wavelength mapping. Please update.".format(wl_int))
    else:
        # Fallback to original mapping order (don't sort)
        ordered_wavelengths = wavelength_mapping

    channel_ranges = {}
    current_idx = 0

    total_intensity_angles = 0
    total_dolp_angles = 0

    for wavelength, instrument, n_vza in ordered_wavelengths:
        wl_str = str(wavelength)
        start_idx = current_idx
        end_idx = current_idx + n_vza

        channel_ranges[wl_str] = (start_idx, end_idx)

        # update totals
        total_intensity_angles += n_vza

        # OCI doesn't measure polarization, so no dolp added to count
        if instrument != 'OCI':
            total_dolp_angles += n_vza

        if debug > 1:
            print("  {} nm ({}): angles {}-{} ({} angles)".format(
                wl_str, instrument, start_idx, end_idx-1, n_vza))

        current_idx = end_idx

    # Create metadata dict
    metadata = {
        'total_intensity_angles': total_intensity_angles,
        'total_dolp_angles': total_dolp_angles,
        'wavelength_mapping': ordered_wavelengths
    }

    if debug > 1:
        print("\nTotal intensity angles: {}".format(total_intensity_angles))
        print("Total DoLP angles: {}".format(total_dolp_angles))

    return channel_ranges, metadata


# =============================================================================
# DATA I/O AND PROCESSING FUNCTIONS
# =============================================================================
def load_retrieval_file(file_path):
    """
    Wrapper function that detects file format and routes to appropriate reader.
    Use this instead of calling read_hdf5_variables or read_rsp_hdf5_variables directly.

    Args:
        file_path: (str) path to the retrieval file

    Returns:
        data_dict (dict):
        sorted_variables (list):
        display_names (dict):
        variable_metadata (dict):
    """
    file_format = detect_file_format(file_path)

    if file_format == 'RSP':
        return read_rsp_hdf5_variables(file_path)
    else:  # HARP2 or default
        return read_hdf5_variables(file_path)


def read_hdf5_variables(file_path):
    """
    Read variables from HDF5 file and return dictionaries of datasets
    and a list of available variables for the dropdown menu.

    Args:
        directory_path: (str) path to directory containing retrieval files

    Returns:
        data_dict (dict):
        sorted_variables:
        display_names:
        variable_metadata:
    """
    excluded_var_strings = ['total_iterations', 'best_iteration', 'data_cf',
                            '_unc', 'prior_cf', 'state_vector', 'total_cf',
                            ]

    try:
        # Open the HDF5 file
        with h5py.File(file_path, 'r') as f:
            # Initialize dictionary to store variables and arrays for plotting
            data_dict = {}

            # Place the file_path in the dictionary (needed later when
            # extracting intensity/dolp when different sensors are used)
            data_dict['file_path'] = file_path

            # First get lat/lon which are our base reference dimensions
            if 'geolocation_data/latitude' in f:
                lat = f['geolocation_data/latitude'][:]
                lon = f['geolocation_data/longitude'][:]

                # store lat/lon arrays
                data_dict['latitude'] = lat
                data_dict['longitude'] = lon
                base_length = len(lat)

                # store original shapes for reference
                original_shape = lat.shape
                data_dict['original_shape'] = original_shape
                if debug > 1:
                    print(f"Base dimension length: {base_length}")

                # check for NaN values although shouldn't be any in lat/lon
                lat_nan_count = np.isnan(lat).sum()
                lon_nan_count = np.isnan(lon).sum()
                if lat_nan_count > 0 or lon_nan_count > 0 and debug > 1:
                    print(f"Warning: Found {lat_nan_count} NaN values in lat and {lon_nan_count} in lon")
            else:
                raise ValueError("Latitude/longitude data not found in HDF5 file")

            # Get viewing zenith angles
            if 'geolocation_data/vza' in f:
                sensor_zenith = f['geolocation_data/vza'][:]
                data_dict['sensor_zenith'] = sensor_zenith

                # check for NaN values
                nan_count = np.isnan(sensor_zenith).sum()
                if nan_count > 0 and debug > 1:
                    print(f"Warning: Found {nan_count} NaN values in sensor_zenith")

            # Get relative azimuth angles
            if 'geolocation_data/raa' in f:
                raa = f['geolocation_data/raa'][:]
                data_dict['raa'] = raa

            # Get solar zenith angles if available
            if 'geolocation_data/sza' in f:
                sza = f['geolocation_data/sza'][:]
                data_dict['sza'] = sza

            # Get measurement and model vectors for intensity and DoLP
            if 'geophysical_data/ymvec' in f:
                ymvec = f['geophysical_data/ymvec'][:]
                data_dict['ymvec'] = ymvec

            if 'geophysical_data/fvec' in f:
                fvec = f['geophysical_data/fvec'][:]
                data_dict['fvec'] = fvec

            # Get cost function if available
            if 'geophysical_data/normalized_data_cf' in f:
                data_dict['cost_function'] = f['geophysical_data/normalized_data_cf'][:]
                data_dict['cost_function_data'] = f['geophysical_data/data_cf'][:]
            else:
                # Create a placeholder cost function
                print("Cost function not found, creating placeholder")
                data_dict['cost_function'] = np.random.uniform(0, 10, base_length)

            # Get wavelength information
            wavelengths = None
            if 'wavelength' in f:
                wavelengths = f['wavelength'][:]
                data_dict['wavelengths'] = wavelengths
                # Set reference wavelength for AOD visualization (first wavelength)
                data_dict['reference_wavelength'] = float(wavelengths[0])
                if debug > 1:
                    print(f"HARP2 AOD reference wavelength: {wavelengths[0]} nm")

            # Get output channels information
            data_dict['output_channels'] = wavelengths

            # Find all arrays in the geophysical_data group
            available_variables = []
            variable_metadata = {}  # Store additional info about variables
            skipped_variables = []

            # Scan through all datasets in the geophysical_data group
            for key in f['geophysical_data'].keys():

                # Exclude variables we don't want in data_dict
                if any(exclude in key for exclude in excluded_var_strings):
                    continue

                # skip these because we already have them
                if key in ['ymvec', 'fvec', '_cf']:
                    continue

                dataset = f['geophysical_data/' + key]
                if debug > 1:
                    print(f"Processing {key} with shape {dataset.shape}")

                # Read the data
                data_array = dataset[:]

                # Check for NaN values
                nan_count = np.isnan(data_array).sum()
                if nan_count > 0 and debug > 1:
                    print(f"Warning: Found {nan_count} NaN values in {key}")

                # Store the original data with original shape
                data_dict[f"{key}_2d"] = data_array

                # Also store flattened version for compatibility with existing code
                if data_array.ndim == 2 and data_array.shape == original_shape:
                    data_dict[key] = data_array.flatten()
                    available_variables.append(key)

                    # Check if spectral variable
                    parts = key.split('_')
                    if len(parts) > 0 and parts[-1].isdigit():
                        wl = int(parts[-1])
                        base_name = '_'.join(parts[:-1])
                        variable_metadata[key] = {
                            'type': 'spectral',
                            'wavelength': wl,
                            'base_name': base_name
                        }
                        if debug > 1:
                            print(f"Added spectral variable {key} with wavelength {wl} nm")
                    else:
                        variable_metadata[key] = {'type': '2D'}
                        if debug > 1:
                            print(f"Added 2D variable {key}")
                # Store arrays of other dimension, but not add to available_variables
                else:
                    print(f"Variable {key} with shape {dataset.shape} stored but not in dropdown")
                    skipped_variables.append(f"{key} (shape: {dataset.shape})")

            # Detect available aerosol modes dynamically
            # Check for any wavelength - if mode exists for any wavelength, it's available
            available_modes = set()
            possible_modes = ['fine', 'coarse', 'dust', 'sea_salt']

            for mode in possible_modes:
                # Check if any variable with this mode exists
                mode_found = any(mode in key for key in available_variables)
                if mode_found:
                    available_modes.add(mode)

            # Store available modes in data_dict
            data_dict['available_modes'] = sorted(list(available_modes))
            if debug > 0:
                print(f"Available aerosol modes: {data_dict['available_modes']}")

            # Compute total AOD for each wavelength
            if wavelengths is not None:
                for wl in wavelengths:
                    # Check if we have component AODs for this wl, flattened versions first
                    fine_key = f"optical_depth_fine_{int(wl)}"
                    coarse_key = f"optical_depth_coarse_{int(wl)}"
                    dust_key = f"optical_depth_dust_{int(wl)}"
                    sea_salt_key = f"optical_depth_sea_salt_{int(wl)}"

                    # Now original 2D shape
                    fine_key_2d = f"{fine_key}_2d"
                    coarse_key_2d = f"{coarse_key}_2d"
                    dust_key_2d = f"{dust_key}_2d"
                    sea_salt_key_2d = f"{sea_salt_key}_2d"

                    has_fine = fine_key_2d in data_dict
                    has_coarse = coarse_key_2d in data_dict
                    has_dust = dust_key_2d in data_dict
                    has_sea_salt = sea_salt_key_2d in data_dict

                    if has_fine or has_coarse or has_dust or has_sea_salt:
                        # Initialize total AOD array
                        total_aod_2d = np.full(original_shape, np.nan)

                        # Mask to track data not NaN (probably an easier way)
                        valid_data_mask = np.zeros(original_shape, dtype=bool)

                        # Initialize sum with zeros
                        aod_sum = np.zeros(original_shape)

                        # Add each component (where valid data)
                        if has_fine:
                            fine_data = data_dict[fine_key_2d]
                            fine_valid = ~np.isnan(fine_data)
                            valid_data_mask |= fine_valid
                            aod_sum = np.where(fine_valid, aod_sum + np.nan_to_num(fine_data, nan=0.0), aod_sum)

                        if has_coarse:
                            coarse_data = data_dict[coarse_key_2d]
                            coarse_valid = ~np.isnan(coarse_data)
                            valid_data_mask |= coarse_valid
                            aod_sum = np.where(coarse_valid, aod_sum + np.nan_to_num(coarse_data, nan=0.0), aod_sum)

                        if has_dust:
                            dust_data = data_dict[dust_key_2d]
                            dust_valid = ~np.isnan(dust_data)
                            valid_data_mask |= dust_valid
                            aod_sum = np.where(dust_valid, aod_sum + np.nan_to_num(dust_data, nan=0.0), aod_sum)

                        if has_sea_salt:
                            sea_salt_data = data_dict[sea_salt_key_2d]
                            sea_salt_valid = ~np.isnan(sea_salt_data)
                            valid_data_mask |= sea_salt_valid
                            aod_sum = np.where(sea_salt_valid, aod_sum + np.nan_to_num(sea_salt_data, nan=0.0), aod_sum)

                        # Only assign total aod where at least one component not NaN (this should be more flexible
                        # because it assumes there could be different NaN locations between coarse, fine, etc)
                        # Everywhere else remains NaN
                        total_aod_2d = np.where(valid_data_mask, aod_sum, np.nan)

                        # Store 2D and flattened versions
                        total_key = f"optical_depth_total_{wl}"
                        data_dict[f"{total_key}_2d"] = total_aod_2d
                        data_dict[total_key] = total_aod_2d.flatten()

                        available_variables.append(total_key)
                        variable_metadata[total_key] = {
                            'type': 'spectral',
                            'wavelength': int(wl),
                            'base_name': 'optical_depth_total'
                        }
                        if debug > 2:
                            print(f"Computed total optical depth for {wl} nm")
                    else:
                        print(f"No component optical depths for {wl} nm, cannot compute aod")

            # Get display names and sort variables
            display_names = {}

            # Group variables by base name for sorting
            grouped_vars = {}
            for var in available_variables:
                metadata = variable_metadata.get(var, {'type': 'other'})
                if metadata['type'] == 'spectral':
                    base_name = metadata['base_name']
                    wl = metadata['wavelength']

                    # Clean up the name for drop down
                    display_name = base_name.replace('_', ' ').title()
                    replacements = {
                        'Ssa': 'Single Scattering Albedo',
                        'Reff': 'Effective Radius',
                        'Veff': 'Effective Variance',
                        'Fine': '(Fine Mode)',
                        'Coarse': '(Coarse Mode)',
                        'Dust': '(Dust)',
                        'Sea Sale': '(Sea Salt)',
                        'Total': '(Total)'
                    }
                    for old, new in replacements.items():
                        display_name = display_name.replace(old, new)

                    display_name = f"{display_name} - {wl} nm"
                    # display_name = base_name.replace('_', ' ').replace('fine', ' (fine mode)').replace('coarse', ' (coarse mode)').replace('dust', ' (dust)').replace('sea_salt', ' (sea salt)')
                    # display_name = f"{display_name.title()} - {wl} nm"
                    display_names[var] = display_name

                    # Group by base name
                    if base_name not in grouped_vars:
                        grouped_vars[base_name] = []
                    grouped_vars[base_name].append((var, wl))  # store with wl for sorting
                else:
                    # Standard display if any non spectral variables added
                    display_name = var.replace('_', ' ').replace('fine', ' (fine mode)').replace('coarse', ' (coarse mode)').replace('dust', ' (dust)').replace('reff', 'effective radius').replace('veff', 'effective variance')
                    display_names[var] = display_name.title()

            # Sort spectral variables by wl within each group
            sorted_variables = []

            # First add non spectral
            non_spectral = [var for var in available_variables if variable_metadata.get(var, {}).get('type') != 'spectral']
            sorted_variables.extend(sorted(non_spectral))

            # Now add spectral var by base name and wl
            for base_name in sorted(grouped_vars.keys()):
                # Sort by wl
                sorted_group = sorted(grouped_vars[base_name], key=lambda x: x[1])
                sorted_variables.extend([var for var, _ in sorted_group])

            # Update display names for total AOD
            for var in available_variables:
                if 'optical_depth_total' in var:
                    wl = var.split('_')[-1]
                    display_names[var] = f"Optical Depth (Total) - {wl} nm"

            # Store file format for later use
            data_dict['file_format'] = 'HARP2'

            return data_dict, sorted_variables, display_names, variable_metadata

    except Exception as e:
        print(f"Error reading file: {str(e)}")
        raise


def read_rsp_hdf5_variables(file_path):
    """
    Read variables from RSP HDF5 file and return dictionaries of datasets
    in the same format as read_hdf5_variables (HARP2 format).

    This function:
    - Reads RSP-specific variable names and structure
    - Converts to HARP2-compatible naming convention
    - Reshapes data to match expected formats
    - Dynamically detects available aerosol modes

    Args:
        file_path: (str) path to the RSP retrieval file

    Returns:
        data_dict (dict): Dictionary with standardized variable names
        sorted_variables (list): List of available variables for dropdown
        display_names (dict): Display names for variables
        variable_metadata (dict): Metadata about variables
    """
    excluded_var_strings = ['total_iterations', 'best_iteration', 'data_cf',
                            '_unc', 'prior_cf', 'state_vector', 'total_cf',
                            'error_covariance', 'unc_woc'
                            ]

    try:
        # Open the HDF5 file
        with h5py.File(file_path, 'r') as f:
            # Initialize dictionary to store variables and arrays for plotting
            data_dict = {}

            # Place the file_path in the dictionary
            data_dict['file_path'] = file_path

            # Read lat/lon - RSP format has shape (42, 1) that needs squeezing
            if 'lat' in f:
                lat = f['lat'][:].squeeze()  # (42, 1) -> (42,)
                lon = f['lon'][:].squeeze()  # (42, 1) -> (42,)

                # Store lat/lon arrays
                data_dict['latitude'] = lat
                data_dict['longitude'] = lon
                base_length = len(lat)

                # Store original shapes for reference (RSP is 1D)
                original_shape = lat.shape
                data_dict['original_shape'] = original_shape
                if debug > 1:
                    print(f"Base dimension length: {base_length}")

                # Check for NaN values
                lat_nan_count = np.isnan(lat).sum()
                lon_nan_count = np.isnan(lon).sum()
                if lat_nan_count > 0 or lon_nan_count > 0 and debug > 1:
                    print(f"Warning: Found {lat_nan_count} NaN values in lat and {lon_nan_count} in lon")
            else:
                raise ValueError("Latitude/longitude data not found in RSP file")

            # Get viewing zenith angles (sensor_zenith in RSP)
            if 'sensor_zenith' in f:
                sensor_zenith = f['sensor_zenith'][:]
                data_dict['sensor_zenith'] = sensor_zenith

                # Check for NaN values
                nan_count = np.isnan(sensor_zenith).sum()
                if nan_count > 0 and debug > 1:
                    print(f"Warning: Found {nan_count} NaN values in sensor_zenith")

            # Get relative azimuth angles
            if 'raa' in f:
                raa = f['raa'][:]
                data_dict['raa'] = raa

            # Get solar zenith angles
            if 'sza' in f:
                sza = f['sza'][:]
                data_dict['sza'] = sza

            # Get measurement and model vectors for intensity and DoLP
            if 'ymvec' in f:
                ymvec = f['ymvec'][:]
                data_dict['ymvec'] = ymvec

            if 'fvec' in f:
                fvec = f['fvec'][:]
                data_dict['fvec'] = fvec

            # Get cost function (RSP naming)
            if 'retrieval_normalized_cost_function_data' in f:
                data_dict['cost_function'] = f['retrieval_normalized_cost_function_data'][:]
                # data_dict['cost_function_data'] = f['retrieval_normalized_cost_function_data'][:]
            else:
                # Create a placeholder cost function
                print("Cost function not found, creating placeholder")
                data_dict['cost_function'] = np.random.uniform(0, 10, base_length)

            # Get wavelength information (output_channels in RSP)
            wavelengths = None
            if 'output_channels' in f:
                wavelengths = f['output_channels'][:]
                data_dict['wavelengths'] = wavelengths
                data_dict['output_channels'] = wavelengths
                if debug > 1:
                    print(f"RSP wavelengths (output_channels): {wavelengths}")

            # Get time data for RSP (if available)
            if 'rsp_time' in f:
                data_dict['rsp_time'] = f['rsp_time'][:]
                if debug > 1:
                    print(f"RSP time data loaded: shape {data_dict['rsp_time'].shape}")

            # Now read all root-level datasets and convert to HARP2 naming
            available_variables = []
            variable_metadata = {}
            skipped_variables = []

            # Get all keys at root level
            root_keys = [k for k in f.keys()]

            # Process each dataset at root level
            for key in root_keys:
                # Skip groups
                if isinstance(f[key], h5py.Group):
                    continue

                # Skip variables we already processed
                if key in ['lat', 'lon', 'sensor_zenith', 'raa', 'sza', 'ymvec', 'fvec',
                           'retrieval_normalized_cost_function_data',
                           'retrieval_normalized_cost_function_total',
                           'output_channels', 'rsp_time']:
                    continue

                # Skip excluded variables
                if any(exclude in key for exclude in excluded_var_strings):
                    continue

                # Only process aerosol-related variables
                if not key.startswith('aerosol_'):
                    continue

                dataset = f[key]
                if debug > 1:
                    print(f"Processing RSP variable: {key} with shape {dataset.shape}")

                # Read the data
                data_array = dataset[:]

                # Convert RSP variable name to HARP2 format
                # Remove "aerosol_" prefix
                harp2_name = key.replace('aerosol_', '')

                # Check for NaN values
                nan_count = np.isnan(data_array).sum()
                if nan_count > 0 and debug > 1:
                    print(f"Warning: Found {nan_count} NaN values in {key}")

                # Handle different data shapes
                if data_array.ndim == 1 and data_array.shape[0] == base_length:
                    # 1D array matching spatial dimension - store as is and as _2d
                    data_dict[harp2_name] = data_array
                    data_dict[f"{harp2_name}_2d"] = data_array  # RSP 1D = HARP2 "2D" flattened
                    available_variables.append(harp2_name)

                    # Check if spectral variable (ends with wavelength number)
                    parts = harp2_name.split('_')
                    if len(parts) > 0 and parts[-1].isdigit():
                        wl = int(parts[-1])
                        base_name = '_'.join(parts[:-1])
                        variable_metadata[harp2_name] = {
                            'type': 'spectral',
                            'wavelength': wl,
                            'base_name': base_name
                        }
                        if debug > 1:
                            print(f"Added spectral variable {harp2_name} with wavelength {wl} nm")
                    else:
                        variable_metadata[harp2_name] = {'type': '1D'}
                        if debug > 1:
                            print(f"Added 1D variable {harp2_name}")

                elif data_array.ndim == 2:
                    # 2D array (wavelength, spatial) - store but don't add to dropdown yet
                    # We'll extract specific wavelengths later if needed
                    data_dict[f"{harp2_name}_multi_wl"] = data_array
                    if debug > 1:
                        print(f"Stored multi-wavelength variable {harp2_name} with shape {data_array.shape}")
                else:
                    # Other dimensions - store but skip
                    data_dict[harp2_name] = data_array
                    skipped_variables.append(f"{harp2_name} (shape: {data_array.shape})")

            # Extract individual wavelengths from multi-wavelength 2D arrays
            # RSP stores wavelength-dependent data as (n_wavelengths, n_spatial_points)
            # with wavelengths in separate arrays like aerosol_optical_depth_wavelengths
            multi_wl_variables = [key for key in data_dict.keys() if key.endswith('_multi_wl')]

            for multi_wl_key in multi_wl_variables:
                # Get the base variable name (remove _multi_wl suffix)
                base_var_name = multi_wl_key[:-9]  # Remove '_multi_wl'

                # Look for corresponding wavelength array
                # For RSP: wavelength arrays are named like aerosol_optical_depth_wavelengths
                # (without the mode suffix like _fine, _dust)
                # Extract base property name (e.g., optical_depth from optical_depth_fine)
                parts = base_var_name.split('_')
                if len(parts) >= 2:
                    # Remove mode suffix to get property name
                    possible_modes_list = ['fine', 'coarse', 'dust', 'sea_salt']
                    if parts[-1] in possible_modes_list:
                        property_base = '_'.join(parts[:-1])
                    else:
                        property_base = base_var_name
                else:
                    property_base = base_var_name

                wl_array_names = [
                    f'aerosol_{property_base}_wavelengths',  # e.g., aerosol_optical_depth_wavelengths
                    f'{property_base}_wavelengths',
                    f'aerosol_{base_var_name}_wavelengths',  # Also try with mode
                    f'{base_var_name}_wavelengths',
                ]

                wavelengths_data = None
                for wl_name in wl_array_names:
                    if wl_name in f:
                        wavelengths_data = f[wl_name][:]
                        if debug > 1:
                            print(f"Found wavelengths for {base_var_name}: {wavelengths_data}")
                        break

                # If no wavelength array found, try to use output_channels
                if wavelengths_data is None and wavelengths is not None:
                    wavelengths_data = wavelengths
                    if debug > 1:
                        print(f"Using output_channels for {base_var_name}")

                if wavelengths_data is not None:
                    multi_wl_data = data_dict[multi_wl_key]

                    # Extract data for each wavelength
                    for wl_idx, wl_value in enumerate(wavelengths_data):
                        wl_int = int(round(wl_value))

                        # Extract the row corresponding to this wavelength
                        if wl_idx < multi_wl_data.shape[0]:
                            wl_data = multi_wl_data[wl_idx, :]

                            # Create variable name with wavelength suffix
                            var_name_with_wl = f"{base_var_name}_{wl_int}"

                            # Store the data
                            data_dict[var_name_with_wl] = wl_data
                            data_dict[f"{var_name_with_wl}_2d"] = wl_data  # RSP 1D = HARP2 "2D" flattened
                            available_variables.append(var_name_with_wl)

                            variable_metadata[var_name_with_wl] = {
                                'type': 'spectral',
                                'wavelength': wl_int,
                                'base_name': base_var_name
                            }

                            if debug > 1:
                                print(f"Extracted {var_name_with_wl} from multi-wavelength data")

            # Detect available aerosol modes dynamically
            available_modes = set()
            possible_modes = ['fine', 'coarse', 'dust', 'sea_salt']

            for mode in possible_modes:
                # Check if any variable with this mode exists
                mode_found = any(mode in key for key in available_variables)
                if mode_found:
                    available_modes.add(mode)

            # Store available modes in data_dict
            data_dict['available_modes'] = sorted(list(available_modes))
            if debug > 0:
                print(f"Available aerosol modes in RSP file: {data_dict['available_modes']}")

            # Compute total AOD for wavelengths where we have component data
            # First, find all unique wavelengths present in the optical_depth variables
            aod_wavelengths = set()
            for var in available_variables:
                if 'optical_depth' in var and var.startswith('optical_depth_'):
                    parts = var.split('_')
                    if len(parts) > 0 and parts[-1].isdigit():
                        aod_wavelengths.add(int(parts[-1]))

            if debug > 1:
                print(f"Found optical depth data for wavelengths: {sorted(aod_wavelengths)}")

            for wl in sorted(aod_wavelengths):
                # Check if we have component AODs for this wavelength
                fine_key = f"optical_depth_fine_{wl}"
                dust_key = f"optical_depth_dust_{wl}"

                fine_key_2d = f"{fine_key}_2d"
                dust_key_2d = f"{dust_key}_2d"

                has_fine = fine_key_2d in data_dict
                has_dust = dust_key_2d in data_dict

                if has_fine or has_dust:
                    # Initialize total AOD array
                    total_aod = np.full(original_shape, np.nan)

                    # Mask to track valid data
                    valid_data_mask = np.zeros(original_shape, dtype=bool)

                    # Initialize sum with zeros
                    aod_sum = np.zeros(original_shape)

                    # Add each component (where valid data)
                    if has_fine:
                        fine_data = data_dict[fine_key_2d]
                        fine_valid = ~np.isnan(fine_data)
                        valid_data_mask |= fine_valid
                        aod_sum = np.where(fine_valid, aod_sum + np.nan_to_num(fine_data, nan=0.0), aod_sum)

                    if has_dust:
                        dust_data = data_dict[dust_key_2d]
                        dust_valid = ~np.isnan(dust_data)
                        valid_data_mask |= dust_valid
                        aod_sum = np.where(dust_valid, aod_sum + np.nan_to_num(dust_data, nan=0.0), aod_sum)

                    # Only assign total aod where at least one component is not NaN
                    total_aod = np.where(valid_data_mask, aod_sum, np.nan)

                    # Store both versions
                    total_key = f"optical_depth_total_{wl}"
                    data_dict[f"{total_key}_2d"] = total_aod
                    data_dict[total_key] = total_aod  # Already 1D for RSP

                    available_variables.append(total_key)
                    variable_metadata[total_key] = {
                        'type': 'spectral',
                        'wavelength': wl,
                        'base_name': 'optical_depth_total'
                    }
                    if debug > 1:
                        print(f"Computed total optical depth for {wl} nm")
                else:
                    if debug > 1:
                        print(f"No component optical depths for {wl} nm, cannot compute total AOD")

            # Get display names and sort variables (same as HARP2)
            display_names = {}

            # Group variables by base name for sorting
            grouped_vars = {}
            for var in available_variables:
                metadata = variable_metadata.get(var, {'type': 'other'})
                if metadata['type'] == 'spectral':
                    base_name = metadata['base_name']
                    wl = metadata['wavelength']

                    # Clean up the name for dropdown
                    display_name = base_name.replace('_', ' ').title()
                    replacements = {
                        'Ssa': 'Single Scattering Albedo',
                        'Reff': 'Effective Radius',
                        'Veff': 'Effective Variance',
                        'Fine': '(Fine Mode)',
                        'Coarse': '(Coarse Mode)',
                        'Dust': '(Dust)',
                        'Sea Salt': '(Sea Salt)',
                        'Total': '(Total)'
                    }
                    for old, new in replacements.items():
                        display_name = display_name.replace(old, new)

                    display_name = f"{display_name} - {wl} nm"
                    display_names[var] = display_name

                    # Group by base name
                    if base_name not in grouped_vars:
                        grouped_vars[base_name] = []
                    grouped_vars[base_name].append((var, wl))
                else:
                    # Standard display if any non-spectral variables added
                    display_name = var.replace('_', ' ').replace('fine', ' (fine mode)').replace('dust', ' (dust)').replace('reff', 'effective radius').replace('veff', 'effective variance')
                    display_names[var] = display_name.title()

            # Sort spectral variables by wavelength within each group
            sorted_variables = []

            # First add non-spectral
            non_spectral = [var for var in available_variables if variable_metadata.get(var, {}).get('type') != 'spectral']
            sorted_variables.extend(sorted(non_spectral))

            # Now add spectral variables by base name and wavelength
            for base_name in sorted(grouped_vars.keys()):
                # Sort by wavelength
                sorted_group = sorted(grouped_vars[base_name], key=lambda x: x[1])
                sorted_variables.extend([var for var, _ in sorted_group])

            # Update display names for total AOD
            for var in available_variables:
                if 'optical_depth_total' in var:
                    wl = var.split('_')[-1]
                    display_names[var] = f"Optical Depth (Total) - {wl} nm"

            # Store file format for later use
            data_dict['file_format'] = 'RSP'

            # Set reference wavelength for AOD visualization (first wavelength in AOD wavelength array)
            if 'aerosol_optical_depth_wavelengths' in f:
                ref_wl = f['aerosol_optical_depth_wavelengths'][:]
                data_dict['reference_wavelength'] = float(ref_wl[0])
                if debug > 1:
                    print(f"RSP AOD reference wavelength: {ref_wl[0]} nm")

            if debug > 0:
                print(f"RSP reader: Loaded {len(available_variables)} variables")

            return data_dict, sorted_variables, display_names, variable_metadata

    except Exception as e:
        print(f"Error reading RSP file: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def filter_by_cost(data_dict, max_cost=None):
    """
    Filter data by cost function, updated to handle 1D and 2D arrays and NaN values
    and preserve dimension. Sets points that fail the filtering to infinity.
    NaN values are preserved (filtered out before plotting) so small edits
    can be made to allow user to see where retrieval fails.

    Args:
        data_dict:
        max_cost:

    Returns:
        filtered_dict (dict):
        original_indices:
    """
    # Get original indices for reshaping
    original_shape = data_dict.get('original_shape')
    if original_shape is None:
        raise ValueError("original_shape not found in data_dict")

    # Calculate total points based on shape dimensionality
    if len(original_shape) == 1:
        # 1D data (e.g., RSP)
        total_points = original_shape[0]
    elif len(original_shape) == 2:
        # 2D data (e.g., HARP2)
        total_points = original_shape[0] * original_shape[1]
    else:
        raise ValueError(f"Unexpected original_shape dimensionality: {len(original_shape)}D")

    # Get the cost function and ensure it matches original_shape
    if 'cost_function' in data_dict:
        cost_array = data_dict['cost_function']
        if cost_array.ndim == 1 and len(original_shape) == 2:
            # Reshape if flattened but should be 2D
            cost_array = cost_array.reshape(original_shape)
        elif cost_array.ndim == 1 and len(original_shape) == 1:
            # Already correct shape for 1D data
            pass
    else:
        raise ValueError("cost_function not found in data_dict")

    # Create cost mask
    if max_cost is None:
        # If no max cost, all finite values pass (we keep NaN: see above)
        cost_mask = ~np.isinf(cost_array)
    else:
        # Points pass if non NaN AND cost <= max_cost
        # Points that are NaN stay NaN and are filtered before plotting
        # Points that fail cost filtering set to infinity
        cost_mask = (~np.isnan(cost_array)) & (cost_array <= max_cost)

    # Create filtered dict
    filtered_dict = {}

    for key, value in data_dict.items():
        if key in ['file_path', 'original_shape', 'wavelengths', 'output_channels', 'available_modes']:
            # Keep metadata as is
            filtered_dict[key] = value

        elif isinstance(value, np.ndarray):
            # Handle arrays based on shape matching
            if value.shape == original_shape:
                # Array matches original shape (1D or 2D): apply cost filter
                filtered_value = value.copy().astype(float)
                filtered_value[~cost_mask] = np.inf
                filtered_dict[key] = filtered_value
            elif len(original_shape) == 2 and value.ndim == 3 and value.shape[:2] == original_shape:
                # 3D arrays for 2D data (ymvec, fvec, vza): set cost filtered points to inf
                filtered_value = value.copy().astype(float)
                filtered_value[~cost_mask, :] = np.inf
                filtered_dict[key] = filtered_value
            elif len(original_shape) == 1 and value.ndim == 2 and value.shape[0] == original_shape[0]:
                # 2D arrays for 1D data (e.g., RSP ymvec with shape (42, 1008))
                filtered_value = value.copy().astype(float)
                filtered_value[~cost_mask, :] = np.inf
                filtered_dict[key] = filtered_value
            elif len(original_shape) == 1 and value.ndim == 3 and value.shape[0] == original_shape[0]:
                # 3D arrays for 1D data (e.g., RSP angles with shape (42, 1, 504))
                filtered_value = value.copy().astype(float)
                filtered_value[~cost_mask, :, :] = np.inf
                filtered_dict[key] = filtered_value
            elif value.ndim == 1 and len(value) == total_points:
                # 1D arrays matching total points: reshape to original_shape and apply filter
                value_reshaped = value.reshape(original_shape).astype(float)
                value_reshaped[~cost_mask] = np.inf
                filtered_dict[key] = value_reshaped
            else:
                # Arrays with different dimension, keep as is
                filtered_dict[key] = value
        else:
            # Non array values keep as is
            filtered_dict[key] = value

    # Return original_indices (all points) for compatibility with existing code
    # The plotting function can determine valid points using np.isfinite
    original_indices = np.arange(total_points)

    # Debug
    valid_count = cost_mask.sum()
    nan_count = np.isnan(cost_array).sum()
    cost_filtered_count = total_points - valid_count - nan_count

    if debug > 1:
        print("Cost filter results:")
        print(f"  {valid_count} points passed cost filter")
        print(f"  {cost_filtered_count} points failed cost filter (set to infinity)")
        print(f"  {nan_count} points have NaN (retrieval failure)")
        print(f"  Total: {total_points} points")
        print(f"Cost range: {np.nanmin(cost_array):.3f} to {np.nanmax(cost_array):.3f}")
        if max_cost is not None:
            print(f"Max cost threshold: {max_cost:.3f}")

    return filtered_dict, original_indices


def get_channel_intensity_dolp_vza(data_dict, row_idx, col_idx):
    """
    Extract channel intensity, DoLP, and viewing zenith angle data for a
    specific point.
    New file structure per Snorre:
    -sensor_zenith shape (lat, lon, n_viewing_angles)
    -ymvec/fvec shape (lat, lon, 2*n_viewing_angles - 4)

    Args:
        data_dict: dict containing data arrays
        idx: index of point in flattened arrays (will be 2d after updating
            plotting function)

    Returns:
        intensity_data:
        dolp_data:
        wavelengths:
    """
    # Extract variables from dictionary
    ymvec = data_dict['ymvec']
    fvec = data_dict['fvec']
    vza = data_dict['sensor_zenith']
    sza = data_dict['sza']
    raa = data_dict['raa']

    # Get original 2D shape
    original_shape = data_dict['original_shape']

    if debug > 1:
        print(f"Processing point at grid position [{row_idx}, {col_idx}]")

    # Get wavel-instrument mapping and build channel ranges dynamically
    file_format = data_dict.get('file_format', 'HARP2')
    wavelength_mapping = get_wavelength_instrument_mapping(file_format)

    # Use order from output_channels or wavelengths in file
    output_channels = data_dict.get('wavelengths', None)
    channel_ranges, metadata = build_channel_ranges(wavelength_mapping, output_channels)

    total_intensity_angles = metadata['total_intensity_angles']
    total_dolp_angles = metadata['total_dolp_angles']

    # Extract data for the specific spatial location [row_idx, col_idx]
    if ymvec.ndim == 3 and ymvec.shape[:2] == original_shape:
        # 3D case: HARP2 format (lat, lon, measurements)
        # Extract measurement and model vectors for this point
        point_ymvec = ymvec[row_idx, col_idx, :]
        point_fvec = fvec[row_idx, col_idx, :]

        # Validate expected length
        expected_length = total_intensity_angles + total_dolp_angles
        if len(point_ymvec) != expected_length:
            print("Warning: Expected ymvec length {}, got {}".format(expected_length, len(point_ymvec)))

        # Separate into intensity and dolp
        ymvec_intensity = point_ymvec[:total_intensity_angles]
        ymvec_dolp = point_ymvec[total_intensity_angles:total_intensity_angles + total_dolp_angles]

        fvec_intensity = point_fvec[:total_intensity_angles]
        fvec_dolp = point_fvec[total_intensity_angles:total_intensity_angles + total_dolp_angles]

        # Check for NaN and warn
        if np.any(np.isnan(ymvec_intensity)) or np.any(np.isnan(ymvec_dolp)):
            print("Warning: NaN values found in measurement data for point [{}, {}]".format(row_idx, col_idx))
        if np.any(np.isnan(fvec_intensity)) or np.any(np.isnan(fvec_dolp)):
            print("Warning: NaN values found in model data for point [{}, {}]".format(row_idx, col_idx))
    elif ymvec.ndim == 2 and len(original_shape) == 1 and ymvec.shape[0] == original_shape[0]:
        # 2D case: RSP format (spatial_points, measurements)
        # Extract measurement and model vectors for this point
        point_ymvec = ymvec[row_idx, :]
        point_fvec = fvec[row_idx, :]

        # Validate expected length
        expected_length = total_intensity_angles + total_dolp_angles
        if len(point_ymvec) != expected_length:
            print("Warning: Expected ymvec length {}, got {}".format(expected_length, len(point_ymvec)))

        # Separate into intensity and dolp
        ymvec_intensity = point_ymvec[:total_intensity_angles]
        ymvec_dolp = point_ymvec[total_intensity_angles:total_intensity_angles + total_dolp_angles]

        fvec_intensity = point_fvec[:total_intensity_angles]
        fvec_dolp = point_fvec[total_intensity_angles:total_intensity_angles + total_dolp_angles]

        # Check for NaN and warn
        if np.any(np.isnan(ymvec_intensity)) or np.any(np.isnan(ymvec_dolp)):
            print("Warning: NaN values found in measurement data for point [{}]".format(row_idx))
        if np.any(np.isnan(fvec_intensity)) or np.any(np.isnan(fvec_dolp)):
            print("Warning: NaN values found in model data for point [{}]".format(row_idx))
    else:
        raise ValueError("Unexpected shape for ymvec: {}, expected shape starting with {}".format(
            ymvec.shape, original_shape))

    # Extract angular data for this point
    if vza.ndim == 3 and vza.shape[:2] == original_shape:
        # 3D case: HARP2 format (lat, lon, angles)
        point_vza = vza[row_idx, col_idx, :]
        point_sza = sza[row_idx, col_idx, :]
        point_raa = raa[row_idx, col_idx, :]

        # Validate expected length
        if len(point_vza) != total_intensity_angles:
            print("Warning: Expected angular data length {}, got {}".format(
                total_intensity_angles, len(point_vza)))

        # Check for NaN values
        if np.any(np.isnan(point_vza)) or np.any(np.isnan(point_sza)) or np.any(np.isnan(point_raa)):
            print("Warning: NaN values found in angular data for point [{}, {}]".format(row_idx, col_idx))
    elif vza.ndim == 3 and len(original_shape) == 1 and vza.shape[0] == original_shape[0]:
        # 3D case: RSP format (spatial_points, 1, angles)
        point_vza = vza[row_idx, 0, :]  # squeeze out the middle dimension
        point_sza = sza[row_idx, 0, :]
        point_raa = raa[row_idx, 0, :]

        # Validate expected length
        if len(point_vza) != total_intensity_angles:
            print("Warning: Expected angular data length {}, got {}".format(
                total_intensity_angles, len(point_vza)))

        # Check for NaN values
        if np.any(np.isnan(point_vza)) or np.any(np.isnan(point_sza)) or np.any(np.isnan(point_raa)):
            print("Warning: NaN values found in angular data for point [{}]".format(row_idx))
    else:
        raise ValueError("Unexpected shape for angular arrays: {}, expected shape starting with {}".format(
            vza.shape, original_shape))

    # Setup dictionaries for each channel
    ymvec_intensity_channels = {}
    fvec_intensity_channels = {}
    ymvec_dolp_channels = {}
    fvec_dolp_channels = {}
    vza_channels = {}
    sza_channels = {}
    raa_channels = {}

    # Extract data for each wavelength using dynamic ranges
    # (probably better way to do this)
    dolp_offset = 0

    for wavelength, instrument, n_vza, in wavelength_mapping:
        wl_str = str(wavelength)

        if wl_str not in channel_ranges:
            continue

        start_idx, end_idx = channel_ranges[wl_str]

        # Extract intensity for channel
        ymvec_intensity_channels[wl_str] = ymvec_intensity[start_idx:end_idx]
        fvec_intensity_channels[wl_str] = fvec_intensity[start_idx:end_idx]

        # Extract dolp for channel (skip if OCI)
        if instrument != 'OCI':
            dolp_start = dolp_offset
            dolp_end = dolp_offset + n_vza
            ymvec_dolp_channels[wl_str] = ymvec_dolp[dolp_start:dolp_end]
            fvec_dolp_channels[wl_str] = fvec_dolp[dolp_start:dolp_end]
            dolp_offset += n_vza
        else:
            # OCI channels don't have polarization
            ymvec_dolp_channels[wl_str] = np.array([np.nan])  # placeholder
            fvec_dolp_channels[wl_str] = np.array([np.nan])

        # Extract angular data for channel
        vzas = point_vza[start_idx:end_idx]
        szas = point_sza[start_idx:end_idx]
        raas = point_raa[start_idx:end_idx]

        # Stor angular data
        sza_channels[wl_str] = szas
        raa_channels[wl_str] = raas

        # Handle vza sign based on instrument (OCI only has 1 vza)
        if instrument in ['SPEX', 'HARP']:
            # ADAM: skip sign change if there are NaN values. This may need to
            # be adjusted based on NaN distribution
            if np.any(np.isnan(vzas)):
                print("Skipping VZA sign correction for {} nm ({}) due to NaN values".format(wl_str, instrument))
                vza_channels[wl_str] = vzas
            else:
                # Apply sign correction. Sort 1st
                sorted_indices = np.argsort(np.abs(vzas))
                sorted_vzas = vzas[sorted_indices]

                # Check if abs val increases monotonically (may be better way to do this)
                abs_vzas = np.abs(sorted_vzas)
                is_monotonic = np.all(np.diff(abs_vzas) >= 0)

                if is_monotonic:
                    # find where abs val starts to dec
                    abs_diffs = np.diff(np.abs(vzas))
                    sign_changes = np.where(np.diff(np.signbit(abs_diffs)))[0]

                    if len(sign_changes) > 0:
                        # Find most significant sign change
                        max_idx = np.argmax(np.abs(abs_diffs[sign_changes]))
                        turning_point = sign_changes[max_idx] + 1

                        # Make vals on left side negative
                        corrected_vzas = vzas.copy()
                        corrected_vzas[:turning_point] = -np.abs(corrected_vzas[:turning_point])
                        corrected_vzas[turning_point:] = np.abs(corrected_vzas[turning_point:])
                        vza_channels[wl_str] = corrected_vzas
                    else:
                        # if no clear turning point use middle
                        middle = len(vzas) // 2
                        corrected_vzas = vzas.copy()
                        corrected_vzas[:middle] = -np.abs(corrected_vzas[:middle])
                        corrected_vzas[middle:] = np.abs(corrected_vzas[middle:])
                        vza_channels[wl_str] = corrected_vzas
                else:
                    # If not monotonic use middle value
                    middle = len(vzas) // 2
                    corrected_vzas = vzas.copy()
                    corrected_vzas[:middle] = -np.abs(corrected_vzas[:middle])
                    corrected_vzas[middle:] = np.abs(corrected_vzas[middle:])
                    vza_channels[wl_str] = corrected_vzas
        else:  # OCI instruments
            # For OCI wavelengths, use as-is
            vza_channels[wl_str] = vzas

    # Now construct data for plotting
    wavelengths = data_dict['wavelengths']
    intensity_data = {}
    dolp_data = {}

    for wl in wavelengths:
        # convert wl to string for dict lookup
        wl_str = str(int(wl))

        # skip wl not in channel ranges (probably not necessary)
        if wl_str not in channel_ranges:
            continue

        # Collevt data for this wavel
        intensity_data[wl] = {
            'x': vza_channels[wl_str],
            'y_meas': ymvec_intensity_channels[wl_str],
            'y_model': fvec_intensity_channels[wl_str]
        }
        dolp_data[wl] = {
            'x': vza_channels[wl_str],
            'y_meas': ymvec_dolp_channels[wl_str],
            'y_model': fvec_dolp_channels[wl_str]
        }

    return intensity_data, dolp_data, wavelengths


# =============================================================================
# FORMATTING AND DISPLAY FUNCTIONS
# =============================================================================
def create_dropdown_options(sorted_variables, display_names, variable_metadata):
    """
    Create properly formatted dropdown options
    """
    # Regular options (without dropdown groups)
    regular_options = []

    # Group spectral variables by base name
    grouped_vars = {}
    for var in sorted_variables:
        metadata = variable_metadata.get(var, {'type': 'other'})

        if metadata['type'] == 'spectral':
            base_name = metadata['base_name']
            if base_name not in grouped_vars:
                grouped_vars[base_name] = []
            grouped_vars[base_name].append((var, display_names[var]))
        else:
            # Add non-spectral variables directly
            regular_options.append({'label': display_names[var], 'value': var})

    # Format the final dropdown options
    dropdown_options = regular_options.copy()

    # Add spectral variables as options with clear labels
    for base_name, vars_list in grouped_vars.items():
        for value, label in vars_list:
            dropdown_options.append({'label': label, 'value': value})

    return dropdown_options


def generate_wavelength_colors(wavelengths):
    """
    Generate a color scheme for the wavelengths.
    Maps visible wavelengths to approximate RGB colors,
    and extends to non-visible with a gradient.
    """
    colors = {}

    # Define visible spectrum range
    visible_min = 380
    visible_max = 750

    for wl in wavelengths:
        if visible_min <= wl <= visible_max:
            # Map visible wavelengths to RGB approximation
            # This is a simple approximation of wavelength to RGB
            if 380 <= wl < 440:
                r = -(wl - 440) / (440 - 380)
                g = 0.0
                b = 1.0
            elif 440 <= wl < 490:
                r = 0.0
                g = (wl - 440) / (490 - 440)
                b = 1.0
            elif 490 <= wl < 510:
                r = 0.0
                g = 1.0
                b = -(wl - 510) / (510 - 490)
            elif 510 <= wl < 580:
                r = (wl - 510) / (580 - 510)
                g = 1.0
                b = 0.0
            elif 580 <= wl < 645:
                r = 1.0
                g = -(wl - 645) / (645 - 580)
                b = 0.0
            else:  # 645 - 750
                r = 1.0
                g = 0.0
                b = 0.0
        else:
            # For non-visible wavelengths, use a gradient
            if wl < visible_min:  # UV
                h = 0.75  # Blue-violet hue
                s = 1.0
                v = 0.8 - 0.4 * (visible_min - wl) / visible_min  # Darker for shorter wavelengths
            else:  # IR
                h = 0.0  # Red hue
                s = 1.0 - 0.5 * (wl - visible_max) / visible_max  # Less saturated for longer wavelengths
                v = 0.8

            r, g, b = colorsys.hsv_to_rgb(h, s, v)

        # Convert to hex color format
        colors[wl] = f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'

    return colors


# =============================================================================
# PLOTTING AND VISUALIZATION FUNCTIONS
# =============================================================================
def create_intensity_plot_only(intensity_data, wavelengths, wl_colors, title):
    """
    Create intensity-only plot
    """
    fig = go.Figure()

    for wl in wavelengths:
        name = f'{wl} nm'

        # Add measured intensity
        fig.add_trace(go.Scatter(
            x=intensity_data[wl]['x'],
            y=intensity_data[wl]['y_meas'],
            mode='markers+lines',
            name=name,
            line=dict(color=wl_colors[wl], width=2),
            marker=dict(color=wl_colors[wl], size=6),
            legendgroup=f'wl{wl}',
            showlegend=True
        ))

        # Add modeled intensity
        fig.add_trace(go.Scatter(
            x=intensity_data[wl]['x'],
            y=intensity_data[wl]['y_model'],
            mode='lines',
            name=name,
            line=dict(color=wl_colors[wl], width=2, dash='dash'),
            legendgroup=f'wl{wl}',
            showlegend=False
        ))

    fig.update_layout(
        title=dict(
            # text=title,
            text="Intensity",
            y=0.99,
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Viewing Zenith Angle (degrees)",
        yaxis_title="Intensity",
        height=800,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.15,  # Middle of the figure (between subplots)
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",  # Semi-transparent background
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            # Force wider legend box
            itemsizing="constant",
            # itemwidth=30,
            tracegroupgap=5,
            valign="middle",
            title=dict(
                text="<b>Wavelengths </b>" + "(Solid: Measured, - - Dashed: Modeled)</b>",
                font=dict(size=14, family="Arial", color="black"),
                side="top"
            )
        ),
        margin=dict(t=120, b=50, l=60, r=50),
        autosize=True
    )

    return fig


def create_dolp_plot_only(dolp_data, wavelengths, wl_colors, title):
    """
    Create DoLP-only plot
    """
    fig = go.Figure()

    for wl in wavelengths:
        name = f'{wl} nm'

        # Add measured DoLP
        fig.add_trace(go.Scatter(
            x=dolp_data[wl]['x'],
            y=dolp_data[wl]['y_meas'],
            mode='markers+lines',
            name=name,
            line=dict(color=wl_colors[wl], width=2),
            marker=dict(color=wl_colors[wl], size=6),
            legendgroup=f'wl{wl}',
            showlegend=True
        ))

        # Add modeled DoLP
        fig.add_trace(go.Scatter(
            x=dolp_data[wl]['x'],
            y=dolp_data[wl]['y_model'],
            mode='lines',
            name=name,
            line=dict(color=wl_colors[wl], width=2, dash='dash'),
            legendgroup=f'wl{wl}',
            showlegend=False
        ))

    fig.update_layout(
        title=dict(
            # text=title,
            text="DoLP",
            y=0.99,
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Viewing Zenith Angle (degrees)",
        yaxis_title="DoLP",
        height=800,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.15,  # Middle of the figure (between subplots)
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",  # Semi-transparent background
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            # Force wider legend box
            itemsizing="constant",
            # itemwidth=30,
            tracegroupgap=5,
            valign="middle",
            title=dict(
                text="<b>Wavelengths </b>" + "(Solid: Measured, - - Dashed: Modeled)</b>",
                font=dict(size=14, family="Arial", color="black"),
                side="top"
            )
        ),
        # margin=dict(t=80, b=50, l=60, r=50),
        margin=dict(t=120, b=50, l=60, r=50),
        autosize=True
    )

    return fig


def create_placeholder_figure(message):
    """
    Create placeholder figure with a message
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=18, color="#7f8c8d")
    )
    fig.update_layout(
            height=800,
            showlegend=False,
            xaxis={'visible': False},
            yaxis={'visible': False},
            plot_bgcolor='white'
    )
    return fig


def create_initial_combined_figure():
    """
    Create initial combined figure with 'click to view' annotation
    """
    combined_fig = go.Figure()

    combined_fig.update_layout(
        height=800,
        showlegend=True,
        margin=dict(
            l=50,  # left margin
            r=40,  # right
            t=40,  # top (default ~80)
            b=40   # bottom
        ),
        autosize=True
    )

    # Add pre-click annotation
    combined_fig.add_annotation(
        text="Click a point on the map to view Intensity and DoLP plots",
        xref="x", yref="y",
        x=2.5, y=1.5,
        showarrow=False,
        font=dict(size=18, color="black"),
        xanchor='center', yanchor='middle'
    )

    return combined_fig


def create_scatter_plot_only(data_dict, selected_property, original_indices, clicked_point_data=None, max_cost=None):
    """
    Creates scatter plot only, not as a subplot like in previous
    versions. This is a simplified version of create_export_figure, but only
    creates the map portion
    """

    # Get data for plotting (same logic as before)
    property_data = data_dict[selected_property].flatten()
    finite_mask = np.isfinite(property_data)
    if finite_mask.any():
        min_val = np.min(property_data[finite_mask])
        max_val = np.max(property_data[finite_mask])
    else:
        min_val = 0
        max_val = 1

    fig = go.Figure()

    lon_flat = data_dict['longitude'].flatten()
    lat_flat = data_dict['latitude'].flatten()
    color_flat = data_dict[selected_property].flatten()
    cost_flat = data_dict['cost_function'].flatten()

    valid_mask = np.isfinite(lon_flat) & np.isfinite(lat_flat) & np.isfinite(color_flat)
    lon_valid = lon_flat[valid_mask]
    lat_valid = lat_flat[valid_mask]
    color_valid = color_flat[valid_mask]
    cost_valid = cost_flat[valid_mask]
    original_indices_valid = original_indices[valid_mask]

    # Let's clean the colorbar label up some
    wavelengths = data_dict['wavelengths']
    colorbar_title = selected_property.replace('_', ' ').title()
    for wl in wavelengths:
        wl_str = str(int(wl))
        colorbar_title = colorbar_title.replace(f'{wl_str}', f'- {wl_str} nm')

    # Final cleanup
    replacements = {
        'Ssa': 'SSA',
        'Reff': 'Effective Radius',
        'Veff': 'Effective Variance',
        'Fine': '(Fine Mode)',
        'Coarse': '(Coarse Mode)',
        'Dust': '(Dust)',
        'Sea Sale': '(Sea Salt)',
        'Total': '(Total)'
    }
    for old, new in replacements.items():
        colorbar_title = colorbar_title.replace(old, new)

    # Main scatter map
    fig.add_trace(
        go.Scattermap(
            lon=lon_valid,
            lat=lat_valid,
            mode='markers',
            marker=dict(
                size=4,
                color=color_valid,
                colorscale='Viridis',
                colorbar=dict(
                    # title=selected_property,
                    title=colorbar_title,
                    x=0.5,
                    y=0.99,
                    lenmode="fraction",
                    len=0.95,
                    orientation='h',
                    yanchor='bottom',
                    title_side='top',
                    thickness=15,
                    outlinewidth=1,
                    outlinecolor='black'
                ),
                showscale=True,
                cmin=min_val,
                cmax=max_val
            ),
            text=[f"{idx},{color_valid[i]:.3f},{cost_valid[i]:.3f}"
                  for i, idx in enumerate(original_indices_valid)],
            hovertemplate=(
                'Lat: %{lat:.2f}<br>' +
                'Lon: %{lon:.2f}<br>' +
                f'{selected_property}: %{{marker.color:.3f}}<br>' +
                'Cost: %{customdata[0]:.2f}' +
                '<extra></extra>'
            ),
            customdata=np.column_stack((cost_valid, original_indices_valid)),
            showlegend=False
        )
    )

    # Highlight selected point (updated to handle single/multi-file modes)
    if clicked_point_data is not None:
        if 'row' in clicked_point_data:
            # Single file mode format
            selected_row = clicked_point_data['row']
            selected_col = clicked_point_data['col']

            # Handle both 1D (RSP) and 2D (HARP2) indexing
            original_shape = data_dict.get('original_shape', (0,))
            if len(original_shape) == 1:
                # 1D data (RSP)
                lat = data_dict['latitude'][selected_row]
                lon = data_dict['longitude'][selected_row]
            elif len(original_shape) == 2:
                # 2D data (HARP2)
                lat = data_dict['latitude'][selected_row, selected_col]
                lon = data_dict['longitude'][selected_row, selected_col]
            else:
                # Fallback: try to use lat_flat/lon_flat
                lat = lat_flat[selected_row] if selected_row < len(lat_flat) else None
                lon = lon_flat[selected_row] if selected_row < len(lon_flat) else None

            if lat is not None and lon is not None:
                fig.add_trace(
                    go.Scattermap(
                        lon=[lon], lat=[lat],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='circle'),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
        elif 'lat' in clicked_point_data and 'lon' in clicked_point_data:
            # Multi file mode format
            fig.add_trace(
                go.Scattermap(
                    lon=[clicked_point_data['lon']], lat=[clicked_point_data['lat']],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='circle',
                    ),
                    showlegend=False,
                    text=[f"Selected: {clicked_point_data['value']:.3f}"],
                    hovertemplate="<b>Selected Point</b><br>" +
                                  f"<b>{selected_property}:</b> {clicked_point_data['value']:.3f}<br>" +
                                  f"<b>Lat:</b> {clicked_point_data['lat']:.3f}<br>" +
                                  f"<b>Lon:</b> {clicked_point_data['lon']:.3f}<br>" +
                                  "<extra></extra>",
                    name="Selected Point"
                )
            )

    center_lat = np.mean(lat_valid) if len(lat_valid) > 0 else 34
    center_lon = np.mean(lon_valid) if len(lon_valid) > 0 else -121

    fig.update_layout(
        uirevision="preserve-zoom"
    )
    fig.update_layout(
        title="",
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            # zoom=4.35
            zoom=4.0
        ),
        margin=dict(
            l=0,
            r=0,
            t=30,
            b=0
        ),
        # Remove fixed width - let it be responsive
        autosize=True
    )

    return fig


def create_combined_intensity_dolp_plot(intensity_data, dolp_data, wavelengths, wl_colors, file_format='HARP2'):
    """
    Create a single plot with intensity and DoLP as subplots
    with legend between them
    """

    DEBUG_PLOTTING = False

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Intensity vs VZA", "DoLP vs VZA"),
        vertical_spacing=0.20,  # spacing between plots
        shared_xaxes=False,
        # remove default subplot title spacing
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Add intensity/dolp traces
    for wl in wavelengths:
        name = f'{wl} nm'

        # Measured intensity
        fig.add_trace(
            go.Scatter(
                x=intensity_data[wl]['x'],
                y=intensity_data[wl]['y_meas'],
                mode='markers+lines',
                name=name,
                line=dict(color=wl_colors[wl], width=2),
                marker=dict(color=wl_colors[wl], size=6),
                legendgroup=f'wl{wl}',
                showlegend=True
            ),
            row=1, col=1
        )

        # Modeled intensity
        fig.add_trace(
            go.Scatter(
                x=intensity_data[wl]['x'],
                y=intensity_data[wl]['y_model'],
                mode='lines',
                name=f'Model {name}',
                line=dict(color=wl_colors[wl], width=2, dash='dash'),
                legendgroup=f'wl{wl}',
                showlegend=False
            ),
            row=1, col=1
        )

        # Measured DoLP
        fig.add_trace(
            go.Scatter(
                x=dolp_data[wl]['x'],
                y=dolp_data[wl]['y_meas'],
                mode='markers+lines',
                name=name,
                line=dict(color=wl_colors[wl], width=2),
                marker=dict(color=wl_colors[wl], size=6),
                legendgroup=f'wl{wl}',  # Same legendgroup!
                showlegend=False  # Don't duplicate in legend
            ),
            row=2, col=1
        )

        # Modeled DoLP
        fig.add_trace(
            go.Scatter(
                x=dolp_data[wl]['x'],
                y=dolp_data[wl]['y_model'],
                mode='lines',
                name=f'Model {name}',
                line=dict(color=wl_colors[wl], width=2, dash='dash'),
                legendgroup=f'wl{wl}',  # Same legendgroup!
                showlegend=False
            ),
            row=2, col=1
        )

    # Update layout with legend positioned in the middle
    fig.update_layout(
        height=1100,
        showlegend=True,
        # minimize overall margins
        margin=dict(
            l=50,  # left margin
            r=40,  # right
            t=40,  # top (default ~80)
            b=40  # bottom
        ),
        legend=dict(
            orientation="h",
            yanchor="middle",
            y=0.5,  # Middle of the figure (between subplots)
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",  # Semi-transparent background
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            title=dict(
                text="<b>Wavelengths </b>" + "(Solid: Measured, - - Dashed: Modeled)</b>",
                font=dict(size=14, family="Arial", color="black"),
                side="top"
            )
        ),
        autosize=True
    )

    # update subplot titles to be closer to the plots
    fig.update_annotations(
        font_size=14,
        yshift=5  # move titles slightly closer to fig (hopefully)
    )

    # minimize axis margins and update axis labels
    fig.update_xaxes(
        title_text="Viewing Zenith Angle (degrees)",
        title_standoff=8,  # reduce space btw axis and title
        row=2, col=1
    )
    fig.update_xaxes(
        title_text="Viewing Zenith Angle (degrees)",
        title_standoff=8,  # reduce space btw axis and title
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Intensity",
        title_standoff=8,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="DoLP",
        title_standoff=8,
        row=2, col=1
    )

    for wl in wavelengths:
        name = f'{wl} nm'
        # get instrument type for this wl
        wl_mapping = get_wavelength_instrument_mapping(file_format)
        instrument = None
        for w, inst, n_vza in wl_mapping:
            if w == wl:
                instrument = inst
                break
        if DEBUG_PLOTTING and instrument == 'OCI':
            print(f"\n=== PLOTTING DEBUG OCI {wl} nm ===")
            print(f"X values (VZA): {intensity_data[wl]['x']}")
            print(f"Y measured: {intensity_data[wl]['y_meas']}")
            print(f"Y modeled: {intensity_data[wl]['y_model']}")
            print(f"Number of points measured: {len(intensity_data[wl]['y_meas'])}")
            print(f"Number of points modeled: {len(intensity_data[wl]['y_model'])}")

    return fig


# Create the total aod plot
def create_aod_total_plot(data_dict, selected_row, selected_col):
    """
    Create a plot showing total AOD as a function of wavelength for the selected point
    """

    try:
        # Build wavelength mapping using actual data wavelengths and existing total AOD variables
        wavelength_aod_mapping = []

        # Look for total AOD variables that actually exist in the data
        for var_name in sorted(data_dict.keys()):
            if var_name.startswith('optical_depth_total_') and not var_name.endswith('_2d'):
                try:
                    # Extract wavelength from var name
                    wl_str = var_name.split('_')[-1]
                    wl = float(wl_str)
                    wavelength_aod_mapping.append((wl, var_name))
                except ValueError:
                    print(f"Could not extract wavelength from {var_name}")
                    continue

        # Sort by wavelength
        wavelength_aod_mapping.sort(key=lambda x: x[0])

        # Extract available wavelengths and AOD values
        wavelengths = []
        aod_values = []

        for wl, var_name in wavelength_aod_mapping:
            try:
                # Check if variable exists
                if var_name not in data_dict:
                    continue

                # Get the data - try 2D version first, then flattened
                data_array = None
                orig_shape = data_dict.get('original_shape', (0,))

                if f"{var_name}_2d" in data_dict:
                    # Use 2D version
                    data_array = data_dict[f"{var_name}_2d"]
                    # Handle both 1D (RSP) and 2D (HARP2) indexing
                    if data_array.ndim == 1:
                        # 1D data (RSP)
                        aod_value = data_array[selected_row]
                    elif data_array.ndim == 2:
                        # 2D data (HARP2)
                        aod_value = data_array[selected_row, selected_col]
                    else:
                        aod_value = np.nan
                elif var_name in data_dict:
                    # Use flattened version - need to convert indices
                    data_array = data_dict[var_name]
                    if len(orig_shape) == 1:
                        # 1D data (RSP)
                        aod_value = data_array[selected_row]
                    elif len(orig_shape) == 2:
                        # 2D data (HARP2)
                        flat_index = selected_row * orig_shape[1] + selected_col
                        aod_value = data_array[flat_index]
                    else:
                        aod_value = np.nan

                # Check if value is valid
                if np.isfinite(aod_value) and aod_value >= 0:  # Valid AOD values should be non-negative
                    wavelengths.append(wl)
                    aod_values.append(aod_value)

            except Exception as e:
                print(f"  ERROR extracting data: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Check if we have any valid data
        if not wavelengths:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid AOD data available for this point",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="#7f8c8d")
            )
            fig.update_layout(
                title="Total AOD vs Wavelength",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Aerosol Optical Depth",
                height=800,
                margin=dict(l=50, r=40, t=60, b=40)
            )
            return fig

        # Create the total AOD plot
        fig = go.Figure()

        # Add the AOD total line
        fig.add_trace(
            go.Scatter(
                x=wavelengths,
                y=aod_values,
                mode='markers+lines',
                name='Total AOD',
                line=dict(color='#2c3e50', width=3),
                marker=dict(color='#3498db', size=8, symbol='circle'),
                hovertemplate='<b>Wavelength:</b> %{x} nm<br>' +
                              '<b>AOD:</b> %{y:.4f}<br>' +
                              '<extra></extra>'
            )
        )

        # Update layout
        fig.update_layout(
            title={
                'text': f"Total AOD vs Wavelength<br><sub>Row: {selected_row}, Col: {selected_col}</sub>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': '#2c3e50'}
            },
            xaxis={
                'title': 'Wavelength (nm)',
                'title_font': {'size': 14, 'color': '#2c3e50'},
                'tickfont': {'size': 12},
                'gridcolor': '#ecf0f1',
                'showgrid': True,
                'zeroline': False
            },
            yaxis={
                'title': 'Aerosol Optical Depth',
                'title_font': {'size': 14, 'color': '#2c3e50'},
                'tickfont': {'size': 12},
                'gridcolor': '#ecf0f1',
                'showgrid': True,
                'zeroline': False
            },
            height=800,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1
            ),
            margin=dict(l=60, r=40, t=80, b=60),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified'
        )

        return fig

    except Exception as e:
        print(f"Error creating total AOD plot: {e}")
        # Return an empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating total AOD plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Total AOD Analysis - Error",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Aerosol Optical Depth",
            height=800
        )
        return fig


def create_residual_plot(data_dict, selected_row, selected_col, residual_type='both'):
    """
    Create a residual plot for intensity and/or dolp (measured - modeled)
    User can display residuals for intensity/dolp/both. However, we should add
    percent difference instead of simple difference.

    Args:
        data_dict:
        selected_row:
        selected_col:
        residual_type:
    """
    try:
        # Get the intensity and DoLP data for the selected point
        intensity_data, dolp_data, wavelengths = get_channel_intensity_dolp_vza(
            data_dict, selected_row, selected_col)

        wl_colors = generate_wavelength_colors(wavelengths)

        # Determine which type was chosen in dropdown
        if residual_type == 'both':
            # Create subplots for both intensity and DoLP residuals
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Intensity Residuals vs VZA", "DoLP Residuals vs VZA"),
                vertical_spacing=0.20,
                shared_xaxes=False,
                # remove default subplot title spacing
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )

            # Add intensity residuals
            for wl in wavelengths:
                name = f'{wl} nm'

                # calc intensity residuals
                intensity_residuals = np.array(intensity_data[wl]['y_meas']) - np.array(intensity_data[wl]['y_model'])

                fig.add_trace(
                    go.Scatter(
                        x=intensity_data[wl]['x'],
                        y=intensity_residuals,
                        mode='markers+lines',
                        name=name,
                        line=dict(color=wl_colors[wl], width=2),
                        marker=dict(color=wl_colors[wl], size=6),
                        legendgroup=f'wl{wl}',
                        showlegend=True
                    ),
                    row=1, col=1
                )

                # calc dolp residuals
                dolp_residuals = np.array(dolp_data[wl]['y_meas']) - np.array(dolp_data[wl]['y_model'])

                fig.add_trace(
                    go.Scatter(
                        x=dolp_data[wl]['x'],
                        y=dolp_residuals,
                        mode='markers+lines',
                        name=name,
                        line=dict(color=wl_colors[wl], width=2),
                        marker=dict(color=wl_colors[wl], size=6),
                        legendgroup=f'wl{wl}',
                        showlegend=False
                    ),
                    row=2, col=1
                )

            # Update layout
            fig.update_layout(
                height=1100,
                showlegend=True,
                margin=dict(l=50, r=40, t=40, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="middle",
                    y=0.5,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.3)",
                    borderwidth=1,
                    title=dict(
                        text="<b>Wavelengths</b> (Residuals = Measured - Modeled)",
                        font=dict(size=14, family="Arial", color="black"),
                        side="top"
                    )
                ),
                autosize=True
            )

            # update the axes
            fig.update_xaxes(title_text="Viewing Zenith Angle (degrees)", row=2, col=1)
            fig.update_xaxes(title_text="Viewing Zenith Angle (degrees)", row=1, col=1)
            fig.update_yaxes(title_text="Intensity Residual", row=1, col=1)
            fig.update_yaxes(title_text="DoLP Residual", row=2, col=1)

        elif residual_type == 'intensity':
            # just intensity plot
            fig = go.Figure()

            for wl in wavelengths:
                name = f'{wl} nm'

                # calc intensity residuals
                intensity_residuals = np.array(intensity_data[wl]['y_meas']) - np.array(intensity_data[wl]['y_model'])

                fig.add_trace(
                    go.Scatter(
                        x=intensity_data[wl]['x'],
                        y=intensity_residuals,
                        mode='markers+lines',
                        name=name,
                        line=dict(color=wl_colors[wl], width=2),
                        marker=dict(color=wl_colors[wl], size=6)
                    )
                )

            fig.update_layout(
                title="Intensity Residuals vs VZA",
                xaxis_title="Viewing Zenith Angle (degrees)",
                yaxis_title="Intensity Residual (Measured - Modeled)",
                height=800,
                showlegend=True,
                margin=dict(l=50, r=40, t=60, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                autosize=True
            )

        elif residual_type == 'dolp':
            # just dolp plot
            fig = go.Figure()

            for wl in wavelengths:
                name = f'{wl} nm'

                # calc dolp residual
                dolp_residuals = np.array(dolp_data[wl]['y_meas']) - np.array(dolp_data[wl]['y_model'])

                fig.add_trace(
                    go.Scatter(
                        x=dolp_data[wl]['x'],
                        y=dolp_residuals,
                        mode='markers+lines',
                        name=name,
                        line=dict(color=wl_colors[wl], width=2),
                        marker=dict(color=wl_colors[wl], size=6)
                    )
                )

            fig.update_layout(
                title="DoLP Residuals vs VZA",
                xaxis_title="Viewing Zenith Angle (degrees)",
                yaxis_title="DoLP Residual (Measured - Modeled)",
                height=800,
                showlegend=True,
                margin=dict(l=50, r=40, t=60, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                autosize=True
            )

        return fig

    except Exception as e:
        print(f"Error creating residual plot: {e}")
        # Returns an empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating residual plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Residual Analysis - Error",
            xaxis_title="Viewing Zenith Angle (degrees)",
            yaxis_title="Residual Value",
            height=800
        )
        return fig


def create_property_vs_time_plot(data_dict, property_name='optical_depth', mode='total', title_suffix="", max_cost=None, highlight_time_index=None, highlight_y_value=None):
    """
    Create a plot showing any retrieval property vs time for airborne data (RSP).
    Shows all available wavelengths for the specified property-mode combination.

    Args:
        data_dict: Dictionary containing data arrays including 'rsp_time' and property variables
        property_name: Base name of the property (e.g., 'optical_depth', 'ssa', 'reff')
        mode: Mode to plot ('total', 'fine', 'coarse', 'dust', 'sea_salt')
        title_suffix: Optional suffix to add to title (e.g., "File 1" for multi-file mode)
        max_cost: Maximum cost threshold for filtering (points above threshold are excluded)
        highlight_time_index: Optional index of point to highlight (adds visual marker)
        highlight_y_value: Optional y-value of the clicked point for single marker highlight

    Returns:
        fig: Plotly figure object
    """
    try:
        # Check if time data exists
        if 'rsp_time' not in data_dict:
            fig = go.Figure()
            fig.add_annotation(
                text="Time data (rsp_time) not available in this file",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Property vs Time - No Data Available")
            return fig

        # Get time data
        time_data = data_dict['rsp_time']
        if time_data.ndim > 1:
            time_data = time_data.flatten()

        # Get cost function data for filtering
        cost_mask = None
        if max_cost is not None and 'cost_function' in data_dict:
            cost_data = data_dict['cost_function']
            if cost_data.ndim > 1:
                cost_data = cost_data.flatten()
            # Create mask: True for points that PASS the filter (cost <= threshold)
            cost_mask = cost_data <= max_cost

        # Define property display names and units
        property_config = {
            'optical_depth': {'display_name': 'Aerosol Optical Depth', 'y_label': 'Aerosol Optical Depth', 'decimals': 5, 'has_wavelength': True},
            'ssa': {'display_name': 'Single Scattering Albedo', 'y_label': 'Single Scattering Albedo', 'decimals': 5, 'has_wavelength': True},
            'real': {'display_name': 'Real Refractive Index', 'y_label': 'Real Refractive Index', 'decimals': 5, 'has_wavelength': True},
            'imag': {'display_name': 'Imaginary Refractive Index', 'y_label': 'Imaginary Refractive Index', 'decimals': 5, 'has_wavelength': True},
            'asymmetry': {'display_name': 'Asymmetry Parameter', 'y_label': 'Asymmetry Parameter', 'decimals': 5, 'has_wavelength': True},
            'absorption_coefficient': {'display_name': 'Absorption Coefficient', 'y_label': 'Absorption Coefficient', 'decimals': 5, 'has_wavelength': True},
            'scattering_coefficient': {'display_name': 'Scattering Coefficient', 'y_label': 'Scattering Coefficient', 'decimals': 5, 'has_wavelength': True},
            'extinction_coefficient': {'display_name': 'Extinction Coefficient', 'y_label': 'Extinction Coefficient', 'decimals': 5, 'has_wavelength': True},
            'reff': {'display_name': 'Effective Radius', 'y_label': 'Effective Radius (μm)', 'decimals': 3, 'has_wavelength': False},
            'veff': {'display_name': 'Effective Variance', 'y_label': 'Effective Variance', 'decimals': 3, 'has_wavelength': False},
            'number_concentration': {'display_name': 'Number Concentration', 'y_label': 'Number Concentration', 'decimals': 5, 'has_wavelength': True},
            'cross_section': {'display_name': 'Cross Section', 'y_label': 'Cross Section', 'decimals': 5, 'has_wavelength': True}
        }

        # Get property configuration
        prop_info = property_config.get(property_name, {
            'display_name': property_name.replace('_', ' ').title(),
            'y_label': property_name.replace('_', ' ').title(),
            'decimals': 5,
            'has_wavelength': True  # Default to wavelength-dependent
        })

        has_wavelength = prop_info['has_wavelength']

        # Create figure
        fig = go.Figure()
        any_valid_data = False

        # Find all variables for this property-mode combination
        wavelength_mapping = []  # List of (wavelength, var_name) tuples

        if has_wavelength:
            # Wavelength-dependent: look for {property}_{mode}_{wavelength}
            search_pattern = f'{property_name}_{mode}_'
            for var_name in sorted(data_dict.keys()):
                if var_name.startswith(search_pattern) and not var_name.endswith('_2d'):
                    parts = var_name.split('_')
                    if len(parts) >= 3 and parts[-1].isdigit():
                        try:
                            wl = float(parts[-1])
                            wavelength_mapping.append((wl, var_name))
                        except ValueError:
                            continue

            if not wavelength_mapping:
                # No data found
                fig.add_annotation(
                    text=f"No {prop_info['display_name']} data available for {mode} mode",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(title=f"{prop_info['display_name']} vs Time - No Data")
                return fig

            # Sort by wavelength
            wavelength_mapping.sort(key=lambda x: x[0])

            # Generate colors for different wavelengths
            wavelengths = [wl for wl, _ in wavelength_mapping]
            wl_colors = generate_wavelength_colors(wavelengths)

            # Plot each wavelength
            for wl, var_name in wavelength_mapping:
                property_data = data_dict.get(var_name)
                if property_data is None:
                    continue

                # Flatten if needed
                if property_data.ndim > 1:
                    property_data = property_data.flatten()

                # Ensure same length
                min_len = min(len(time_data), len(property_data))
                time_subset = time_data[:min_len].copy()
                property_subset = property_data[:min_len].copy()

                # Create index array to preserve mapping through cost filtering
                original_indices = np.arange(min_len)

                # Apply cost filter
                if cost_mask is not None:
                    cost_subset = cost_mask[:min_len]
                    property_subset[~cost_subset] = np.nan
                    # Note: original_indices stays unchanged - this is key!

                # Check if we have any valid data
                valid_mask = np.isfinite(time_subset) & np.isfinite(property_subset)
                if not np.any(valid_mask):
                    continue

                any_valid_data = True

                # Convert to list of lists for JSON serialization
                customdata_list = [[int(idx)] for idx in original_indices]

                # Add trace
                fig.add_trace(go.Scatter(
                    x=time_subset,
                    y=property_subset,
                    mode='lines+markers',
                    name=f'{int(wl)} nm',
                    line=dict(color=wl_colors.get(wl, '#000000'), width=2),
                    marker=dict(size=8),
                    connectgaps=False,
                    customdata=customdata_list,
                    hovertemplate=f'<b>Time:</b> %{{x:.2f}} UTC<br><b>Wavelength:</b> {int(wl)} nm<br><b>{prop_info["display_name"]}:</b> %{{y:.{prop_info["decimals"]}f}}<extra></extra>'
                ))

        else:
            # Wavelength-independent: look for {property}_{mode}
            var_name = f'{property_name}_{mode}'

            if var_name in data_dict:
                property_data = data_dict[var_name]

                # Flatten if needed
                if property_data.ndim > 1:
                    property_data = property_data.flatten()

                # Ensure same length
                min_len = min(len(time_data), len(property_data))
                time_subset = time_data[:min_len].copy()
                property_subset = property_data[:min_len].copy()

                # Create index array to preserve mapping through cost filtering
                original_indices = np.arange(min_len)

                # Apply cost filter
                if cost_mask is not None:
                    cost_subset = cost_mask[:min_len]
                    property_subset[~cost_subset] = np.nan
                    # Note: original_indices stays unchanged - this is key!

                # Check if we have any valid data
                valid_mask = np.isfinite(time_subset) & np.isfinite(property_subset)
                if np.any(valid_mask):
                    any_valid_data = True

                    # Get mode color
                    mode_colors = get_mode_colors()

                    # Debug: verify customdata
                    print(f"Adding trace with customdata shape: {original_indices.shape}, range: {original_indices.min()}-{original_indices.max()}")

                    # Convert to list of lists for JSON serialization
                    customdata_list = [[int(idx)] for idx in original_indices]
                    print(f"First 3 customdata entries: {customdata_list[:3]}")

                    # Add trace
                    fig.add_trace(go.Scatter(
                        x=time_subset,
                        y=property_subset,
                        mode='lines+markers',
                        name=f'{mode.capitalize()} Mode',
                        line=dict(color=mode_colors.get(mode, '#000000'), width=2),
                        marker=dict(size=8),
                        connectgaps=False,
                        customdata=customdata_list,
                        hovertemplate=f'<b>Time:</b> %{{x:.2f}} UTC<br><b>Mode:</b> {mode.capitalize()}<br><b>{prop_info["display_name"]}:</b> %{{y:.{prop_info["decimals"]}f}}<extra></extra>'
                    ))

            if not any_valid_data:
                # No data found
                fig.add_annotation(
                    text=f"No {prop_info['display_name']} data available for {mode} mode",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(title=f"{prop_info['display_name']} vs Time - No Data")
                return fig

        # Check if no data passed the filter
        if not any_valid_data and cost_mask is not None:
            min_cost = None
            if 'cost_function' in data_dict:
                cost_data = data_dict['cost_function']
                if cost_data.ndim > 1:
                    cost_data = cost_data.flatten()
                valid_costs = cost_data[np.isfinite(cost_data)]
                if len(valid_costs) > 0:
                    min_cost = np.min(valid_costs)

            fig.add_annotation(
                text=f"No data passes current cost filter (threshold: {max_cost:.3f})",
                xref="paper", yref="paper",
                x=0.5, y=0.6,
                showarrow=False,
                font=dict(size=16, color="#e74c3c"),
                xanchor='center'
            )

            if min_cost is not None:
                fig.add_annotation(
                    text=f"Minimum cost in this file: {min_cost:.3f}<br>Try increasing the cost threshold to at least {min_cost:.3f}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.4,
                    showarrow=False,
                    font=dict(size=14, color="#7f8c8d"),
                    xanchor='center'
                )

        # Build title
        mode_label = "Total" if mode == 'total' else f"{mode.capitalize()} Mode"
        if has_wavelength:
            plot_title = f"{prop_info['display_name']} ({mode_label}) vs Time (Flight Path)"
            legend_title = "Wavelength"
        else:
            plot_title = f"{prop_info['display_name']} - {mode_label} vs Time (Flight Path)"
            legend_title = ""

        if title_suffix:
            plot_title = f"{plot_title}<br>{title_suffix}"

        # Update layout
        fig.update_layout(
            title=plot_title,
            xaxis_title="Time (UTC hours)",
            yaxis_title=prop_info['y_label'],
            yaxis=dict(
                exponentformat='none',
                tickformat=f'.{prop_info["decimals"]}f'
            ),
            height=800,
            hovermode='closest',  # Show only the hovered point, not all traces at that x-position
            showlegend=True if has_wavelength else False,
            legend=dict(
                title=legend_title,
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="Black",
                borderwidth=1
            ) if has_wavelength else None,
            uirevision='time-plot'  # Preserve UI state (trace visibility, zoom) across updates
        )

        # Add single highlight marker if a point is selected
        if highlight_time_index is not None and highlight_y_value is not None and any_valid_data:
            try:
                # Get the time value at the highlighted index
                time_data = data_dict['rsp_time']
                if time_data.ndim > 1:
                    time_data = time_data.flatten()

                if highlight_time_index < len(time_data):
                    highlight_time = time_data[highlight_time_index]

                    # Add single highlight marker at the clicked point
                    fig.add_trace(go.Scatter(
                        x=[highlight_time],
                        y=[highlight_y_value],
                        mode='markers',
                        marker=dict(
                            size=16,
                            color='red',
                            symbol='circle',
                            line=dict(color='white', width=2)
                        ),
                        name='Selected Point',
                        showlegend=True,
                        hovertemplate='<b>Selected</b><br>Time: %{x:.2f} UTC<br>Value: %{y:.5f}<extra></extra>'
                    ))
            except Exception as e:
                print(f"Warning: Could not add highlight marker: {e}")
                import traceback
                traceback.print_exc()

        return fig

    except Exception as e:
        print(f"Error creating property vs time plot: {e}")
        import traceback
        traceback.print_exc()

        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating property vs time plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color='red')
        )
        fig.update_layout(title="Property vs Time - Error")
        return fig
def create_polarized_reflectance_comparison_plot(intensity_data_1, dolp_data_1,
                                                 intensity_data_2, dolp_data_2,
                                                 wavelengths, wl_colors,
                                                 file_path_1, file_path_2,
                                                 difference_type='simple'):
    """
    Create a comparison plot showing polarized reflectance from two different
    files. Compares measured data from both files
    """

    # Extract shortened filenames for title and legend
    file1_name = file_path_1.split('/')[-1].replace('.h5', '').replace('.nc', '') if file_path_1 else 'File 1'
    file2_name = file_path_2.split('/')[-1].replace('.h5', '').replace('.nc', '') if file_path_2 else 'File 2'

    fig = go.Figure()

    # Calculate/plot polarized reflectance diff for each wavelength
    # from both files (modeled and measured)
    for wl in wavelengths:
        name = f'{wl} nm'

        # Check if both files have data for this wavelength
        if (wl in intensity_data_1 and wl in dolp_data_1 and wl in intensity_data_2 and wl in dolp_data_2):

            # Begin with MEASURED data
            # FILE 1 - Calc polarized reflectance (measured)
            min_len_1 = min(len(intensity_data_1[wl]['y_meas']), len(dolp_data_1[wl]['y_meas']))
            x_1 = intensity_data_1[wl]['x'][:min_len_1]
            polarized_refl_1 = (
                np.array(intensity_data_1[wl]['y_meas'][:min_len_1]) *
                np.array(dolp_data_1[wl]['y_meas'][:min_len_1])
            )

            # FILE 2 - Calc polarized reflectance (measured)
            min_len_2 = min(len(intensity_data_2[wl]['y_meas']), len(dolp_data_2[wl]['y_meas']))
            # x_2 = intensity_data_2[wl]['x'][:min_len_2]
            polarized_refl_2 = (
                np.array(intensity_data_2[wl]['y_meas'][:min_len_2]) *
                np.array(dolp_data_2[wl]['y_meas'][:min_len_2])
            )

            # Calculate refl difference based on method (measured)
            # Need to match the arrays: (probably bettter way to do this)
            #   use the shorter length and corresponding x values
            min_common_len = min(len(polarized_refl_1), len(polarized_refl_2))
            x_common = x_1[:min_common_len]  # Use x values from file 1
            file1_values = polarized_refl_1[:min_common_len]
            file2_values = polarized_refl_2[:min_common_len]

            if difference_type == 'percent':
                # Percent difference: ((File1 - File2) / File2) * 100
                # Handle division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    difference = np.where(file2_values != 0,
                                          ((file1_values - file2_values) / file2_values) * 100,
                                          np.nan)  # set to NaN where File2 is zero
            else:
                # Simple difference: file1 - File2
                # difference = polarized_refl_1[:min_common_len] - polarized_refl_2[:min_common_len]
                difference = file1_values - file2_values

            # Add difference trace (measured)
            fig.add_trace(
                go.Scatter(
                    x=x_common,
                    y=difference,
                    mode='markers+lines',
                    name=f'{name} (measured)',
                    line=dict(color=wl_colors[wl], width=2),
                    marker=dict(color=wl_colors[wl], size=6),
                    legendgroup=f'wl{wl}',
                    showlegend=True
                )
            )

            # Now with MODELED data
            # FILE 1 - Calc polarized reflectance (modeled)
            min_len_1 = min(len(intensity_data_1[wl]['y_model']), len(dolp_data_1[wl]['y_model']))
            x_1 = intensity_data_1[wl]['x'][:min_len_1]
            polarized_refl_1 = (
                np.array(intensity_data_1[wl]['y_model'][:min_len_1]) *
                np.array(dolp_data_1[wl]['y_model'][:min_len_1])
            )

            # FILE 2 - Calculate polarized reflectance (modeled data)
            min_len_2 = min(len(intensity_data_2[wl]['y_model']), len(dolp_data_2[wl]['y_model']))
            # x_2 = intensity_data_2[wl]['x'][:min_len_2]
            polarized_refl_2 = (
                np.array(intensity_data_2[wl]['y_model'][:min_len_2]) *
                np.array(dolp_data_2[wl]['y_model'][:min_len_2])
            )

            # Calculate refl difference (modeled)
            # Need to match the arrays - use the shorter length and corresponding x values
            min_common_len = min(len(polarized_refl_1), len(polarized_refl_2))
            x_common = x_1[:min_common_len]  # Use x values from file 1
            file1_values = polarized_refl_1[:min_common_len]
            file2_values = polarized_refl_2[:min_common_len]

            if difference_type == 'percent':
                # Percent difference: ((File1 - File2) / File2) * 100
                # Handle division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    difference = np.where(file2_values != 0,
                                          ((file1_values - file2_values) / file2_values) * 100,
                                          np.nan)  # set to NaN where File2 is zero
            else:
                # Simple difference: file1 - File2
                # difference = polarized_refl_1[:min_common_len] - polarized_refl_2[:min_common_len]
                difference = file1_values - file2_values

            # Add difference trace (modeled)
            fig.add_trace(
                go.Scatter(
                    x=x_common,
                    y=difference,
                    mode='markers+lines',
                    name=f'{name} (modeled)',
                    line=dict(color=wl_colors[wl], width=2, dash='dash'),
                    marker=dict(color=wl_colors[wl], size=6),
                    legendgroup=f'wl{wl}',
                    showlegend=True
                )
            )

    # Add horizontal line at 0 for reference
    if len(fig.data) > 0:  # only if we have data
        x_range = [min([min(trace.x) for trace in fig.data]),
                   max([max(trace.x) for trace in fig.data])]
        fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[0, 0],
                    mode='lines',
                    name='Zero Reference',
                    line=dict(color='black', width=1, dash='dot'),
                    showlegend=False
                )
        )

    # Set up labels based on difference type
    if difference_type == 'percent':
        title_prefix = "Polarized Reflectance % Difference"
        y_label = "Percent Difference [(File1 - File2) / File2 — 100]"
        legend_text = f"<b>Percent Difference</b><br>({file1_name} - {file2_name})<br>/ {file2_name} — 100<br><br><b>Wavelengths</b>"
    else:
        title_prefix = "Polarized Reflectance Difference"
        y_label = "Polarized Reflectance Difference"
        legend_text = f"<b>Difference</b><br>{file1_name}<br>minus<br>{file2_name}<br><br><b>Wavelengths</b>"

    # Update layout
    fig.update_layout(
        # title=f"Polarized Reflectance Comparison: {file1_name} vs {file2_name}",
        # title=f"Polarized Reflectance ({file1_name} vs {file2_name}",
        # title=f"{title_prefix}<br><b> File 1:</b> {file1_name}<br><b> File 2:</b> {file2_name}<br>",
        title=f"{title_prefix}:<br><b>{file1_name}</b> vs <b>{file2_name}</b>",
        xaxis_title="Viewing Zenith Angle (degrees)",
        # yaxis_title="Polarized Reflectance Difference (File 1 - File 2)",
        yaxis_title=y_label,
        height=800,
        showlegend=True,
        margin=dict(l=50, r=40, t=60, b=40),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            title=dict(
                # text=f"<b>Solid: {file1_name}</b><br><b>- - Dashed: {file2_name}</b><br><br><b>Wavelengths</b>",
                # text=f"<b>Wavelengths</b><br>({file1_name}, - - {file2_name})",
                font=dict(size=12, family="Arial", color="black"),
                side="top"
            )
        ),
        autosize=True
    )

    return fig


# Create AOD Histogram
def create_aod_histogram(data_dict, selected_property, max_cost, bin_size=0.1):
    """
    Create histogram of AOD values for selected wavelength with cost filtering
    bin_size: Size of histogram bins (set at 0.1)
    """

    # Filter data by cost function first
    filtered_data, original_indices = filter_by_cost(data_dict, max_cost)

    # Get the AOD data for the selected property
    aod_data = filtered_data[selected_property].flatten()

    # Remove invalid values
    valid_mask = np.isfinite(aod_data)
    aod_valid = aod_data[valid_mask]

    if len(aod_valid) == 0:
        # Return empty figure if there's no valid data
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data for selected property and cost filter",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="AOD Frequency Histogram",
            xaxis_title="AOD Value",
            yaxis_title="Frequency",
            height=500
        )
        return fig

    # Calculate bins
    min_val = np.floor(np.min(aod_valid) / bin_size) * bin_size
    max_val = np.ceil(np.max(aod_valid) / bin_size) * bin_size
    bins = np.arange(min_val, max_val + bin_size, bin_size)

    # Compute the histogram
    counts, bin_edges = np.histogram(aod_valid, bins=bins)

    # Create bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fix property name to make title
    wavelengths = data_dict['wavelengths']
    title_property = selected_property.replace('_', ' ').title()
    for wl in wavelengths:
        wl_str = str(int(wl))
        title_property = title_property.replace(f'{wl_str}', f'{wl_str} nm')

    # More cleanup
    replacements = {
        'Optical Depth': 'AOD',
        'Fine': '(Fine Mode)',
        'Coarse': '(Coarse Mode)',
        'Total': '(Total)'
    }
    for old, new in replacements.items():
        title_property = title_property.replace(old, new)

    # Create the histogram plot
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=counts,
            width=bin_size,
            marker=dict(
                color='steelblue',
                opacity=0.7,
                line=dict(color='darkblue', width=1)
            ),
            hovertemplate=(
                'AOD Range: %{x:.2f} - %{customdata:.2f}<br>' +
                'Frequency: %{y}<br>' +
                '<extra></extra>'
            ),
            customdata=bin_centers + bin_size/2
        )
    )

    # Add statistics summary
    stats_text = (
        f"Total Points: {len(aod_valid)}<br>" +
        f"Mean: {np.mean(aod_valid):.3f}<br>" +
        f"Std: {np.std(aod_valid):.3f}<br>" +
        f"Min: {np.min(aod_valid):.3f}<br>" +
        f"Max: {np.max(aod_valid):.3f}"
    )

    fig.add_annotation(
        text=stats_text,
        x=0.98, y=0.98,
        xref="paper", yref="paper",
        xanchor="right", yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=10)
    )

    fig.update_layout(
        title=f"AOD Frequency Histogram: {title_property}<br><sub>Cost Filter: {max_cost:.2f}</sub>",
        xaxis_title="AOD Value",
        yaxis_title="Frequency",
        height=500,
        margin=dict(l=60, r=20, t=80, b=60),
        showlegend=False
    )

    return fig


def create_polarized_reflectance_plot(intensity_data, dolp_data, wavelengths, wl_colors):
    """
    Create a plot showing polarized reflectance (DoLP — Intensity) vs VZA
    with both measured and modeled data
    """

    DEBUG_PLOTTING = False

    # Create single plot
    fig = go.Figure()

    # Calculate polarized reflectance for each wavelength
    for wl in wavelengths:
        name = f'{wl} nm'

        # Calculate polarized reflectance - measured data
        min_len_meas = min(len(intensity_data[wl]['y_meas']), len(dolp_data[wl]['y_meas']))
        x_meas = intensity_data[wl]['x'][:min_len_meas]
        polarized_refl_meas = (
            np.array(intensity_data[wl]['y_meas'][:min_len_meas]) *
            np.array(dolp_data[wl]['y_meas'][:min_len_meas])
        )

        # Calculate polarized reflectance - modeled data
        min_len_model = min(len(intensity_data[wl]['y_model']), len(dolp_data[wl]['y_model']))
        x_model = intensity_data[wl]['x'][:min_len_model]
        polarized_refl_model = (
            np.array(intensity_data[wl]['y_model'][:min_len_model]) *
            np.array(dolp_data[wl]['y_model'][:min_len_model])
        )

        # Add measured polarized reflectance trace
        fig.add_trace(
            go.Scatter(
                x=x_meas,
                y=polarized_refl_meas,
                mode='markers+lines',
                name=f'Measured {name}',
                line=dict(color=wl_colors[wl], width=2),
                marker=dict(color=wl_colors[wl], size=6),
                legendgroup=f'wl{wl}',
                showlegend=True
            )
        )

        # Add modeled polarized reflectance trace
        fig.add_trace(
            go.Scatter(
                x=x_model,
                y=polarized_refl_model,
                mode='lines',
                name=f'Model {name}',
                line=dict(color=wl_colors[wl], width=2, dash='dash'),
                legendgroup=f'wl{wl}',
                showlegend=True
            )
        )

    # Update layout
    fig.update_layout(
        title="Polarized Reflectance (DoLP — Intensity) vs Viewing Zenith Angle",
        xaxis_title="Viewing Zenith Angle (degrees)",
        yaxis_title="Polarized Reflectance (DoLP — Intensity)",
        height=800,
        showlegend=True,
        margin=dict(l=50, r=40, t=60, b=40),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            title=dict(
                text="<b>Wavelengths</b><br>(Solid: Measured, - - Dashed: Modeled)",
                font=dict(size=12, family="Arial", color="black"),
                side="top"
            )
        ),
        autosize=True
    )

    return fig


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================
def create_export_figure(data_dict, selected_property, original_indices,
                         clicked_point_data=None, max_cost=None):
    """
    Function to create the scatter plot. NaN values are where the retrieval
    was not completed (for whatever reason) and inf values are where the cost
    function filtering fails. We currently remove both before plotting.

    Note: data_dict is intended to be filtered_dict, since the cost function
    filtering has already been done.
    """
    # DEBUG: Check what we received
    if debug > 1:
        print("=== PLOTTING DEBUG ===")
        print(f"DEBUG: clicked_point_data type: {type(clicked_point_data)}")
        print(f"DEBUG: clicked_point_data value: {clicked_point_data}")
        print(f"data_dict[longitude] shape: {data_dict['longitude'].shape}")
        print(f"data_dict[latitude] shape: {data_dict['latitude'].shape}")
        print(f"data_dict[{selected_property}] shape: {data_dict[selected_property].shape}")

    # Make sure selected property exists in data_dict, if not choose 1
    if selected_property not in data_dict:
        # Default to 1st available property that isn't lat/lon/cost functiomn
        for key in data_dict.keys():
            if key not in ['latitude', 'longitude', 'cost_function',
                           'wavelengths', 'sensor_zenith', 'raa', 'saa',
                           'ymvec', 'fvec']:
                selected_property = key
                break

    # Flatten the selected property for ease of plotting
    property_data = data_dict[selected_property].flatten()

    # Get min/max vals for colorbar, excluding nan and inf
    finite_mask = np.isfinite(property_data)
    if finite_mask.any():
        min_val = np.min(property_data[finite_mask])
        max_val = np.max(property_data[finite_mask])
    else:
        # Fall back if no finite values
        min_val = 0
        max_val = 10

    # Create the figure
    fig = go.Figure()

    # The data is already filtered by filter_by_cost
    lon_flat = data_dict['longitude'].flatten()
    lat_flat = data_dict['latitude'].flatten()
    color_flat = data_dict[selected_property].flatten()
    cost_flat = data_dict['cost_function'].flatten()

    if debug > 1:
        print("After flattening:")
        print(f"  lon_flat shape: {lon_flat.shape}")
        print(f"  lat_flat shape: {lat_flat.shape}")
        print(f"  color_flat shape: {color_flat.shape}")

    # Check for finite values
    lon_finite = np.isfinite(lon_flat)
    lat_finite = np.isfinite(lat_flat)
    color_finite = np.isfinite(color_flat)

    if debug > 1:
        print("Finite values:")
        print(f"  lon finite count: {lon_finite.sum()} / {len(lon_flat)}")
        print(f"  lat finite count: {lat_finite.sum()} / {len(lat_flat)}")
        print(f"  color finite count: {color_finite.sum()} / {len(color_flat)}")

    # Create mask for non-nan values
    valid_mask = np.isfinite(lon_flat) & np.isfinite(lat_flat) & np.isfinite(color_flat)
    if debug > 1:
        print(f"Combined valid mask: {valid_mask.sum()} valid points")
        if valid_mask.sum() == 0:
            print("ERROR: No valid points for plotting!")
            print(f"Sample lon values: {lon_flat[:10]}")
            print(f"Sample lat values: {lat_flat[:10]}")
            print(f"Sample color values: {color_flat[:10]}")

    lon_valid = lon_flat[valid_mask]
    lat_valid = lat_flat[valid_mask]
    color_valid = color_flat[valid_mask]
    cost_valid = cost_flat[valid_mask]
    original_indices_valid = original_indices[valid_mask]

    if debug > 1:
        print("Data being passed to Scattermap:")
        print(f"  lon_valid length: {len(lon_valid)}")
        print(f"  lat_valid length: {len(lat_valid)}")
        print(f"  color_valid length: {len(color_valid)}")
        if len(lon_valid) > 0:
            print(f"  lon range: {lon_valid.min():.3f} to {lon_valid.max():.3f}")
            print(f"  lat range: {lat_valid.min():.3f} to {lat_valid.max():.3f}")
            print(f"  color range: {color_valid.min():.3f} to {color_valid.max():.3f}")

    # Create map with scattermap
    fig.add_trace(
        go.Scattermap(
            lon=lon_valid,
            lat=lat_valid,
            mode='markers',
            marker=dict(
                size=4,
                color=color_valid,
                colorscale='Viridis',
                colorbar=dict(
                    title=selected_property,
                    x=0.175,  # position over map
                    y=1.0,  # position at top
                    lenmode="fraction",
                    len=0.335,
                    orientation='h',
                    yanchor='bottom',
                    title_side='top',
                    thickness=15,
                    outlinewidth=1,
                    outlinecolor='black'
                ),
                showscale=True,
                cmin=min_val,
                cmax=max_val
            ),
            text=["{},{:.3f},{:.3f}".format(
                idx,
                color_valid[i],
                cost_valid[i]
            ) for i, idx in enumerate(original_indices_valid)],

            hovertemplate=(
                'Lat: %{lat:.2f}<br>' +
                'Lon: %{lon:.2f}<br>' +
                f'{selected_property}: %{{marker.color:.3f}}<br>' +
                'Cost: %{customdata[0]:.2f}' +
                '<extra></extra>'
            ),
            customdata=np.column_stack((
                cost_valid,
                original_indices_valid
            )),
            showlegend=False
        )
    )

    # If point selected, populate intensity and dolp plots
    if clicked_point_data is not None:
        if isinstance(clicked_point_data, dict) and 'row' in clicked_point_data:
            # Find position in filtered data
            # local_idx = np.where(original_indices == point_idx)[0][0]
            # New format: spatial coords
            row_idx = clicked_point_data['row']
            col_idx = clicked_point_data['col']
        elif isinstance(clicked_point_data, (int, np.integer)):
            # old format: convert flat idx to spatial coords
            original_shape = data_dict['original_shape']
            if len(original_shape) == 1:
                # 1D data (RSP)
                row_idx = clicked_point_data
                col_idx = 0
            elif len(original_shape) == 2:
                # 2D data (HARP2)
                row_idx = clicked_point_data // original_shape[1]
                col_idx = clicked_point_data % original_shape[1]
            else:
                row_idx = col_idx = None
        else:
            row_idx = col_idx = None

        intensity_data, dolp_data, wavelengths = \
            get_channel_intensity_dolp_vza(data_dict, row_idx, col_idx)

        # Generate colors for each wavelength
        wl_colors = generate_wavelength_colors(wavelengths)

        # Add intensity traces (then dolp)
        for wl in wavelengths:
            name = f'{wl} nm'

            # Add measured intensity (1st half ymvec)
            fig.add_trace(
                go.Scatter(
                    x=intensity_data[wl]['x'],
                    y=intensity_data[wl]['y_meas'],
                    mode='markers+lines',
                    name=name,
                    line=dict(color=wl_colors[wl], width=1.5),
                    marker=dict(color=wl_colors[wl], size=5),
                    legendgroup=f'wl{wl}',
                    xaxis='x',
                    yaxis='y',
                )
            )

            # Add modeled intensity (1st half fvec)
            fig.add_trace(
                go.Scatter(
                    x=intensity_data[wl]['x'],
                    y=intensity_data[wl]['y_model'],
                    mode='lines',
                    name=f'Modeled {name}',
                    line=dict(color=wl_colors[wl], width=1.5, dash='dash'),
                    legendgroup=f'wl{wl}',
                    showlegend=False,
                    xaxis='x',
                    yaxis='y',
                )
            )

            # Add measured DoLP (2nd half ymvec)
            fig.add_trace(
                go.Scatter(
                    x=dolp_data[wl]['x'],
                    y=dolp_data[wl]['y_meas'],
                    mode='markers+lines',
                    name=name,
                    line=dict(color=wl_colors[wl], width=1.5),
                    marker=dict(color=wl_colors[wl], size=5),
                    legendgroup=f'wl{wl}',
                    showlegend=False,
                    xaxis='x2',
                    yaxis='y2',
                ),
            )

            # Add modeled DolP (2nd half of fvec)
            fig.add_trace(
                go.Scatter(
                    x=dolp_data[wl]['x'],
                    y=dolp_data[wl]['y_model'],
                    mode='lines',
                    name=f'Modeled {name}',
                    line=dict(color=wl_colors[wl], width=1.5, dash='dash'),
                    legendgroup=f'wl{wl}',
                    showlegend=False,
                    xaxis='x2',
                    yaxis='y2'
                )
            )

        # Add legend traces for measured/modeled
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='lines',
                name='Measured',
                line=dict(color='black', width=1.5),
                showlegend=True,
                xaxis='x',
                yaxis='y'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='lines',
                name='Modeled',
                line=dict(color='black', width=1.5, dash='dash'),
                showlegend=True,
                xaxis='x',
                yaxis='y'
            )
        )

        # Highlight selected point on the map
        lat = data_dict['latitude'][row_idx, col_idx]
        lon = data_dict['longitude'][row_idx, col_idx]

        fig.add_trace(
            go.Scattermap(
                lon=[lon],
                lat=[lat],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='circle',
                    # line=dict(width=2, color='red')
                ),
                showlegend=False,
                hoverinfo='skip'
            )
        )
    else:
        # Add placeholder traces with instructions
        fig.add_trace(
            go.Scatter(
                # x=[0, 1],
                # y=[0.5, 0.5],
                x=[0.5],
                y=[0.5],
                mode='text',
                text=["Click a point on the map to view Intensity fit"],
                textposition="middle center",
                textfont=dict(
                    size=14,
                ),
                hoverinfo='none',
                showlegend=False,
                xaxis='x',
                yaxis='y'
            )
        )
        fig.add_trace(
            go.Scatter(
                # x=[0, 1],
                # y=[0.5, 0.5],
                x=[0.5],
                y=[0.5],
                mode='text',
                text=["Click a point on the map to view DoLP fit"],
                textposition="middle center",
                textfont=dict(
                    size=14,
                ),
                hoverinfo='none',
                showlegend=False,
                xaxis='x2',
                yaxis='y2'
            )
        )

    # Get retrieval scenario (i.e., intruments used) for fig/legend format
    scenario_id = determine_retrieval_scenario(data_dict['file_path'])
    formatted_cost = f"{max_cost:.2f}" if max_cost is not None else "All"
    if scenario_id == 1:
        legend_y = 1.13
        instrument = "HARP/SPEX"
    elif scenario_id == 2:
        legend_y = 1.14
        instrument = "HARP/SPEX/OCI"
    elif scenario_id == 3:
        legend_y = 1.0
        instrument = "HARP"
    elif scenario_id == 4:
        legend_y = 1.0
        instrument = "SPEX"

    # Configure the layout
    f1_start = 0.01
    f1_width = 0.33
    f1_end = f1_start + f1_width
    f2_start = 0.055 + f1_end
    f2_width = 0.245
    f2_end = f2_start + f2_width
    f3_start = 0.10 + f2_end
    f3_width = 0.245
    f3_end = f3_start + f3_width
    legend_start = 0.68  # started with 0.72

    center_lat = np.mean(lat_valid) if len(lat_valid) > 0 else 34
    center_lon = np.mean(lon_valid) if len(lon_valid) > 0 else -121
    if debug > 1:
        print(f"Map center: lat={center_lat:.3f}, lon={center_lon:.3f}")

    # PLOT_WIDTH = 1550  # in pixels
    fig.update_layout(
        title=f"PACE-MAPP Aerosol Properties (with {instrument}): {selected_property} (Showing points with Cost {formatted_cost})",
        height=680,
        # width=1550,
        width=PLOT_WIDTH,
        showlegend=True,

        # Setup the map style
        map=dict(
            style="open-street-map",
            center=dict(
                # lat=np.mean(data_dict['latitude']),
                # lon=np.mean(data_dict['longitude'])
                lat=center_lat,
                lon=center_lon
            ),
            zoom=4
        ),

        # Define domains for subplots
        map_domain=dict(
            x=[f1_start, f1_end],  # reduced from 0.45 to 0.4
            y=[0, 1]
        ),

        # X and Y axes for intensity plot
        xaxis=dict(
            # domain=[0.5, 0.73],
            # domain=[0.47, 0.67],
            domain=[f2_start, f2_end],
            title="Viewing Zenith Angle (degrees)",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            anchor="y"
        ),
        yaxis=dict(
            domain=[0, 1],
            title="Intensity",
            title_font=dict(size=12),
            title_standoff=0.1,
            automargin=False,
            ticksuffix=" ",
            tickfont=dict(size=10),
            anchor="x",
            side="left"
        ),

        # X and Y axes for dolp plot
        xaxis2=dict(
            # domain=[0.77, 1.0],
            # domain=[0.77, 0.97],
            domain=[f3_start, f3_end],
            title="Viewing Zenith Angle (degrees)",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            anchor="y2"
        ),
        yaxis2=dict(
            domain=[0, 1],
            title="DoLP",
            title_font=dict(size=12),
            title_standoff=2,
            automargin=False,
            ticksuffix=" ",
            tickfont=dict(size=10),
            anchor="x2",
            side="left"
        ),

        # Legend configuration
        legend=dict(
            # x=0.70,
            x=legend_start,
            y=legend_y,
            traceorder="grouped",
            orientation="v",
            xanchor="center",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(
                size=12,
                family="Arial"
            ),
        ),

        margin=dict(l=30, r=30, t=50, b=30)
    )

    # Add subplot titles
    fig.add_annotation(
        xref='x domain', yref='y domain',
        x=0.5, y=1.05,
        text="Intensity vs VZA",
        showarrow=False,
        font=dict(size=14),
        xanchor='center'
    )
    fig.add_annotation(
        xref='x2 domain', yref='y2 domain',
        x=0.5, y=1.05,
        text="DoLP vs VZA",
        showarrow=False,
        font=dict(size=14),
        xanchor='center'
    )

    # Add annotation for number of points displayed
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.01, y=0.01,
        # text=f"Showing {len(data_dict['latitude'])} points",
        text="Showing {} points".format(len(original_indices_valid)),
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )

    return fig


def create_simple_kml_content(data_dict, selected_property, original_indices):
    """
    Creates a very simple KML file with fewer than 20,000 points,
    with enhanced placemark descriptions including cost function and coordinates.
    Updated to handle 2D arrays consistently with the rest of the code.
    Removed problematic base64 image that causes red X in Google Earth Pro.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if debug > 1:
        print("KML DEBUG:")
        print(f"selected_property = {selected_property}")
        print(f"data_dict[{selected_property}] shape = {data_dict[selected_property].shape}")
        print(f"data_dict['latitude'] shape = {data_dict['latitude'].shape}")
        print(f"data_dict['longitude'] shape = {data_dict['longitude'].shape}")

    # Get original shape and flatten arrays for processing
    original_shape = data_dict['original_shape']

    # Flatten the data arrays for easier processing
    lat_flat = data_dict['latitude'].flatten()
    lon_flat = data_dict['longitude'].flatten()
    prop_flat = data_dict[selected_property].flatten()
    cost_flat = data_dict['cost_function'].flatten()

    # Create mask for finite (valid) values - exclude NaN and infinity
    valid_mask = (np.isfinite(lat_flat) &
                  np.isfinite(lon_flat) &
                  np.isfinite(prop_flat) &
                  np.isfinite(cost_flat))

    # Apply mask to get only valid points
    lat_valid = lat_flat[valid_mask]
    lon_valid = lon_flat[valid_mask]
    prop_valid = prop_flat[valid_mask]
    cost_valid = cost_flat[valid_mask]

    # Get the valid original indices (spatial grid positions)
    all_indices = np.arange(len(lat_flat))
    valid_indices = all_indices[valid_mask]

    if debug > 1:
        print(f"Total points: {len(lat_flat)}")
        print(f"Valid points: {len(lat_valid)}")
        print(f"prop_valid min/max: {np.min(prop_valid):.3f} / {np.max(prop_valid):.3f}")

    # Sample data to stay under limits
    MAX_POINTS = 18000  # Well under the 20,000 limit

    if len(valid_indices) > MAX_POINTS:
        # Random sampling from valid points
        sample_mask = np.random.choice(len(valid_indices), MAX_POINTS, replace=False)
        sample_mask = np.sort(sample_mask)  # Sort for better organization

        # Apply sampling to valid arrays
        lat_sample = lat_valid[sample_mask]
        lon_sample = lon_valid[sample_mask]
        prop_sample = prop_valid[sample_mask]
        cost_sample = cost_valid[sample_mask]
        indices_sample = valid_indices[sample_mask]
    else:
        # Use all valid points
        lat_sample = lat_valid
        lon_sample = lon_valid
        prop_sample = prop_valid
        cost_sample = cost_valid
        indices_sample = valid_indices

    # Get min/max values for color mapping
    if len(prop_sample) > 0:
        min_val = np.min(prop_sample)
        max_val = np.max(prop_sample)
        val_range = max_val - min_val
    else:
        min_val = max_val = val_range = 0
        print("Warning: No valid data points for KML export")

    if debug > 1:
        print(f"Sampled points: {len(indices_sample)}")
        print(f"Value range: {min_val:.3f} to {max_val:.3f}")

    # Start KML document - keep it very simple
    kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>PACE-MAPP {selected_property}</name>
    <description>PACE-MAPP visualization of {selected_property}</description>

    <!-- Define styles for different value ranges -->
'''

    # Create styles for diff value ranges (use 20 steps for more granularity)
    NUM_STYLES = 20

    # Generate colors using matplotlib's viridis colormap
    viridis = plt.colormaps['viridis']

    # Create a template for BalloonStyle with CSS
    # IMPORTANT: Double curly braces to escape them in f-string
    balloon_style_template = '''
      <BalloonStyle>
        <text><![CDATA[
          <style>
            table {{
              border-collapse: collapse;
              width: 100%;
            }}
            th, td {{
              padding: 8px;
              text-align: left;
              border-bottom: 1px solid #ddd;
            }}
            th {{
              background-color: #f2f2f2;
            }}
            tr:hover {{
              background-color: #f5f5f5;
            }}
          </style>
          <h3>Point $[name]</h3>
          <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>{property_name}</td><td>$[value]</td></tr>
            <tr><td>Latitude</td><td>$[latitude]</td></tr>
            <tr><td>Longitude</td><td>$[longitude]</td></tr>
            <tr><td>Cost Function</td><td>$[cost]</td></tr>
          </table>
        ]]></text>
      </BalloonStyle>
    '''

    # Replace property_name in the template
    balloon_style = \
        balloon_style_template.format(property_name=selected_property)

    for i in range(NUM_STYLES):
        # Get color from colormap
        norm_pos = i / (NUM_STYLES - 1)
        rgba = viridis(norm_pos)
        r, g, b = [int(255 * c) for c in rgba[:3]]

        # KML uses AABBGGRR format (alpha, blue, green, red)
        color_hex = f"ff{b:02x}{g:02x}{r:02x}"

        kml += f'''
    <Style id="style{i}">
      <IconStyle>
        <color>{color_hex}</color>
        <scale>0.8</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>
        </Icon>
      </IconStyle>
      <LabelStyle>
        <scale>0</scale>
      </LabelStyle>{balloon_style}
    </Style>
'''

    # Add a folder for the data points
    kml += '''
    <Folder>
      <name>Data Points</name>
'''

    # Add points
    for i in range(len(indices_sample)):
        # Convert flat index back to coordinates for naming
        flat_idx = indices_sample[i]
        if len(original_shape) == 1:
            # 1D data (RSP)
            row = flat_idx
            col = 0
        elif len(original_shape) == 2:
            # 2D data (HARP2)
            row = flat_idx // original_shape[1]
            col = flat_idx % original_shape[1]
        else:
            row = col = flat_idx

        lat = lat_sample[i]
        lon = lon_sample[i]
        val = prop_sample[i]
        cost = cost_sample[i]

        # Determine style based on value
        if val_range > 0:
            norm_val = (val - min_val) / val_range
            style_idx = min(int(norm_val * NUM_STYLES), NUM_STYLES - 1)
        else:
            style_idx = 0

        # Add point with enhanced description and extended data for balloon
        # Use row,col for point name to be consistent with spatial indexing
        kml += f'''
      <Placemark>
        <name>[{row},{col}]</name>
        <styleUrl>#style{style_idx}</styleUrl>
        <ExtendedData>
          <Data name="value">
            <value>{val:.5f}</value>
          </Data>
          <Data name="latitude">
            <value>{lat:.5f}</value>
          </Data>
          <Data name="longitude">
            <value>{lon:.5f}</value>
          </Data>
          <Data name="cost">
            <value>{cost:.5f}</value>
          </Data>
          <Data name="grid_row">
            <value>{row}</value>
          </Data>
          <Data name="grid_col">
            <value>{col}</value>
          </Data>
        </ExtendedData>
        <Point>
          <coordinates>{lon},{lat}</coordinates>
        </Point>
      </Placemark>
'''

    # Close folder
    kml += '''
    </Folder>
'''

    # Calculate legend position (place to right of data for now
    if len(lon_sample) > 0 and len(lat_sample) > 0:
        # Get data bounds
        min_lon, max_lon = np.min(lon_sample), np.max(lon_sample)
        min_lat, max_lat = np.min(lat_sample), np.max(lat_sample)

        # Calc data range
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat

        # Position legend to right of data with som epadding
        legend_lon = max_lon + (lon_range * 0.05)  # 15% of data width to right
        legend_lat = max_lat - (lat_range * 0.5)  # middle of data height

        # ensure legend doesn't go too far if data range is very small
        if lon_range < 0.1:
            legend_lon = max_lon + 0.05  # fixed offset
        if lat_range < 0.1:
            legend_lat = np.mean(lat_sample)
    else:
        legend_lon, legend_lat = 0, 0

    # Add legend info as a separate placemark instead of problematic image overlay
    kml += f'''
    <Placemark>
      <name>ðŸ“Š Legend</name>
      <description><![CDATA[
        <h3>PACE-MAPP Data Legend</h3>
        <table border="1" style="border-collapse: collapse; margin: 10px;">
          <tr><th style="background-color: #f2f2f2; padding: 8px;">Property</th><td style="padding: 8px;">{selected_property}</td></tr>
          <tr><th style="background-color: #f2f2f2; padding: 8px;">Minimum Value</th><td style="padding: 8px;">{min_val:.5f}</td></tr>
          <tr><th style="background-color: #f2f2f2; padding: 8px;">Maximum Value</th><td style="padding: 8px;">{max_val:.5f}</td></tr>
          <tr><th style="background-color: #f2f2f2; padding: 8px;">Valid Points Shown</th><td style="padding: 8px;">{len(indices_sample)}</td></tr>
          <tr><th style="background-color: #f2f2f2; padding: 8px;">Color Scale</th><td style="padding: 8px;">Viridis (Purple=Low → Yellow=High)</td></tr>
        </table>
        <p><strong>Note:</strong> Click on any data point to see detailed information.</p>
      ]]></description>
      <Style>
        <IconStyle>
          <Icon>
            <href>http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href>
          </Icon>
          <scale>1.2</scale>
        </IconStyle>
      </Style>
      <Point>
        <coordinates>{legend_lon},{legend_lat},0</coordinates>
      </Point>
    </Placemark>
'''

    # Close document
    kml += '''
  </Document>
</kml>
'''

    return kml


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def run_app(initial_file_path, directory_path):

    # Scan directory for files
    all_files = scan_directory_for_files(directory_path)

    # Get file basenames for dropdown display
    file_basenames = [os.path.basename(file) for file in all_files]
    file_options = [{'label': basename, 'value': full_path}
                    for basename, full_path in zip(file_basenames, all_files)]

    # Don't read the initial file automatically - let user select a file
    # Set up placeholder/default values for app initialization
    max_cost_value = 1.0  # Default max value for cost slider
    default_cost_value = default_cost  # Use the global default
    default_var = None  # No default variable until file is loaded

    # Create the Dash app
    app = Dash(__name__, suppress_callback_exceptions=True)

    # Note: dropdown_options will be populated when a file is selected
    # through the file selector callback

    # New streamlined layout with dropdown-based plot selection
    app.layout = html.Div([
        # Stores and download components
        dcc.Download(id='download-image'),
        dcc.Store(id='current-file-data', data={
            'file_path': None,
            'max_cost_value': max_cost_value,
            'default_var': default_var
        }),
        dcc.Store(id='clicked-point-store'),
        dcc.Store(id='time-plot-clicked-point-store'),
        dcc.Store(id='applied-cost-value', data=default_cost_value),

        # Page header
        html.H1("PACE-MAPP Aerosol Properties Interactive Visualization",
                style={
                    'textAlign': 'center',
                    'marginBottom': '20px',
                    'marginTop': '10px',
                    'color': '#2c3e50'
                }),

        # Main container with left and right panels
        html.Div([
            # ============================================================
            # LEFT PANEL - Universal Controls
            # ============================================================
            html.Div([
                # Controls section
                html.Div([
                    html.H3("Controls", style={
                        'margin': '0 0 10px 0',
                        'color': '#34495e',
                        'fontSize': '18px',
                        'textDecoration': 'underline',
                        'display': 'inline-block'
                    }),

                    # Universal Analysis Mode
                    html.Div([
                        html.Label("Analysis Mode:", style={
                            'fontWeight': 'bold',
                            'marginBottom': '5px',
                            'display': 'block',
                            'fontSize': '16px',
                        }),
                        dcc.RadioItems(
                            id='individual-analysis-mode',
                            options=[
                                {'label': ' Single File (Measured vs Modeled)', 'value': 'single'},
                                {'label': ' Compare Files (File 1 vs File 2)', 'value': 'multiple'}
                            ],
                            value='single',
                            style={'margin': '10px 0'},
                            labelStyle={'display': 'block', 'margin': '5px 0'}
                        ),
                    ]),

                    # File selector
                    html.Div([
                        html.Label("Select File:", style={
                            'fontWeight': 'bold',
                            'marginBottom': '5px',
                            'display': 'block',
                            'fontSize': '16px'
                        }),
                        dcc.Dropdown(
                            id='file-selector',
                            options=file_options,
                            value=None,
                            placeholder="Please select a file...",
                            style={
                                'marginBottom': '15px',
                                'fontSize': '12px'
                            }
                        ),
                    ]),

                    # 2nd File selector (conditional on analysis mode)
                    html.Div([
                        html.Label("Select Second File:", style={
                            'fontWeight': 'bold',
                            'marginBottom': '5px',
                            'display': 'block',
                            'fontSize': '16px'
                        }),
                        dcc.Dropdown(
                            id='individual-file-selector-2',
                            options=file_options,
                            value=None,
                            placeholder="Please select a second file...",
                            style={
                                'marginBottom': '15px',
                                'fontSize': '12px'
                            }
                        ),
                    ], id='individual-file-2-container', style={'display': 'none'}),

                    # PLOT TYPE SELECTOR (NEW)
                    html.Div([
                        html.Label("Select Plot Type:", style={
                            'fontWeight': 'bold',
                            'marginBottom': '5px',
                            'display': 'block',
                            'fontSize': '16px'
                        }),
                        dcc.Dropdown(
                            id='plot-type-selector',
                            options=[
                                {'label': 'About', 'value': 'about'},
                                {'label': 'Scatter + Intensity/DoLP', 'value': 'scatter'},
                                {'label': 'Polarized Reflectance', 'value': 'polarized'},
                                {'label': 'Residual Analysis', 'value': 'residual'},
                                {'label': 'Histogram', 'value': 'histogram'},
                                {'label': 'AOD Total', 'value': 'aod_total'}
                            ],
                            value='about',  # Default to About on startup
                            style={
                                'marginBottom': '15px',
                                'fontSize': '12px'
                            }
                        ),
                    ]),

                    # PLOT-SPECIFIC CONTROLS
                    # Polarized Reflectance controls
                    html.Div([
                        html.Label("Comparison Method:", style={
                            'fontWeight': 'bold',
                            'marginBottom': '5px',
                            'display': 'block',
                            'fontSize': '14px'
                        }),
                        dcc.Dropdown(
                            id='polarized-difference-type',
                            options=[
                                {'label': 'Simple Difference (File1 - File2)', 'value': 'simple'},
                                {'label': 'Percent Difference ((File1 - File2) / File2 × 100)', 'value': 'percent'}
                            ],
                            value='simple',
                            style={'marginBottom': '15px', 'fontSize': '12px'}
                        ),
                    ], id='polarized-controls-container', style={'display': 'none'}),

                    # Residual Analysis controls
                    html.Div([
                        html.Label("Residual Type:", style={
                            'fontWeight': 'bold',
                            'marginBottom': '5px',
                            'display': 'block',
                            'fontSize': '14px'
                        }),
                        dcc.Dropdown(
                            id='residual-type-selector',
                            options=[
                                {'label': 'Intensity Residual', 'value': 'intensity'},
                                {'label': 'DoLP Residual', 'value': 'dolp'},
                                {'label': 'Both Residuals', 'value': 'both'}
                            ],
                            value='both',
                            style={'marginBottom': '15px', 'fontSize': '12px'}
                        ),
                    ], id='residual-controls-container', style={'display': 'none'}),

                    # Aerosol property selector
                    html.Div([
                        html.Label("Select Retrieval Property:", style={
                            'fontWeight': 'bold',
                            'marginBottom': '5px',
                            'display': 'block',
                            'fontSize': '16px'
                        }),
                        dcc.Dropdown(
                            id='property-selector',
                            options=[],
                            value=None,
                            placeholder="",
                            style={
                                'marginBottom': '25px',
                                'height': '24px',
                                'fontSize': '16px'
                            }
                        ),
                    ]),

                    # Cost selector
                    html.Div([
                        html.Label("Cost Filter:",
                                   id='cost-filter-label',
                                   style={
                                      'fontWeight': 'bold',
                                      'marginBottom': '5px',
                                      'display': 'block',
                                      'fontSize': '16px'
                                    }),
                        html.Div([
                            html.Button('-', id='cost-decrement-button', n_clicks=0,
                                        style={'width': '8%', 'padding': '8px', 'backgroundColor': '#95a5a6',
                                               'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                               'cursor': 'pointer', 'marginRight': '2%', 'fontWeight': 'bold'}),
                            dcc.Input(
                                id='cost-input',
                                type='text',
                                value=None,
                                placeholder="Select file first.",
                                style={
                                    'width': '45%',
                                    'height': '24px',
                                    'fontSize': '12px',
                                    'marginRight': '2%',
                                    'textAlign': 'center'
                                }
                            ),
                            html.Button('+', id='cost-increment-button', n_clicks=0,
                                        style={'width': '8%', 'padding': '8px', 'backgroundColor': '#95a5a6',
                                               'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                               'cursor': 'pointer', 'marginRight': '5%', 'fontWeight': 'bold'}),
                            html.Button('Apply', id='apply-cost-button', n_clicks=0,
                                        style={'width': '30%', 'padding': '8px', 'backgroundColor': '#27ae60',
                                               'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                               'cursor': 'pointer'}),
                        ], style={'display': 'flex', 'marginBottom': '30px', 'alignItems': 'center'}),
                        html.Div(id='cost-input-message', style={'fontSize': '12px', 'color': '#7f8c8d'}),
                    ]),

                    # Lat/Lon inputs
                    html.Div([
                        html.Label("Enter Coordinates (optional):", style={
                            'fontWeight': 'bold',
                            'marginBottom': '5px',
                            'display': 'block',
                            'fontSize': '16px'
                        }),
                        html.Div([
                            dcc.Input(
                                id='latitude-input',
                                type='number',
                                placeholder='Latitude',
                                step=0.01,
                                style={
                                    'width': '48%',
                                    'marginRight': '4%',
                                    'height': '24px',
                                    'fontSize': '14px',
                                    'padding': '4px 8px'
                                }
                            ),
                            dcc.Input(
                                id='longitude-input',
                                type='number',
                                placeholder='Longitude',
                                step=0.01,
                                style={
                                    'width': '48%',
                                    'height': '24px',
                                    'fontSize': '14px',
                                    'padding': '4px 8px'
                                }
                            ),
                        ], style={'display': 'flex', 'marginBottom': '10px'}),
                        html.Button('Find Closest Point', id='find-point-button', n_clicks=0,
                                    style={'width': '100%', 'padding': '8px', 'backgroundColor': '#3498db',
                                           'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                           'cursor': 'pointer', 'marginBottom': '15px'}),
                    ]),

                    # Export buttons
                    html.Div([
                        html.Button('Export PNG', id='export-button', n_clicks=0,
                                    style={'width': '48%', 'marginRight': '4%', 'padding': '10px',
                                           'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                                           'borderRadius': '4px', 'cursor': 'pointer'}),
                        html.Button('Export KML', id='export-kml-button', n_clicks=0,
                                    style={'width': '48%', 'padding': '10px', 'backgroundColor': '#9b59b6',
                                           'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                           'cursor': 'pointer'}),
                    ], style={'display': 'flex', 'marginBottom': '5px'}),

                    html.Div(id='export-status', style={'fontSize': '12px', 'color': '#7f8c8d', 'minHeight': '10px'}),

                ], style={
                    'padding': '15px',
                    'border': '1px solid #bdc3c7',
                    'borderRadius': '5px',
                    'backgroundColor': '#ffffff',
                    'marginBottom': '5px'
                }),

                # Selected point properties section
                html.Div([
                    html.H3("Selected Point Properties", style={
                        'margin': '0 0 15px 0',
                        'color': '#34495e',
                        'fontSize': '18px',
                        'textDecoration': 'underline',
                        'display': 'inline-block'
                    }),
                    html.Div(id='click-info', style={'marginBottom': '15px', 'fontSize': '14px'}),
                    html.Div(id='panel-properties-table', style={'maxHeight': '400px', 'overflowY': 'auto'}),
                ], id='selected-properties-container', style={
                    'padding': '15px',
                    'border': '1px solid #bdc3c7',
                    'borderRadius': '5px',
                    'backgroundColor': '#ffffff',
                    'display': 'none'  # Hidden by default until file loaded and point clicked
                }),

            ], style={
                'flex': '0 0 25%',
                'marginRight': '1%'
            }),

            # ============================================================
            # RIGHT PANEL - Plot Display Area
            # ============================================================
            html.Div([
                # About content
                html.Div([
                    html.H2("About PACE-MAPP Aerosol Properties Visualization Tool",
                            style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#2c3e50'}),
                    html.P("This Python application creates an interactive web-based visualization tool for exploring atmospheric aerosol properties from PACE (Plankton, Aerosol, Cloud, ocean Ecosystem). PACE instruments are HARP2 (polarimeter), SPEXone (polarimeter), and OCI and the Microphysical Aerosol Properties from Polarimetry (MAPP) retrieval framework (Stamnes et al., 2023).", 
                           style={'textAlign': 'center', 'marginRight': '10%', 'marginLeft': '10%', 'marginBottom': '20px'}),
                    html.P("Select a file and plot type from the dropdown menu on the left. Use the controls to filter data, select points, and export visualizations.",
                           style={'textAlign': 'center', 'marginRight': '10%', 'marginLeft': '10%'}),
                ], id='plot-about', style={'display': 'block', 'padding': '40px'}),

                # Scatter + Intensity/DoLP plots container
                html.Div([
                    # Multi-file comparison mode
                    html.Div([
                        # Row 1: side by side scatter plots
                        html.Div([
                            html.Div([
                                html.H4("File 1", id='file-1-scatter-header', style={'textAlign': 'center', 'margin': '0 0 10px 0'}),
                                dcc.Graph(
                                    id='scatter-plot-1',
                                    figure=create_placeholder_figure("Select files to compare"),
                                    style={'height': '700px', 'border': '1px solid #bdc3c7', 'borderRadius': '5px'}
                                ),
                            ], style={'width': '47%', 'display': 'inline-block', 'marginRight': '6%'}),

                            html.Div([
                                html.H4("File 2", id='file-2-scatter-header', style={'textAlign': 'center', 'margin': '0 0 10px 0'}),
                                dcc.Graph(
                                    id='scatter-plot-2',
                                    figure=create_placeholder_figure("Select files to compare"),
                                    style={'height': '700px', 'border': '1px solid #bdc3c7', 'borderRadius': '5px'}
                                ),
                            ], style={'width': '47%', 'display': 'inline-block'}),
                        ], style={'marginBottom': '50px'}),

                        # Comparison info panel
                        html.Div([
                            html.H5("Point Comparison Details", style={
                                'textAlign': 'center',
                                'margin': '0 0 10px 0',
                                'color': '#34495e',
                                'fontSize': '16px'
                            }),
                            html.Div(id='comparison-info-panel', style={
                                'border': '2px solid #ddd',
                                'padding': '15px',
                                'backgroundColor': '#f9f9f9',
                                'borderRadius': '8px',
                                'textAlign': 'center'
                            })
                        ], style={'marginBottom': '30px'}),

                        # Row 2: Side-by-side Intensity Plots
                        html.Div([
                            html.Div([
                                html.H4("Intensity - File 1", id='file-1-intensity-header', style={'textAlign': 'center', 'margin': '0 0 10px 0'}),
                                dcc.Graph(
                                    id='intensity-plot-1',
                                    figure=create_initial_combined_figure(),
                                    style={'height': '800px', 'border': '1px solid #bdc3c7', 'borderRadius': '5px'}
                                ),
                            ], style={'width': '49%', 'display': 'inline-block', 'marginRight': '1%'}),

                            html.Div([
                                html.H4("Intensity - File 2", id='file-2-intensity-header', style={'textAlign': 'center', 'margin': '0 0 10px 0'}),
                                dcc.Graph(
                                    id='intensity-plot-2',
                                    figure=create_initial_combined_figure(),
                                    style={'height': '800px', 'border': '1px solid #bdc3c7', 'borderRadius': '5px'}
                                ),
                            ], style={'width': '49%', 'display': 'inline-block'}),
                        ], style={'marginBottom': '20px'}),

                        # Row 3: Side-by-side DoLP Plots
                        html.Div([
                            html.Div([
                                html.H4("DoLP - File 1", id='file-1-dolp-header', style={'textAlign': 'center', 'margin': '0 0 10px 0'}),
                                dcc.Graph(
                                    id='dolp-plot-1',
                                    figure=create_initial_combined_figure(),
                                    style={'height': '800px', 'border': '1px solid #bdc3c7', 'borderRadius': '5px'}
                                ),
                            ], style={'width': '49%', 'display': 'inline-block', 'marginRight': '1%'}),

                            html.Div([
                                html.H4("DoLP - File 2", id='file-2-dolp-header', style={'textAlign': 'center', 'margin': '0 0 10px 0'}),
                                dcc.Graph(
                                    id='dolp-plot-2',
                                    figure=create_initial_combined_figure(),
                                    style={'height': '800px', 'border': '1px solid #bdc3c7', 'borderRadius': '5px'}
                                ),
                            ], style={'width': '49%', 'display': 'inline-block'}),
                        ]),
                    ], id='multi-file-plots-container', style={
                            'display': 'none',
                            'padding': '20px',
                            'paddingTop': '40px'
                        }),

                    # Single file mode
                    html.Div([
                        # Middle column - Scatter plot
                        html.Div([
                            dcc.Graph(
                                id='aerosol-plot',
                                figure=create_placeholder_figure("Please select a file to view the scatter plot"),
                                style={'height': '800px', 'border': '1px solid #bdc3c7', 'borderRadius': '5px'}
                            ),
                        ], style={'width': '42%', 'display': 'inline-block', 'marginRight': '1%', 'verticalAlign': 'top'}),

                        # Right column - Combined intensity and DoLP plot
                        html.Div([
                            dcc.Graph(
                                id='combined-plot',
                                figure=create_placeholder_figure(""),
                                style={'height': '800px', 'border': '1px solid #bdc3c7', 'borderRadius': '5px'}
                            ),
                        ], style={'width': '57%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    ], id='single-file-plots-container', style={'display': 'block'}),
                ], id='plot-scatter', style={'display': 'none'}),

                # Polarized Reflectance plot
                html.Div([
                    dcc.Graph(
                        id='polarized-reflectance-plot',
                        figure=create_placeholder_figure("Select a point to view polarized reflectance"),
                        style={'height': '800px', 'border': '1px solid #bdc3c7', 'borderRadius': '5px'}
                    ),
                ], id='plot-polarized', style={'display': 'none'}),

                # Residual Analysis plot
                html.Div([
                    dcc.Graph(
                        id='residual-plot',
                        figure=create_placeholder_figure("Select a point to view residuals"),
                        style={'height': '800px', 'border': '1px solid #bdc3c7', 'borderRadius': '5px'}
                    ),
                    # Sync'd properties table
                    html.Div(id='residual-click-info', style={'display': 'none'}),
                    html.Div(id='residual-panel-properties-table', style={'display': 'none'}),
                ], id='plot-residual', style={'display': 'none'}),

                # Histogram plot
                html.Div([
                    html.H2("AOD Frequency Distribution",
                            style={'textAlign': 'center', 'marginBottom': '20px'}),
                    html.P([
                        "This histogram shows the frequency distribution of AOD values for the selected property and cost filter. ",
                        "Use the controls on the left to select different wavelengths and adjust the cost threshold."
                    ], style={'textAlign': 'center', 'marginBottom': '20px', 'fontSize': '14px', 'marginLeft': '10%', 'marginRight': '10%'}),
                    dcc.Graph(
                        id='aod-histogram',
                        figure=create_placeholder_figure("Select a file and property to view histogram"),
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        }
                    )
                ], id='plot-histogram', style={'display': 'none', 'padding': '20px'}),

                # AOD Total plot
                html.Div([
                    dcc.Graph(
                        id='aod-total-plot',
                        figure=create_placeholder_figure("Select a point to view total AOD vs wavelength"),
                        style={'height': '800px', 'border': '1px solid #bdc3c7', 'borderRadius': '5px'}
                    ),
                    # Sync'd properties table
                    html.Div(id='aod-total-click-info', style={'display': 'none'}),
                    html.Div(id='aod-total-panel-properties-table', style={'display': 'none'}),
                ], id='plot-aod-total', style={'display': 'none'}),

                # Property vs Time plot container (for airborne/RSP data)
                html.Div([
                    # Property selector dropdown
                    html.Div([
                        html.Label("Select Property to Plot:", style={
                            'fontWeight': 'bold',
                            'marginBottom': '5px',
                            'display': 'block',
                            'fontSize': '14px',
                            'textAlign': 'center'
                        }),
                        dcc.Dropdown(
                            id='property-time-selector',
                            options=[],
                            value='optical_depth|total',
                            placeholder="Aerosol Optical Depth - Total",
                            style={
                                'marginBottom': '15px',
                                'fontSize': '12px',
                                'maxWidth': '500px',
                                'margin': '0 auto'
                            }
                        ),
                    ], style={'marginBottom': '20px'}),

                    # Container for single or dual plots (controlled by callback)
                    html.Div([
                        # Single file plot (default)
                        html.Div([
                            dcc.Graph(id='aod-time-plot-single', style={'height': '800px'}),
                        ], id='aod-time-single-container', style={'display': 'block'}),

                        # Multi-file plots (side-by-side)
                        html.Div([
                            html.Div([
                                dcc.Graph(id='aod-time-plot-1', style={'height': '800px'}),
                            ], style={'width': 'calc(50% - 20px)', 'display': 'inline-block', 'padding': '0 10px', 'vertical-align': 'top'}),
                            html.Div([
                                dcc.Graph(id='aod-time-plot-2', style={'height': '800px'}),
                            ], style={'width': 'calc(50% - 20px)', 'display': 'inline-block', 'padding': '0 10px', 'vertical-align': 'top'}),
                        ], id='aod-time-multi-container', style={'display': 'none', 'white-space': 'nowrap'}),
                    ]),

                    # Warning message when only one file has time data
                    html.Div(id='aod-time-warning', style={'textAlign': 'center', 'color': '#e67e22', 'marginTop': '10px'}),

                    # Properties table for clicked time point
                    html.Div([
                        html.H3("Selected Time Point Properties", style={
                            'margin': '20px 0 15px 0',
                            'color': '#34495e',
                            'fontSize': '18px',
                            'textDecoration': 'underline',
                            'textAlign': 'center'
                        }),
                        html.Div(id='time-plot-click-info', style={
                            'marginBottom': '15px',
                            'fontSize': '14px',
                            'textAlign': 'center'
                        }),
                        html.Div(id='time-plot-properties-table', style={
                            'maxHeight': '400px',
                            'overflowY': 'auto'
                        }),
                    ], id='time-plot-properties-container', style={
                        'padding': '15px',
                        'border': '1px solid #bdc3c7',
                        'borderRadius': '5px',
                        'backgroundColor': '#ffffff',
                        'marginTop': '20px',
                        'maxWidth': '1200px',
                        'margin': '20px auto',
                        'display': 'none'  # Hidden until point clicked
                    }),
                ], id='plot-aod-time', style={'display': 'none'}),

            ], style={
                'flex': '0 0 74%'
            }),

        ], style={
            'display': 'flex',
            'flexDirection': 'row',
            'padding': '0 20px',
            'gap': '0px'
        }),
    ], style={'backgroundColor': '#ecf0f1', 'minHeight': '100vh', 'padding': '20px 0'})
    # END of APP LAYOUT

    # Begin Callbacks
    # ---------------------------------------------------
    # NEW CALLBACKS FOR DROPDOWN-BASED PLOT SELECTION
    # ---------------------------------------------------
    # Callback to dynamically update plot type dropdown options based on available data
    @app.callback(
        Output('plot-type-selector', 'options'),
        [Input('file-selector', 'value'),
         Input('individual-file-selector-2', 'value')]
    )
    def update_plot_type_options(file_path_1, file_path_2):
        """
        Dynamically update plot type dropdown options based on file type.
        Shows "Property vs Time" option only for files with rsp_time data (RSP files).
        Checks both file selectors for multi-file mode.
        """
        # Base options available for all files
        base_options = [
            {'label': 'About', 'value': 'about'},
            {'label': 'Scatter + Intensity/DoLP', 'value': 'scatter'},
            {'label': 'Polarized Reflectance', 'value': 'polarized'},
            {'label': 'Residual Analysis', 'value': 'residual'},
            {'label': 'Histogram', 'value': 'histogram'},
            {'label': 'AOD Total', 'value': 'aod_total'}
        ]

        # Check both files for time data (RSP files)
        has_time_data = False
        for file_path in [file_path_1, file_path_2]:
            if file_path:
                try:
                    cache_entry = get_cached_data(file_path)
                    data_dict = cache_entry['data_dict']

                    # If rsp_time exists in either file, add the Property vs Time option
                    if 'rsp_time' in data_dict:
                        has_time_data = True
                        print(f"Found rsp_time in {os.path.basename(file_path)}")
                        break
                except Exception as e:
                    print(f"Error checking for time data in {file_path}: {e}")

        if has_time_data:
            base_options.append({'label': 'Property vs Time', 'value': 'aod_time'})

        return base_options

    # Callback to populate property selector for time series plots
    @app.callback(
        Output('property-time-selector', 'options'),
        [Input('file-selector', 'value'),
         Input('individual-file-selector-2', 'value')]
    )
    def update_property_time_options(file_path_1, file_path_2):
        """
        Dynamically populate the property selector dropdown with available property-mode combinations.
        Each entry represents a specific property and mode (e.g., "AOD - Fine Mode", "SSA - Dust Mode").
        For AOD, also includes a "Total" option.
        """
        print("Doing callback: update_property_time_options")

        # Collect property-mode combinations from both files
        # Structure: {property_name: set of modes}
        property_modes = {}

        for file_path in [file_path_1, file_path_2]:
            if file_path:
                try:
                    cache_entry = get_cached_data(file_path)
                    data_dict = cache_entry['data_dict']

                    # Only process if file has time data
                    if 'rsp_time' not in data_dict:
                        continue

                    # Extract property-mode combinations from variables
                    for var_name in data_dict.keys():
                        parts = var_name.split('_')

                        # Skip non-relevant variables
                        if var_name.startswith('rsp_') or var_name.endswith('_2d'):
                            continue
                        if var_name in ['latitude', 'longitude', 'cost_function', 'reference_wavelength',
                                        'file_format', 'available_modes', 'output_channels']:
                            continue

                        # Pattern 1: {property}_total_{wavelength} (e.g., optical_depth_total_556)
                        # This means we can offer a "total" option for this property
                        if len(parts) >= 3 and parts[-2] == 'total' and parts[-1].isdigit():
                            base_property = '_'.join(parts[:-2])
                            if base_property not in property_modes:
                                property_modes[base_property] = set()
                            property_modes[base_property].add('total')

                        # Pattern 2: {property}_{mode}_{wavelength} (e.g., ssa_fine_556)
                        elif len(parts) >= 3 and parts[-1].isdigit():
                            mode = parts[-2]
                            if mode in ['fine', 'coarse', 'dust', 'sea_salt']:
                                base_property = '_'.join(parts[:-2])
                                if base_property not in property_modes:
                                    property_modes[base_property] = set()
                                property_modes[base_property].add(mode)

                        # Pattern 3: {property}_{mode} WITHOUT wavelength (e.g., reff_fine, veff_dust)
                        elif len(parts) == 2 and parts[-1] in ['fine', 'coarse', 'dust', 'sea_salt']:
                            base_property = parts[0]
                            mode = parts[-1]
                            if base_property not in property_modes:
                                property_modes[base_property] = set()
                            property_modes[base_property].add(mode)

                except Exception as e:
                    print(f"Error extracting properties from {file_path}: {e}")

        # Create friendly display names for properties
        property_display_names = {
            'optical_depth': 'Aerosol Optical Depth',
            'ssa': 'Single Scattering Albedo',
            'real': 'Real Refractive Index',
            'imag': 'Imaginary Refractive Index',
            'asymmetry': 'Asymmetry Parameter',
            'absorption_coefficient': 'Absorption Coefficient',
            'scattering_coefficient': 'Scattering Coefficient',
            'extinction_coefficient': 'Extinction Coefficient',
            'reff': 'Effective Radius',
            'veff': 'Effective Variance',
            'number_concentration': 'Number Concentration',
            'cross_section': 'Cross Section'
        }

        # Build options list with property-mode combinations
        options = []

        # Define custom property order: AOD first, then real/imag together, rest alphabetically
        priority_order = ['optical_depth', 'real', 'imag']

        # Get remaining properties (not in priority list) and sort alphabetically
        remaining_props = sorted([p for p in property_modes.keys() if p not in priority_order])

        # Combine: priority properties + remaining sorted properties
        ordered_properties = [p for p in priority_order if p in property_modes] + remaining_props

        # Build options in custom order
        for prop in ordered_properties:
            display_name = property_display_names.get(prop, prop.replace('_', ' ').title())
            modes = property_modes[prop]

            # For AOD (optical_depth), add Total option first
            if prop == 'optical_depth' and 'total' in modes:
                options.append({
                    'label': f'{display_name} - Total',
                    'value': f'{prop}|total'
                })

            # Add mode-specific options
            mode_order = ['fine', 'coarse', 'dust', 'sea_salt']
            for mode in mode_order:
                if mode in modes and mode != 'total':
                    mode_label = mode.capitalize()
                    options.append({
                        'label': f'{display_name} - {mode_label} Mode',
                        'value': f'{prop}|{mode}'
                    })

        # If no properties found, add a default optical_depth option
        if not options:
            options = [{'label': 'Aerosol Optical Depth - Total', 'value': 'optical_depth|total'}]

        if debug > 0:
            print(f"Available time-series property-mode combinations: {[opt['value'] for opt in options]}")

        return options

    # Callback to control which plot container is visible
    @app.callback(
        [Output('plot-about', 'style'),
         Output('plot-scatter', 'style'),
         Output('plot-polarized', 'style'),
         Output('plot-residual', 'style'),
         Output('plot-histogram', 'style'),
         Output('plot-aod-total', 'style'),
         Output('plot-aod-time', 'style')],
        Input('plot-type-selector', 'value')
    )
    def update_plot_visibility(plot_type):
        print(f"Updating plot visibility for: {plot_type}")
        # Default: hide all plots
        styles = [
            {'display': 'none'},  # about
            {'display': 'none'},  # scatter
            {'display': 'none'},  # polarized
            {'display': 'none'},  # residual
            {'display': 'none'},  # histogram
            {'display': 'none'},  # aod_total
            {'display': 'none'}   # aod_time
        ]

        # Show the selected plot
        plot_map = {
            'about': 0,
            'scatter': 1,
            'polarized': 2,
            'residual': 3,
            'histogram': 4,
            'aod_total': 5,
            'aod_time': 6
        }

        if plot_type in plot_map:
            idx = plot_map[plot_type]
            styles[idx] = {'display': 'block', 'padding': '20px'} if plot_type == 'about' else {'display': 'block'}

        return tuple(styles)

    # Callback to control plot-specific controls visibility
    @app.callback(
        [Output('polarized-controls-container', 'style'),
         Output('residual-controls-container', 'style')],
        [Input('plot-type-selector', 'value'),
         Input('individual-analysis-mode', 'value')]
    )
    def update_plot_specific_controls(plot_type, analysis_mode):
        print(f"Updating plot-specific controls for: {plot_type}, mode: {analysis_mode}")

        # Polarized controls: show only for polarized plot AND when in compare mode
        polarized_style = {'display': 'block'} if (plot_type == 'polarized' and analysis_mode == 'multiple') else {'display': 'none'}

        # Residual controls: show only for residual plot
        residual_style = {'display': 'block'} if plot_type == 'residual' else {'display': 'none'}

        return polarized_style, residual_style

    # ---------------------------------------------------
    # ORIGINAL CALLBACKS START HERE
    # ---------------------------------------------------
    # TOTAL AOD CALLBACK #1 (1 of 18 total)
    #   -Update aod-total-click-info with the same data
    #   as click-info
    # ---------------------------------------------------
    # Multi-file mode - Update all plots when both files selected
    @app.callback(
        [Output('scatter-plot-1', 'figure'),
         Output('scatter-plot-2', 'figure'),
         Output('intensity-plot-1', 'figure'),
         Output('intensity-plot-2', 'figure'),
         Output('dolp-plot-1', 'figure'),
         Output('dolp-plot-2', 'figure'),
         Output('file-1-scatter-header', 'children'),
         Output('file-2-scatter-header', 'children'),
         Output('file-1-intensity-header', 'children'),
         Output('file-2-intensity-header', 'children'),
         Output('file-1-dolp-header', 'children'),
         Output('file-2-dolp-header', 'children'),
         Output('comparison-info-panel', 'children')],
        [Input('individual-analysis-mode', 'value'),
         Input('file-selector', 'value'),
         Input('individual-file-selector-2', 'value'),
         Input('property-selector', 'value'),
         Input('applied-cost-value', 'data'),
         # Input('multi-file-clicked-point', 'data')],
         Input('scatter-plot-1', 'clickData')],
        prevent_initial_call=True
    )
    def update_multi_file_plots(analysis_mode, file_path_1, file_path_2, selected_property, max_cost, clickData):
        from dash import callback_context
        print("Doing callback: update_multi_file_plots")

        # Define number of returned figures/headers
        NUM_FIGURES = 6
        NUM_HEADERS = 6

        # Set defaults/initialize file comparison info
        default_info = html.P("No comparison active")
        default_headers = ["", "", "",
                           "", "", ""]
        comparison_info = html.P("Click a point on File 1 scatter plot to see comparison details",
                                 style={'color': '#7f8c8d', 'margin': '0'})

        # Check what triggered this callback
        ctx = callback_context
        files_changed = False

        if ctx.triggered:
            trigger_id = ctx.triggered[0]['prop_id']
            if ('file-selector.value' in trigger_id or
                'individual-file-selector-2.value' in trigger_id or
                'individual-analysis-mode.value' in trigger_id):
                files_changed = True

        # use click data only if files have NOT changed
        # and use effective_clickData instead of clickData below
        effective_clickData = None if files_changed else clickData

        # Early returns for invalid states
        # When NOT in multi-file mode
        if analysis_mode != 'multiple':
            # Return placeholder figures for all 6 plots
            placeholder_fig = create_placeholder_figure("Switch to Multiple File mode")
            default_headers = default_headers
            default_info = html.P("Switch to Multiple File mode to compare files")
            return [placeholder_fig] * NUM_FIGURES + default_headers + [default_info]

        # When files are NONE
        if file_path_1 is None or file_path_2 is None:
            placeholder_msg = "Please select both files to compare"
            placeholder_fig = create_placeholder_figure(placeholder_msg)
            default_headers = default_headers
            default_info = html.P("Select both files to enable comparison")
            return [placeholder_fig] * NUM_FIGURES + default_headers + [default_info]

        # When selected_property is NONE (i.e., user not selected yet)
        if selected_property is None:
            placeholder_fig = create_placeholder_figure("Please select a property to display")
            default_headers = default_headers
            default_info = html.P("Select a property to display")
            return [placeholder_fig] * NUM_FIGURES + default_headers + [default_info]

        # HERE IS WHERE THE MAIN FUNCTION BEGINS
        # If in multi-file mode, both files and property selected
        try:
            # Load data for both files using cache
            cached_data_1 = get_cached_data(file_path_1)
            cached_data_2 = get_cached_data(file_path_2)

            data_dict_1 = cached_data_1['data_dict']
            data_dict_2 = cached_data_2['data_dict']

            # Filter data by cost for both files
            filtered_data_1, original_indices_1 = filter_by_cost(data_dict_1, max_cost)
            filtered_data_2, original_indices_2 = filter_by_cost(data_dict_2, max_cost)

            # Initialize click-related variables
            clicked_data_1 = None
            clicked_data_2 = None
            file1_row = None
            file1_col = None
            file2_row = None
            file2_col = None

            # Process the clicked point...
            if effective_clickData is not None and 'points' in effective_clickData and len(effective_clickData['points']) > 0:
                try:
                    # Get click lat/lon
                    clicked_point = effective_clickData['points'][0]
                    click_lat = clicked_point['lat']
                    click_lon = clicked_point['lon']

                    print(f"DEBUG: Processing click at ({click_lat:.3f}, {click_lon:.3f})")

                    # Extract file 1 index directly from text fiels (faster and more accurate than searching)
                    text_data = clicked_point.get('text', '')
                    print(f"DEBUG: click text data: {text_data}")
                    try:
                        # Text format is "idx, value, cost"
                        original_idx = int(text_data.split(',')[0])
                        original_shape_1 = data_dict_1['original_shape']

                        # Handle both 1D (RSP) and 2D (HARP2) data
                        if len(original_shape_1) == 1:
                            # 1D data (RSP): row is the index, col is always 0
                            file1_row = original_idx
                            file1_col = 0
                        else:
                            # 2D data (HARP2): convert flat index to row/col
                            file1_row = original_idx // original_shape_1[1]
                            file1_col = original_idx % original_shape_1[1]
                        print(f"DEBUG: File 1 extracted directly: idx={original_idx}, row={file1_row}, col={file1_col}")
                    except (ValueError, IndexError) as e:
                        print(f"WARNING: Could not parse clicked point index from text; falling back to search: {e}")
                        # Fallback to search if text parsing fails
                        file1_closest_idx = find_nearest_point(
                            data_dict_1['latitude'].flatten(),
                            data_dict_1['longitude'].flatten(),
                            click_lat, click_lon
                        )
                        original_shape_1 = data_dict_1['original_shape']

                        # Handle both 1D (RSP) and 2D (HARP2) data
                        if len(original_shape_1) == 1:
                            # 1D data (RSP): row is the index, col is always 0
                            file1_row = file1_closest_idx
                            file1_col = 0
                        else:
                            # 2D data (HARP2): convert flat index to row/col
                            file1_row = file1_closest_idx // original_shape_1[1]
                            file1_col = file1_closest_idx % original_shape_1[1]

                    # Find file 2 closest point
                    lat_2d = filtered_data_2['latitude']
                    lon_2d = filtered_data_2['longitude']
                    prop_2d = filtered_data_2[selected_property] if selected_property in filtered_data_2 else None

                    # Make sure we have valid lat/lon (should always happen)
                    valid_mask = np.isfinite(lat_2d) & np.isfinite(lon_2d)
                    if prop_2d is not None:
                        valid_mask = valid_mask & np.isfinite(prop_2d)

                    if np.any(valid_mask):
                        # Get indices of valid points (handle both 1D and 2D)
                        original_shape_2 = data_dict_2['original_shape']
                        if len(original_shape_2) == 1:
                            # 1D data (RSP)
                            valid_rows, = np.where(valid_mask)
                            valid_cols = np.zeros_like(valid_rows)
                        else:
                            # 2D data (HARP2)
                            valid_rows, valid_cols = np.where(valid_mask)

                        valid_lats = lat_2d[valid_mask]
                        valid_lons = lon_2d[valid_mask]

                        # Find nearest point in file 2
                        file2_closest_idx = find_nearest_point(
                            valid_lats,
                            valid_lons,
                            click_lat, click_lon
                        )

                        # Map back to original coords
                        file2_row = valid_rows[file2_closest_idx]
                        file2_col = valid_cols[file2_closest_idx]

                        # Get actual values (handle both 1D and 2D)
                        if len(original_shape_2) == 1:
                            # 1D data (RSP)
                            file2_actual_lat = lat_2d[file2_row]
                            file2_actual_lon = lon_2d[file2_row]
                            file2_value = prop_2d[file2_row] if prop_2d is not None else 0
                        else:
                            # 2D data (HARP2)
                            file2_actual_lat = lat_2d[file2_row, file2_col]
                            file2_actual_lon = lon_2d[file2_row, file2_col]
                            file2_value = prop_2d[file2_row, file2_col] if prop_2d is not None else 0

                        # Create clicked data for red circles
                        clicked_data_1 = {
                            'lat': click_lat,
                            'lon': click_lon,
                            'value': clicked_point.get('marker.color', 0)
                        }

                        clicked_data_2 = {
                            'lat': float(file2_actual_lat),
                            'lon': float(file2_actual_lon),
                            'value': float(file2_value)
                        }

                        # Generate comparison info
                        distance_deg = np.sqrt((click_lat - file2_actual_lat)**2 + (click_lon - file2_actual_lon)**2)
                        distance_km = distance_deg * 111
                        warning = get_distance_warning(distance_km)

                        filename1 = os.path.basename(file_path_1)
                        filename2 = os.path.basename(file_path_2)

                        # Create property tables
                        properties_table_1 = create_properties_table_compact(filtered_data_1, file1_row, file1_col, selected_property)
                        properties_table_2 = create_properties_table_compact(filtered_data_2, file2_row, file2_col, selected_property)

                        # Build comparison info panel (3-column layout as before)
                        comparison_info = html.Div([
                            html.Div([
                                # LEFT COLUMN
                                html.Div([
                                    html.H6("Selected Point Properties (file 1)", style={
                                        'textAlign': 'center',
                                        'marginTop': '5px',
                                        'marginBottom': '5px',
                                        'color': '#2c3e50',
                                        'fontSize': '16px',
                                        'fontWeight': 'bold'
                                    }),
                                    html.P(f"Location: ({click_lat:.4f}°, {click_lon:.4f}°)",
                                           style={'fontSize': '12px',
                                                  'color': '#666',
                                                  'marginBottom': '10px',
                                                  'marginTop': '5px'}),
                                    properties_table_1
                                ], style={'flex': '1', 'padding': '5px', 'minWidth': '0'}),

                                # MIDDLE COLUMN
                                html.Div([
                                    html.H6("Spatiotemporal Comparison", style={
                                        'textAlign': 'center',
                                        'marginBottom': '15px',
                                        'color': '#2c3e50',
                                        'fontSize': '16px',
                                        'fontWeight': 'bold'
                                    }),

                                    html.Div([
                                        html.Strong("Distance: "),
                                        html.Div(f"{distance_km:.1f} km", style={
                                            'fontSize': '16px',
                                            'fontWeight': 'bold'
                                        })
                                    ], style={'marginBottom': '15px', 'textAlign': 'center'}),

                                    html.Div([
                                        html.Span(f"{warning['icon']} {warning['text']}", style={
                                            'color': warning['color'],
                                            'fontWeight': 'bold',
                                            'fontSize': '13px',
                                            'padding': '8px 12px',
                                            'backgroundColor': 'white',
                                            'border': f"2px solid {warning['color']}",
                                            'borderRadius': '6px',
                                            'display': 'block',
                                            'textAlign': 'center'
                                        })
                                    ], style={'marginBottom': '15px'}),

                                    html.Div([
                                        html.Div([
                                            html.Strong("File 1: ", style={'fontSize': '11px'}),
                                            html.Span(filename1, style={'fontSize': '11px'})
                                        ], style={'marginBottom': '5px'}),
                                        html.Div([
                                            html.Strong("File 2: ", style={'fontSize': '11px'}),
                                            html.Span(filename2, style={'fontSize': '11px'})
                                        ], style={'marginBottom': '5px'})
                                    ], style={'textAlign': 'center'})

                                ], style={
                                    'flex': '0 0 250px',
                                    'padding': '10px',
                                    'borderLeft': '2px solid #ddd',
                                    'borderRight': '2px solid #ddd',
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                    'justifyContent': 'center'
                                }),

                                # RIGHT COLUMN
                                html.Div([
                                    html.H6("Selected Point Properties (file 2)", style={
                                        'textAlign': 'center',
                                        'marginTop': '1px',
                                        'marginBottom': '5px',
                                        'color': '#2c3e50',
                                        'fontSize': '16px',
                                        'fontWeight': 'bold'
                                    }),
                                    html.P(f"Location: ({file2_actual_lat:.4f}°, {file2_actual_lon:.4f}°)",
                                           style={'fontSize': '11px',
                                                  'color': '#666',
                                                  'marginBottom': '10px',
                                                  'marginTop': '5px'}),
                                    properties_table_2
                                ], style={
                                    'flex': '1',
                                    'padding': '5px',
                                    'minWidth': '0'
                                })
                            ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '0px'})
                        ])

                        print(f"DEBUG: File1[{file1_row},{file1_col}], File2[{file2_row},{file2_col}]")

                except Exception as e:
                    print(f"ERROR processing click: {e}")
                    import traceback
                    traceback.print_exc()

            # Create scatter plots (using clicked_data or None)
            scatter_fig_1 = create_scatter_plot_only(filtered_data_1, selected_property, original_indices_1, clicked_data_1, max_cost)
            scatter_fig_2 = create_scatter_plot_only(filtered_data_2, selected_property, original_indices_2, clicked_data_2, max_cost)

            # Create intensity/dolp plots
            if file1_row is not None and file2_row is not None:
                # Generate actual plots
                intensity_data_1, dolp_data_1, wavelengths_1 = get_channel_intensity_dolp_vza(data_dict_1, file1_row, file1_col)
                intensity_data_2, dolp_data_2, wavelengths_2 = get_channel_intensity_dolp_vza(data_dict_2, file2_row, file2_col)

                wl_colors_1 = generate_wavelength_colors(wavelengths_1)
                wl_colors_2 = generate_wavelength_colors(wavelengths_2)

                intensity_fig_1 = create_intensity_plot_only(intensity_data_1, wavelengths_1, wl_colors_1, "Intensity")
                intensity_fig_2 = create_intensity_plot_only(intensity_data_2, wavelengths_2, wl_colors_2, "Intensity")
                dolp_fig_1 = create_dolp_plot_only(dolp_data_1, wavelengths_1, wl_colors_1, "DoLP")
                dolp_fig_2 = create_dolp_plot_only(dolp_data_2, wavelengths_2, wl_colors_2, "DoLP")
            else:
                # Placeholder plots
                # No click data - show "click to select" messages
                intensity_fig_1 = create_initial_combined_figure()
                intensity_fig_1.layout.annotations[0].text = "Click a point on File 1 map to view Intensity plot"
                intensity_fig_2 = create_initial_combined_figure()
                intensity_fig_2.layout.annotations[0].text = "Click a point on File 2 map to view Intensity plot"
                dolp_fig_1 = create_initial_combined_figure()
                dolp_fig_1.layout.annotations[0].text = "Click a point on File 1 map to view DoLP plot"
                dolp_fig_2 = create_initial_combined_figure()
                dolp_fig_2.layout.annotations[0].text = "Click a point on File 2 map to view DoLP plot"

            # Create headers
            filename1 = os.path.basename(file_path_1)
            filename2 = os.path.basename(file_path_2)
            headers = [
                filename1,
                filename2,
                filename1,
                filename2,
                filename1,
                filename2
            ]

            return (scatter_fig_1, scatter_fig_2, intensity_fig_1, intensity_fig_2,
                    dolp_fig_1, dolp_fig_2, *headers, comparison_info)

        except Exception as e:
            print(f"ERROR in multi-file plot callback: {e}")
            import traceback
            traceback.print_exc()
            error_fig = create_placeholder_figure(f"Error: {str(e)}")
            return [error_fig] * 6 + default_headers + [default_info]

    @app.callback(
        [Output('single-file-plots-container', 'style'),
         Output('multi-file-plots-container', 'style')],
        Input('individual-analysis-mode', 'value'),
        prevent_initial_call=True
    )
    def toggle_individual_layout(analysis_mode):
        # print(f"DEBUG: Switching layout to {analysis_mode} mode")
        print("Doing callback: toggle_individual_layout")

        if analysis_mode == 'single':
            # Show single-file layout, hide multi-file layout
            return (
                {'display': 'block'},  # Show single-file container
                {'display': 'none', 'padding': '20px', 'paddingTop': '40px'}  # Hide multi-file container
            )
        else:  # analysis_mode == 'multiple'
            # Hide single-file layout, show multi-file layout
            return (
                {'display': 'none'},  # Hide single-file container
                {'display': 'block', 'padding': '20px', 'paddingTop': '40px'}  # Show multi-file container
            )

    @app.callback(
        Output('individual-file-2-container', 'style'),
        Input('individual-analysis-mode', 'value'),
        prevent_initial_call=True
    )
    def toggle_individual_file_2_visibility(analysis_mode):
        '''
        NOTE: This callback can be combined with the above "toggle_individual_lauyout"
        Display 2nd file dropdown if Analysis Mode is multi file
        '''
        print("Doing callback: toggle_individual_file_2_visibility")
        if analysis_mode == 'multiple':
            return {'display': 'block'}  # Show 2nd file selector
        else:
            return {'display': 'none'}  # hide file selector

    # Removed redundant sync callback for aod-total-click-info - no longer needed without tabs

    # ---------------------------------------------------
    # TOTAL AOD CALLBACK #2 (2 of 18 total)
    #   -Update the aod-total-panel-properties-table based
    #   on clicked point
    # ---------------------------------------------------
    @app.callback(
        Output('aod-total-panel-properties-table', 'children'),
        Input('panel-properties-table', 'children'),
        Input('clicked-point-store', 'data'),
        Input('plot-type-selector', 'value'),
        prevent_initial_call=True
    )
    def update_aod_total_panel_properties(properties_table_content, clicked_data, active_tab):
        print("Doing callback: update_aod_total_panel_properties")
        # Empty if not on aod total tab
        if active_tab != 'aod_total':
            return []

        # If no point clicked yet
        if clicked_data is None:
            return html.Div("Click on a point in the Individual tab to see properties")

        # Use the same table as the one in the main visualization
        return properties_table_content

    # ---------------------------------------------------
    # TOTAL AOD CALLBACK #3 (3 of 18 total)
    #   -Callback to update the AOD total plot
    # ---------------------------------------------------
    @app.callback(
        Output('aod-total-plot', 'figure'),
        [Input('file-selector', 'value'),
         Input('clicked-point-store', 'data'),
         Input('plot-type-selector', 'value')],
        prevent_initial_call=True
    )
    def update_aod_total_plot(file_path, clicked_data, active_tab):
        print("Doing callback: update_aod_total_plot")
        """
        Updated callback for AOD total plot that uses actual data
        """
        if active_tab != 'aod_total':
            # Return empty figure when not on aod total tab
            return go.Figure()

        # Check if we have a selected point
        if clicked_data is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Click on a point in the Individual tab to see total AOD vs wavelength",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="#7f8c8d")
            )
            fig.update_layout(
                title="Total AOD Analysis",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Aerosol Optical Depth",
                height=800,
                margin=dict(l=60, r=40, t=80, b=60)
            )
            return fig

        # Get the selected point data
        selected_row = clicked_data.get('row')
        selected_col = clicked_data.get('col')

        if selected_row is None or selected_col is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Invalid point selected! Please select a point again.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig

        # Load the data for the current file
        try:
            data_dict, _, _, _ = load_retrieval_file(file_path)
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error loading data: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig

        # Create the AOD total plot
        return create_aod_total_plot(data_dict, selected_row, selected_col)

    # ---------------------------------------------------
    # PROPERTY VS TIME CALLBACKS
    #   -Callbacks to update the Property vs Time plot (for RSP/airborne data)
    #   -Supports both single and multi-file comparison modes
    # ---------------------------------------------------
    @app.callback(
        [Output('aod-time-single-container', 'style'),
         Output('aod-time-multi-container', 'style'),
         Output('aod-time-plot-single', 'figure'),
         Output('aod-time-plot-1', 'figure'),
         Output('aod-time-plot-2', 'figure'),
         Output('aod-time-warning', 'children')],
        [Input('file-selector', 'value'),
         Input('individual-file-selector-2', 'value'),
         Input('individual-analysis-mode', 'value'),
         Input('plot-type-selector', 'value'),
         Input('property-time-selector', 'value'),
         Input('applied-cost-value', 'data'),
         Input('time-plot-clicked-point-store', 'data')],
        prevent_initial_call=True
    )
    def update_aod_time_plots(file_path_1, file_path_2, analysis_mode, active_tab, selected_property, max_cost, clicked_point_data):
        print(f"Doing callback: update_aod_time_plots with property={selected_property}")
        """
        Callback for Property vs Time plot for airborne/RSP data.
        Shows single plot for single-file mode, or dual side-by-side plots for multi-file mode.
        """
        # Parse selected_property to extract property and mode
        # Format: "property|mode" (e.g., "optical_depth|total", "ssa|fine")
        if not selected_property or '|' not in selected_property:
            # Default to total AOD if no valid property selected
            property_name = 'optical_depth'
            mode = 'total'
        else:
            parts = selected_property.split('|')
            property_name = parts[0]
            mode = parts[1]

        # Extract highlight info from clicked point data
        highlight_time_index = None
        highlight_y_value = None
        if clicked_point_data and 'time_index' in clicked_point_data:
            highlight_time_index = clicked_point_data['time_index']
            highlight_y_value = clicked_point_data.get('y_value')
            print(f"Highlighting time index: {highlight_time_index}, y_value: {highlight_y_value}")

        empty_fig = go.Figure()

        if active_tab != 'aod_time':
            # Return empty figures when not on aod_time tab
            return (
                {'display': 'block'},  # single container
                {'display': 'none'},   # multi container
                empty_fig, empty_fig, empty_fig,  # all plots
                ""  # no warning
            )

        # Check analysis mode
        is_multi_file = analysis_mode == 'multiple'

        if is_multi_file:
            # Multi-file mode - show side-by-side plots
            warning_msg = ""

            # Check if both files exist
            if not file_path_1 or not file_path_2:
                msg_fig = go.Figure()
                msg_fig.add_annotation(
                    text="Please select both files for comparison",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="#7f8c8d")
                )
                return (
                    {'display': 'none'},
                    {'display': 'block'},
                    empty_fig, msg_fig, empty_fig,
                    ""
                )

            # Load both files
            fig1 = empty_fig
            fig2 = empty_fig
            has_time_1 = False
            has_time_2 = False

            # Load file 1
            try:
                cache_entry_1 = get_cached_data(file_path_1)
                data_dict_1 = cache_entry_1['data_dict']
                if 'rsp_time' in data_dict_1:
                    has_time_1 = True
                    filename1 = os.path.basename(file_path_1)
                    # Only highlight if this plot was clicked
                    highlight_idx_1 = highlight_time_index if (clicked_point_data and clicked_point_data.get('source_plot') == 'plot-1') else None
                    highlight_y_1 = highlight_y_value if (clicked_point_data and clicked_point_data.get('source_plot') == 'plot-1') else None
                    fig1 = create_property_vs_time_plot(data_dict_1, property_name=property_name, mode=mode, title_suffix=f"File 1: {filename1}", max_cost=max_cost, highlight_time_index=highlight_idx_1, highlight_y_value=highlight_y_1)
                else:
                    fig1 = go.Figure()
                    fig1.add_annotation(
                        text=f"No time data in {os.path.basename(file_path_1)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=14, color="#95a5a6")
                    )
            except Exception as e:
                print(f"Error loading file 1: {e}")
                fig1 = go.Figure()
                fig1.add_annotation(
                    text=f"Error loading File 1: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red")
                )

            # Load file 2
            try:
                cache_entry_2 = get_cached_data(file_path_2)
                data_dict_2 = cache_entry_2['data_dict']
                if 'rsp_time' in data_dict_2:
                    has_time_2 = True
                    filename2 = os.path.basename(file_path_2)
                    # Only highlight if this plot was clicked
                    highlight_idx_2 = highlight_time_index if (clicked_point_data and clicked_point_data.get('source_plot') == 'plot-2') else None
                    highlight_y_2 = highlight_y_value if (clicked_point_data and clicked_point_data.get('source_plot') == 'plot-2') else None
                    fig2 = create_property_vs_time_plot(data_dict_2, property_name=property_name, mode=mode, title_suffix=f"File 2: {filename2}", max_cost=max_cost, highlight_time_index=highlight_idx_2, highlight_y_value=highlight_y_2)
                else:
                    fig2 = go.Figure()
                    fig2.add_annotation(
                        text=f"No time data in {os.path.basename(file_path_2)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=14, color="#95a5a6")
                    )
            except Exception as e:
                print(f"Error loading file 2: {e}")
                fig2 = go.Figure()
                fig2.add_annotation(
                    text=f"Error loading File 2: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red")
                )

            # Check if only one file has time data
            if has_time_1 and not has_time_2:
                warning_msg = "⚠️ Only File 1 has time data. File 2 does not contain airborne/RSP measurements."
            elif has_time_2 and not has_time_1:
                warning_msg = "⚠️ Only File 2 has time data. File 1 does not contain airborne/RSP measurements."
            elif not has_time_1 and not has_time_2:
                warning_msg = "⚠️ Neither file contains time data (rsp_time). This plot type requires airborne/RSP data."

            # Synchronize y-axes when both files have valid data
            if has_time_1 and has_time_2:
                try:
                    # Extract all y-values from both figures
                    y_values = []

                    # Get y-values from fig1
                    for trace in fig1.data:
                        if hasattr(trace, 'y') and trace.y is not None:
                            y_vals = [y for y in trace.y if y is not None and not np.isnan(y)]
                            y_values.extend(y_vals)

                    # Get y-values from fig2
                    for trace in fig2.data:
                        if hasattr(trace, 'y') and trace.y is not None:
                            y_vals = [y for y in trace.y if y is not None and not np.isnan(y)]
                            y_values.extend(y_vals)

                    if y_values:
                        # Compute global min and max
                        y_min = min(y_values)
                        y_max = max(y_values)

                        # Add padding (10% on each side as a start)
                        y_range = y_max - y_min
                        y_padding = y_range * 0.1 if y_range > 0 else 0.1 * abs(y_max)

                        # Set the same y-axis range for both figures
                        fig1.update_yaxes(range=[y_min - y_padding, y_max + y_padding])
                        fig2.update_yaxes(range=[y_min - y_padding, y_max + y_padding])
                except Exception as e:
                    print(f"Warning: Could not synchronize y-axes: {e}")

            return (
                {'display': 'none'},   # hide single
                {'display': 'block'},  # show multi
                empty_fig,  # single plot (not used)
                fig1, fig2,  # multi plots
                warning_msg
            )

        else:
            # Single-file mode - show single plot
            if not file_path_1:
                msg_fig = go.Figure()
                msg_fig.add_annotation(
                    text="Please select a file",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="#7f8c8d")
                )
                msg_fig.update_layout(title="Property vs Time")
                return (
                    {'display': 'block'},
                    {'display': 'none'},
                    msg_fig, empty_fig, empty_fig,
                    ""
                )

            # Load the data for the file
            try:
                cache_entry = get_cached_data(file_path_1)
                data_dict = cache_entry['data_dict']
            except Exception as e:
                error_fig = go.Figure()
                error_fig.add_annotation(
                    text=f"Error loading data: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="red")
                )
                error_fig.update_layout(title="Property vs Time - Error")
                return (
                    {'display': 'block'},
                    {'display': 'none'},
                    error_fig, empty_fig, empty_fig,
                    ""
                )

            # Create the property vs time plot
            single_fig = create_property_vs_time_plot(data_dict, property_name=property_name, mode=mode, max_cost=max_cost, highlight_time_index=highlight_time_index, highlight_y_value=highlight_y_value)
            return (
                {'display': 'block'},
                {'display': 'none'},
                single_fig, empty_fig, empty_fig,
                ""
            )

    # ---------------------------------------------------
    # TIME PLOT CLICK HANDLER CALLBACK
    #   -Handle clicks on Property vs Time plots
    #   -Display properties table for clicked time point
    # ---------------------------------------------------
    @app.callback(
        [Output('time-plot-clicked-point-store', 'data'),
         Output('time-plot-click-info', 'children'),
         Output('time-plot-properties-table', 'children'),
         Output('time-plot-properties-container', 'style')],
        [Input('aod-time-plot-single', 'clickData'),
         Input('aod-time-plot-1', 'clickData'),
         Input('aod-time-plot-2', 'clickData'),
         Input('plot-type-selector', 'value')],
        [State('file-selector', 'value'),
         State('individual-file-selector-2', 'value'),
         State('individual-analysis-mode', 'value'),
         State('time-plot-clicked-point-store', 'data')],
        prevent_initial_call=True
    )
    def handle_time_plot_click(clickData_single, clickData_1, clickData_2,
                               active_tab, file_path_1, file_path_2,
                               analysis_mode, stored_click_data):
        """
        Handle clicks on Property vs Time plots.
        Extracts time point data and displays properties table.
        """
        print("Doing callback: handle_time_plot_click")
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # Clear click state when switching away from aod_time tab
        if trigger_id == 'plot-type-selector' and active_tab != 'aod_time':
            return None, "", "", {'display': 'none'}

        # Determine which plot was clicked and get file path
        clicked_data = None
        source_plot = None
        file_path = None

        if trigger_id == 'aod-time-plot-single' and clickData_single:
            clicked_data = clickData_single
            source_plot = 'single'
            file_path = file_path_1
        elif trigger_id == 'aod-time-plot-1' and clickData_1:
            clicked_data = clickData_1
            source_plot = 'plot-1'
            file_path = file_path_1
        elif trigger_id == 'aod-time-plot-2' and clickData_2:
            clicked_data = clickData_2
            source_plot = 'plot-2'
            file_path = file_path_2
        else:
            # No new click, preserve existing state if available
            if stored_click_data:
                return stored_click_data, no_update, no_update, no_update
            else:
                return None, "", "", {'display': 'none'}

        # Extract click information
        try:
            point_data = clicked_data['points'][0]
            time_value = point_data['x']
            y_value = point_data['y']  # Get the y-value of the clicked point

            # Debug: print available keys and customdata if present
            print(f"Point data keys: {point_data.keys()}")
            if 'customdata' in point_data:
                print(f"customdata value: {point_data['customdata']}, type: {type(point_data['customdata'])}")

            # Get original index from customdata (with fallback to pointNumber)
            if 'customdata' in point_data:
                # customdata is a 2D array, extract first element
                customdata_value = point_data['customdata']
                if isinstance(customdata_value, (list, tuple, np.ndarray)):
                    time_index = int(customdata_value[0])
                else:
                    time_index = int(customdata_value)
                print(f"Using customdata: time_index = {time_index}")
            elif 'pointNumber' in point_data:
                time_index = int(point_data['pointNumber'])
                print(f"WARNING: customdata not found, using pointNumber: {time_index}")
                print(f"This may not work correctly with cost filtering applied!")
            else:
                raise ValueError("Neither customdata nor pointNumber found in click data")

            # Load data for the clicked file
            cache_entry = get_cached_data(file_path)
            data_dict = cache_entry['data_dict']

            # Helper function to extract scalar from potentially multi-dimensional data
            def extract_scalar(value):
                """Extract first scalar value from potentially nested array"""
                if value is None:
                    return None
                # Keep extracting first element until we get a scalar
                while isinstance(value, (np.ndarray, list)) and len(value) > 0:
                    value = value[0] if hasattr(value, '__len__') else value
                    if np.ndim(value) == 0:  # Scalar
                        break
                return float(value) if value is not None and np.isfinite(value) else None

            # Extract all data for this time point
            lat = extract_scalar(data_dict['latitude'][time_index])
            lon = extract_scalar(data_dict['longitude'][time_index])

            # Handle viewing angles
            if 'sensor_zenith' in data_dict:
                vza = extract_scalar(data_dict['sensor_zenith'][time_index])
                vza_deg = np.degrees(vza) if vza is not None else None
            else:
                vza_deg = None

            # SZA (stored as cosine, convert to degrees)
            if 'sza' in data_dict:
                sza = extract_scalar(data_dict['sza'][time_index])
                sza_deg = np.degrees(np.arccos(sza)) if sza is not None and abs(sza) <= 1.0 else None
            else:
                sza_deg = None

            # RAA
            if 'raa' in data_dict:
                raa = extract_scalar(data_dict['raa'][time_index])
            else:
                raa = None

            # Cost function
            if 'cost_function' in data_dict:
                cost = extract_scalar(data_dict['cost_function'][time_index])
            else:
                cost = None

            # Store click data
            click_data_to_store = {
                'time_index': time_index,
                'time_value': float(time_value),
                'y_value': float(y_value),  # Store the y-value of clicked point
                'file_path': file_path,
                'source_plot': source_plot
            }

            # Create click info display
            click_info_parts = [
                html.Strong("Time: "), f"{time_value:.2f} UTC",
                html.Br(),
                html.Strong("Location: "), f"Lat {lat:.4f}°, Lon {lon:.4f}°",
                html.Br()
            ]

            if vza_deg is not None and np.isfinite(vza_deg):
                click_info_parts.extend([
                    html.Strong("Viewing Zenith Angle: "), f"{vza_deg:.2f}°",
                    html.Br()
                ])

            if sza_deg is not None and np.isfinite(sza_deg):
                click_info_parts.extend([
                    html.Strong("Solar Zenith Angle: "), f"{sza_deg:.2f}°",
                    html.Br()
                ])

            if raa is not None and np.isfinite(raa):
                click_info_parts.extend([
                    html.Strong("Relative Azimuth Angle: "), f"{raa:.2f}°",
                    html.Br()
                ])

            if cost is not None and np.isfinite(cost):
                click_info_parts.extend([
                    html.Strong("Cost Function: "), f"{cost:.3f}",
                ])

            click_info = html.Div(click_info_parts)

            # Create properties table using new helper function
            properties_table = create_time_point_properties_table(
                data_dict, time_index, source_plot, file_path
            )

            # Show container
            container_style = {
                'padding': '15px',
                'border': '1px solid #bdc3c7',
                'borderRadius': '5px',
                'backgroundColor': '#ffffff',
                'marginTop': '20px',
                'maxWidth': '1200px',
                'margin': '20px auto',
                'display': 'block'
            }

            return click_data_to_store, click_info, properties_table, container_style

        except Exception as e:
            print(f"Error handling time plot click: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}", "", {'display': 'none'}

    # ---------------------------------------------------
    # AOD HISTOGRAM CALLBACK (4 of 18 total)
    # ---------------------------------------------------
    @app.callback(
      Output('aod-histogram', 'figure'),
      [Input('property-selector', 'value'),
       # Input('cost-input', 'value'),
       Input('applied-cost-value', 'data'),
       Input('current-file-data', 'data')],
      prevent_initial_call=True
      )
    def update_histogram(selected_property, max_cost, current_file_data):
        """
        Update histogram based on selected property and cost threshold
        """
        print("Doing callback: update_histogram")

        # Check if we have valid inputs
        if current_file_data is None or selected_property is None:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No data available - please select a file and property in the main visualization tab",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            empty_fig.update_layout(
                title="AOD Frequency Histogram",
                xaxis_title="AOD Value",
                yaxis_title="Frequency",
                height=500
            )
            return empty_fig

        # Get current file path and read
        file_path = current_file_data.get('file_path')
        if file_path is None:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No file path found",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            return empty_fig

        try:
            # Read data from current file path
            data_dict, sorted_variables, display_names, variable_metadata = load_retrieval_file(file_path)

            # Use 200 as a default max cost if not given
            if max_cost is None:
                max_cost = current_file_data.get('max_cost_value', 200.0)

            # Create the histogram
            histogram_fig = create_aod_histogram(data_dict, selected_property, max_cost)
            return histogram_fig

        except Exception as e:
            print(f"Error creating histogram: {e}")
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"Error loading histogram data: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            error_fig.update_layout(
                title="AOD Frequency Histogram",
                xaxis_title="AOD Value",
                yaxis_title="Frequency",
                height=500
            )
            return error_fig

    # ---------------------------------------------------
    # POLARIZED REFLECTANCE CALLBACK #1 (5 of 18 total)
    #   -Update the polarized-panel-properties-table based
    #   on clicked point
    # ---------------------------------------------------
    @app.callback(
        Output('polarized-panel-properties-table', 'children'),
        Input('panel-properties-table', 'children'),
        Input('clicked-point-store', 'data'),
        Input('plot-type-selector', 'value'),
        prevent_initial_call=True
    )
    def update_polarized_panel_properties(properties_table_content, clicked_data, active_tab):
        print("Doing callback: update_polarized_panel_properties")
        # Empty if not on polarized reflectance tab
        if active_tab != 'polarized':
            return []

        # If no point clicked yet
        if clicked_data is None:
            return html.Div("Click on a point in the Individual tab to see properties")

        # Use the same table as the one in the main visualization
        return properties_table_content

    # ---------------------------------------------------
    # POLARIZED REFLECTANCE CALLBACK #2 (6 of 18 total)
    #   -Update the polarized plot
    #   -Enhanced callback for polarized reflectance plot
    # ---------------------------------------------------
    @app.callback(
        Output('polarized-reflectance-plot', 'figure'),
        [Input('file-selector', 'value'),
         Input('individual-file-selector-2', 'value'),
         Input('individual-analysis-mode', 'value'),
         Input('polarized-difference-type', 'value'),
         Input('clicked-point-store', 'data'),
         Input('plot-type-selector', 'value')],
        prevent_initial_call=True
    )
    def update_polarized_reflectance_plot(file_path_1, file_path_2, analysis_mode, difference_type, clicked_data, active_tab):
        """
        Enhanced callback for polarized reflectance plot with dual-file comparison capability
        """
        print("Doing callback: update_polarized_reflectance_plot")
        if active_tab != 'polarized':
            return go.Figure()

        # Check if we have a selected point
        if clicked_data is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Click on a point in the Individual tab to see polarized reflectance",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="#7f8c8d")
            )
            fig.update_layout(
                title="Polarized Reflectance Analysis",
                xaxis_title="Viewing Zenith Angle (degrees)",
                yaxis_title="Polarized Reflectance (DoLP — Intensity)",
                height=800,
                margin=dict(l=50, r=40, t=60, b=40)
            )
            return fig

        # Get the selected point data
        selected_row = clicked_data.get('row')
        selected_col = clicked_data.get('col')

        if selected_row is None or selected_col is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Invalid point selected! Please select a point again.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig

        try:
            if analysis_mode == 'single':
                # SINGLE FILE MODE - Current behavior (measured vs modeled)
                if not file_path_1:
                    raise ValueError("No file selected")

                # Load data for file 1
                data_dict, _, _, _ = load_retrieval_file(file_path_1)
                intensity_data, dolp_data, wavelengths = get_channel_intensity_dolp_vza(
                    data_dict, selected_row, selected_col
                )
                wl_colors = generate_wavelength_colors(wavelengths)

                # Use existing function
                return create_polarized_reflectance_plot(intensity_data, dolp_data, wavelengths, wl_colors)

            else:
                # COMPARE FILES MODE - File 1 vs File 2
                if not file_path_1 or not file_path_2:
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Please select both files for comparison",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=16, color="orange")
                    )
                    fig.update_layout(
                        title="Polarized Reflectance Comparison",
                        xaxis_title="Viewing Zenith Angle (degrees)",
                        yaxis_title="Polarized Reflectance (DoLP — Intensity)",
                        height=800,
                        margin=dict(l=50, r=40, t=60, b=40)
                    )
                    return fig

                # Load data for both files
                data_dict_1, _, _, _ = load_retrieval_file(file_path_1)
                data_dict_2, _, _, _ = load_retrieval_file(file_path_2)

                # Get data for the same point from both files
                intensity_data_1, dolp_data_1, wavelengths_1 = get_channel_intensity_dolp_vza(
                    data_dict_1, selected_row, selected_col
                )
                intensity_data_2, dolp_data_2, wavelengths_2 = get_channel_intensity_dolp_vza(
                    data_dict_2, selected_row, selected_col
                )

                # Use wavelengths from first file for consistency
                wavelengths = wavelengths_1
                wl_colors = generate_wavelength_colors(wavelengths)

                # Create comparison plot
                return create_polarized_reflectance_comparison_plot(
                    intensity_data_1, dolp_data_1,
                    intensity_data_2, dolp_data_2,
                    wavelengths, wl_colors,
                    file_path_1, file_path_2, difference_type or 'simple'
                )

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating polarized reflectance plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title="Polarized Reflectance Analysis - Error",
                xaxis_title="Viewing Zenith Angle (degrees)",
                yaxis_title="Polarized Reflectance (DoLP — Intensity)",
                height=800,
                margin=dict(l=50, r=40, t=60, b=40)
            )
            return fig

    # ---------------------------------------------------
    # POLARIZED REFLECTANCE CALLBACK #3 (6 of 19 total) - REMOVED
    #   -Now handled by update_plot_specific_controls callback
    # ---------------------------------------------------

    # ---------------------------------------------------
    # SYNCING POINT TABLE CALLBACK (7 of 18 total)
    #   -Update polarized-click-info with the same data
    #   as click-info
    # ---------------------------------------------------
    @app.callback(
        Output('polarized-click-info', 'children'),
        Input('click-info', 'children')
    )
    def sync_polarized_click_info(click_info_content):
        print("Doing callback: sync_polarized_click_info")
        return click_info_content

    # ---------------------------------------------------
    # RESIDUAL PLOT CALLBACK #1 (8 of 18 total)
    #   -Update the residual-panel-properties-table based
    #   on clicked point
    # ---------------------------------------------------
    @app.callback(
        Output('residual-panel-properties-table', 'children'),
        Input('panel-properties-table', 'children'),
        Input('clicked-point-store', 'data'),
        Input('plot-type-selector', 'value'),
        prevent_initial_call=True
      )
    def update_residual_panel_properties(properties_table_content, clicked_data, active_tab):
        print("Doing callback: update_residual_panel_properties")
        # Empty if not on residual tab
        if active_tab != 'residual':
            return []

        # If no point clicked yet
        if clicked_data is None:
            return html.Div("Click on a point in the Individual tab to see properties")

        # Use the same table as the one in the main visualization
        return properties_table_content

    # ---------------------------------------------------
    # RESIDUAL PLOT CALLBACK #2 (9 of 18 total)
    #   -Update the residual plot with current one
    # ---------------------------------------------------
    @app.callback(
        Output('residual-plot', 'figure'),
        [Input('file-selector', 'value'),
         Input('residual-type-selector', 'value'),
         Input('clicked-point-store', 'data'),
         Input('plot-type-selector', 'value')],
        prevent_initial_call=True
    )
    def update_residual_plot(file_path, residual_type, clicked_data, active_tab):
        """
        Updated callback for residual plot that uses actual data
        """
        print("Doing callback: update_residual_plot")
        if active_tab != 'residual':
            # Return empty figure when not on residual tab
            return go.Figure()

        # Check if we have a selected point
        if clicked_data is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Click on a point in the Individual tab to see residuals",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="#7f8c8d")
            )
            fig.update_layout(
                title="Residual Analysis",
                xaxis_title="Viewing Zenith Angle (degrees)",
                yaxis_title="Residual Value",
                height=800,
                margin=dict(l=50, r=40, t=60, b=40)
            )
            return fig

        # Get the selected point data
        selected_row = clicked_data.get('row')
        selected_col = clicked_data.get('col')

        if selected_row is None or selected_col is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Invalid point used! Please select a point again.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig

        # Load the data for the current file
        try:
            data_dict, _, _, _ = load_retrieval_file(file_path)
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error loading data: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig

        # Create the residual graph based on what variable is selected
        if residual_type == 'intensity':
            return create_residual_plot(data_dict, selected_row, selected_col, 'intensity')
        elif residual_type == 'dolp':
            return create_residual_plot(data_dict, selected_row, selected_col, 'dolp')
        else:
            return create_residual_plot(data_dict, selected_row, selected_col, 'both')

    # ---------------------------------------------------
    # SYNCING POINT TABLE CALLBACK (10 of 18 total)
    #   -Update residual-click-info with the same data
    #   as click-info
    # ---------------------------------------------------
    @app.callback(
        Output('residual-click-info', 'children'),
        Input('click-info', 'children')
    )
    def sync_click_info(click_info_content):
        print("Doing callback: sync_click_info")
        return click_info_content

    # ---------------------------------------------------
    # TAB CALLBACKS (11 of 18 total)
    #   -Keep file selector same for both pages so they are synced
    # ---------------------------------------------------
    @app.callback(
        Output('file-selector', 'value'),
        Input('file-selector', 'value')
    )
    def sync_file_selectors(file_selector_value):
        print("Doing callback: sync_file_selectors")
        return file_selector_value

    # ---------------------------------------------------
    # 1. DATA/FILE MANAGEMENT CALLBACKS (highest level)
    # FILE SELECTOR CALLBACK (12 of 18 total)
    # ---------------------------------------------------
    @app.callback(
        [Output('property-selector', 'options'),
         Output('property-selector', 'value'),
         Output('current-file-data', 'data'),
         Output('cost-input', 'max'),
         Output('cost-input', 'value'),
         Output('clicked-point-store', 'data'),
         Output('applied-cost-value', 'data'),
         Output('cost-filter-label', 'children')],
        [Input('file-selector', 'value'),
         Input('individual-analysis-mode', 'value'),
         Input('individual-file-selector-2', 'value')],
        prevent_initial_call=True
    )
    def load_and_process_data(selected_file_path, analysis_mode, selected_file_path_2):
        print("Doing callback: load_and_process_data")

        # Handle case where no file is selected
        if selected_file_path is None:
            print("No file selected, returning empty values")
            return (
                [],  # empty property options
                None,  # no default property
                {'file_path': None, 'max_cost_value': 1.0, 'default_var': None},  # store data
                1.0,  # cost max
                None,  # cost value
                None,  # clicked point
                None,  # applied cost value
                "Cost Filter (Please select a file first):"  # label
            )

        # Read new file
        try:
            print(f"selected_file_path = {selected_file_path}")
            new_data_dict, new_sorted_variables, new_display_names, new_variable_metadata = load_retrieval_file(selected_file_path)

            # Get new max cost value
            new_max_cost_value = np.nanmax(new_data_dict['cost_function'])
            new_min_cost_value = np.nanmin(new_data_dict['cost_function'])
            # new_default_cost_value = min(0.7, new_max_cost_value)
            new_default_cost_value = min(default_cost, new_max_cost_value)

            # Create new dropdown options
            new_dropdown_options = create_dropdown_options(new_sorted_variables, new_display_names, new_variable_metadata)

            # Select default variable
            new_default_var = None
            for var in new_sorted_variables:
                if 'optical_depth' in var and ('556' in var or '_556' in var) and 'fine' in var:
                    new_default_var = var
                    break

            if new_default_var is None and new_sorted_variables:
                new_default_var = new_sorted_variables[0]

            # Update current file data store
            new_file_data = {
                'file_path': selected_file_path,
                'max_cost_value': float(new_max_cost_value),
                'default_var': new_default_var
            }

            # Reset clicked point data when changing files
            # return (
            #     new_dropdown_options,
            #     new_default_var,
            #     new_file_data,
            #     # new_max_cost_value,
            #     min(default_cost, new_max_cost_value),
            #     # new_max_cost_value,
            #     new_default_cost_value,
            #     None
            # )
            return (
                new_dropdown_options,
                new_default_var,
                new_file_data,
                new_max_cost_value,
                new_default_cost_value,
                None,
                new_default_cost_value,
                f"Cost Filter (Default={default_cost:.2f}/Range=[{new_min_cost_value:.3f}, {new_max_cost_value:.3f}]):",
            )

        except Exception as e:
            import dash
            print(f"Error loading file {selected_file_path}: {str(e)}")
            # If error, return current values
            raise dash.exceptions.PreventUpdate

    # ---------------------------------------------------
    # COST INCREMENT/DECREMENT CALLBACK
    # ---------------------------------------------------
    @app.callback(
        Output('cost-input', 'value', allow_duplicate=True),
        [Input('cost-increment-button', 'n_clicks'),
         Input('cost-decrement-button', 'n_clicks')],
        [State('cost-input', 'value'),
         State('current-file-data', 'data')],
        prevent_initial_call=True
    )
    def increment_decrement_cost(inc_clicks, dec_clicks, current_value, current_file_data):
        """Handle +/- buttons for cost input with 0.01 step"""
        from dash import callback_context

        if not callback_context.triggered:
            return no_update

        # Get which button was clicked
        button_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        # Get max cost from current file
        max_cost_value = 200.0
        if current_file_data is not None:
            max_cost_value = current_file_data.get('max_cost_value', 10.0)

        # Get current value or default
        if current_value is None:
            current_value = min(default_cost, max_cost_value)

        # Increment or decrement by 0.01
        if button_id == 'cost-increment-button':
            new_value = current_value + 0.01
        elif button_id == 'cost-decrement-button':
            new_value = current_value - 0.01
        else:
            return no_update

        # Round to avoid floating point precision issues
        new_value = round(new_value, 3)

        # Clamp to valid range
        new_value = max(0, min(new_value, max_cost_value))

        return new_value

    # ---------------------------------------------------
    # COST FUNCTION FILTER CALLBACK (13 of 18 total)
    # ---------------------------------------------------
    @app.callback(
        [Output('cost-input', 'value', allow_duplicate=True),
         Output('cost-input-message', 'children'),
         Output('applied-cost-value', 'data', allow_duplicate=True)],
        [Input('apply-cost-button', 'n_clicks')],
        [State('cost-input', 'value'),
         State('current-file-data', 'data')],
        # prevent_initial_call='initial_duplicate'
        prevent_initial_call=True
    )
    def validate_cost_input(n_clicks, input_value, current_file_data):
        print("Doing callback: validate_cost_input")
        # Debugging
        if debug > 1:
            print("Current file data:", current_file_data)

        if n_clicks == 0:
            # return no_update, ""
            return no_update, "", no_update

        # Safer to use default value if max_cost_value can't be found
        max_cost_value = 200.0
        if current_file_data is not None:
            # Get current max cost val from store
            max_cost_value = current_file_data.get('max_cost_value', 10.0)

        if debug > 1:
            print("Max cost value:", max_cost_value)

        # Handle text input - convert to float
        if input_value is None or input_value == "":
            default_val = min(default_cost, max_cost_value)
            return default_val, f"Using default cost value ({default_cost})", default_val

        # Try to parse the input as a number
        try:
            if isinstance(input_value, str):
                input_value = float(input_value)
        except (ValueError, TypeError):
            default_val = min(default_cost, max_cost_value)
            return default_val, f"Invalid input. Using default cost value ({default_cost})", default_val

        # Ensure cost val is within bounds
        if input_value < 0:
            return 0, "Input was less than 0. Using minimum value (0).", 0

        if input_value > max_cost_value:
            return max_cost_value, f"Input exceeded maximum. Using maximum value ({max_cost_value:.2f}).", max_cost_value

        # Valid input - use it
        return input_value, f"Using cost threshold: {input_value:.3f}", input_value

    # 3. UI SYNCHRONIZATION CALLBACKS
    # ---------------------------------------------------
    # LAT-LON INPUT CALLBACK (14 of 18 total)
    # ---------------------------------------------------
    @app.callback(
            [Output('latitude-input', 'value'),
             Output('longitude-input', 'value')],
            [Input('aerosol-plot', 'clickData')],
            [State('clicked-point-store', 'data'),
             State('current-file-data', 'data')],
            prevent_initial_call=True
            )
    def update_latlon_inputs(clickData, stored_point_data, current_file_data):
        print("Doing callback: update_latlon_inputs")
        # if point is clicked on map, update the text boxes
        if clickData is not None:
            try:
                point_data = clickData['points'][0]
                if 'lat' in point_data:  # For scattergeo
                    return point_data['lat'], point_data['lon']
                else:  # for scatter
                    return point_data['y'], point_data['x']
            except Exception as e:
                print(f"Error updating lat/lon inputs: {e}")

        # if no new click, maintain previous values
        if stored_point_data is not None and current_file_data is not None:
            idx = stored_point_data.get('original_idx')
            file_path = current_file_data.get('file_path')

            if idx is not None and file_path is not None:
                try:
                    # Read data for current file
                    data_dict, _, _, _ = load_retrieval_file(file_path)

                    # Filter by current cost value
                    max_cost = current_file_data.get('max_cost_value')
                    filtered_data, original_indices = filter_by_cost(data_dict, max_cost)

                    # Check if index exists in original_indices
                    if idx in original_indices:
                        local_idx = np.where(original_indices == idx)[0][0]
                        return filtered_data['latitude'][local_idx], filtered_data['longitude'][local_idx]
                except Exception as e:
                    print(f"Error retrieving coordinates for stored point: {e}")

        # Default - no values
        return None, None

    # ---------------------------------------------------
    # MAIN VISUALIZATION CALLBACK (15 of 18 total)
    #   -Core functionality
    # ---------------------------------------------------
    @app.callback(
        [Output('aerosol-plot', 'figure'),
         Output('combined-plot', 'figure'),
         Output('clicked-point-store', 'data', allow_duplicate=True),
         Output('click-info', 'children'),
         Output('panel-properties-table', 'children'),
         Output('selected-properties-container', 'style')],
        [Input('property-selector', 'value'),
         # Input('cost-input', 'value'),
         Input('applied-cost-value', 'data'),
         Input('aerosol-plot', 'clickData'),
         Input('find-point-button', 'n_clicks'),
         Input('current-file-data', 'data')],
        [State('latitude-input', 'value'),
         State('longitude-input', 'value'),
         State('clicked-point-store', 'data')],
        # prevent_initial_call='initial_duplicate'
        prevent_initial_call=True
    )
    def update_all_plots(selected_property, max_cost, clickData, find_button_clicks,
                         current_file_data, input_lat, input_lon, stored_point_data):
        print("Doing callback: update_all_plots")

        # Determine which input triggered callback
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # Do the below to preserve the zoom before and after any clicks
        # have been made
        has_active_click = clickData is not None and 'points' in clickData
        if has_active_click:
            uirevision_value = "preserve-zoom"
        else:
            uirevision_value = "preserve-zoom"

        # Use the current file data instead of global data_dict
        if current_file_data is None:
            empty_fig = create_placeholder_figure("No file selected")
            hidden_style = {'display': 'none'}
            return (empty_fig, empty_fig, None, "No file selected", "", hidden_style)

        # Get current file path from store and read data
        file_path = current_file_data.get('file_path')
        if file_path is None:
            empty_fig = create_placeholder_figure("Please select a file to view data")
            hidden_style = {'display': 'none'}
            return (empty_fig, empty_fig, None, "Please select a file", "", hidden_style)

        # Read data from cache NOT from current file path
        cached_data = get_cached_data(file_path)
        data_dict = cached_data['data_dict']
        # data_dict, sorted_variables, display_names, variable_metadata = \
        #     read_hdf5_variables(file_path)

        original_shape = data_dict['original_shape']

        # Filter data by cost function
        filtered_data, original_indices = filter_by_cost(data_dict, max_cost)

        # Handle point selection (same logic as before)
        clicked_point_data = None

        # User clicked "Find nearest point" button with lat/lon values
        if trigger_id == 'find-point-button' and find_button_clicks > 0 and input_lat is not None:
            try:
                # Find nearest point logic (same as before)
                lon_flat = filtered_data['longitude'].flatten()
                lat_flat = filtered_data['latitude'].flatten()
                color_flat = filtered_data[selected_property].flatten()
                valid_mask = np.isfinite(lon_flat) & np.isfinite(lat_flat) & np.isfinite(color_flat)

                if valid_mask.any():
                    valid_lats = lat_flat[valid_mask]
                    valid_lons = lon_flat[valid_mask]
                    distances = np.sqrt((valid_lats - float(input_lat))**2 + (valid_lons - float(input_lon))**2)
                    nearest_valid_idx = np.argmin(distances)
                    valid_original_indices = np.arange(len(valid_mask))[valid_mask]
                    valid_indices = np.where(valid_mask)[0]
                    original_flat_idx = valid_indices[nearest_valid_idx]

                    # Handle both 1D (RSP) and 2D (HARP2) data
                    if len(original_shape) == 1:
                        # 1D data (RSP)
                        selected_row = original_flat_idx
                        selected_col = 0
                    elif len(original_shape) == 2:
                        # 2D data (HARP2)
                        selected_row = original_flat_idx // original_shape[1]
                        selected_col = original_flat_idx % original_shape[1]
                    else:
                        raise ValueError(f"Unexpected original_shape dimensionality: {len(original_shape)}D")

                    selected_point_idx = valid_original_indices[nearest_valid_idx]
                    clicked_point_data = {'row': selected_row, 'col': selected_col, 'original_idx': int(selected_point_idx)}
            except Exception as e:
                print(f"Error finding nearest point: {e}")
                clicked_point_data = stored_point_data

        # If user clicks directly on map
        elif trigger_id == 'aerosol-plot' and clickData is not None:
            try:
                # Map click logic (same as before)
                point_data = clickData['points'][0]
                lon_flat = filtered_data['longitude'].flatten()
                lat_flat = filtered_data['latitude'].flatten()
                color_flat = filtered_data[selected_property].flatten()
                valid_mask = np.isfinite(lon_flat) & np.isfinite(lat_flat) & np.isfinite(color_flat)
                valid_original_indices = np.arange(len(valid_mask))[valid_mask]

                pointNumber = point_data.get('pointNumber', 0)
                clicked_original_flat_idx = valid_original_indices[pointNumber]

                # Handle both 1D (RSP) and 2D (HARP2) data
                if len(original_shape) == 1:
                    # 1D data (RSP): row is the index, col is always 0
                    selected_row = clicked_original_flat_idx
                    selected_col = 0
                elif len(original_shape) == 2:
                    # 2D data (HARP2): convert flat index to row/col
                    selected_row = clicked_original_flat_idx // original_shape[1]
                    selected_col = clicked_original_flat_idx % original_shape[1]
                else:
                    raise ValueError(f"Unexpected original_shape dimensionality: {len(original_shape)}D")

                selected_point_idx = valid_original_indices[pointNumber]
                clicked_point_data = {'row': selected_row, 'col': selected_col, 'original_idx': int(selected_point_idx)}
            except Exception as e:
                print(f"Error processing click: {e}")
                import traceback
                traceback.print_exc()
                clicked_point_data = stored_point_data
        else:
            clicked_point_data = stored_point_data

        # Create the main scatter plot
        scatter_fig = create_scatter_plot_only(filtered_data, selected_property, original_indices, clicked_point_data, max_cost)

        # Preserve the zoom when updating the figure with uirevision
        scatter_fig.update_layout(uirevision=uirevision_value)

        # Create intensity and DoLP plots (first below is pre-click)
        combined_fig = go.Figure()

        # Update figure format pre-click to match with post-click
        combined_fig.update_layout(
            height=800,
            showlegend=True,
            # minimize overall margins
            margin=dict(
                l=50,  # left margin
                r=40,  # right
                t=40,  # top (default ~80)
                b=40  # bottom
            ),
            autosize=True
        )
        # add pre click annotations
        combined_fig.add_annotation(
            text="Click a point on the map to view Intensity and DoLP plots",
            xref="x", yref="y",
            x=2.5, y=1.5,
            showarrow=False,
            # font=dict(size=18, color="#7f8c8d"),
            font=dict(size=18, color="black"),
            xanchor='center', yanchor='middle'
        )

        if clicked_point_data is not None and 'row' in clicked_point_data:
            selected_row = clicked_point_data['row']
            selected_col = clicked_point_data['col']

            try:
                intensity_data, dolp_data, wavelengths = get_channel_intensity_dolp_vza(data_dict, selected_row, selected_col)
                wl_colors = generate_wavelength_colors(wavelengths)

                # Create combined plot with subplots
                file_format = data_dict.get('file_format', 'HARP2')
                combined_fig = create_combined_intensity_dolp_plot(
                    intensity_data, dolp_data, wavelengths, wl_colors, file_format
                )

            except Exception as e:
                print(f"Error creating intensity/DoLP plots: {e}")
                combined_fig.add_annotation(text="Error loading plot data", x=0.5, y=0.5, showarrow=False)

        # Create click info and properties table
        click_info = "No point selected"
        properties_table = ""

        if clicked_point_data is not None and 'row' in clicked_point_data:
            selected_row = clicked_point_data['row']
            selected_col = clicked_point_data['col']

            # Handle both 1D (RSP) and 2D (HARP2) indexing
            if len(original_shape) == 1:
                # 1D data (RSP): use selected_row as index
                lat = data_dict['latitude'][selected_row]
                lon = data_dict['longitude'][selected_row]
                sza = data_dict['sza'][selected_row]
                raa = data_dict['raa'][selected_row]
                val = filtered_data[selected_property][selected_row]
                cost = data_dict['cost_function'][selected_row]
            elif len(original_shape) == 2:
                # 2D data (HARP2): use [row, col] indexing or flatten
                if filtered_data[selected_property].ndim == 2:
                    lat = data_dict['latitude'][selected_row, selected_col]
                    lon = data_dict['longitude'][selected_row, selected_col]
                    sza = data_dict['sza'][selected_row, selected_col]
                    raa = data_dict['raa'][selected_row, selected_col]
                    val = filtered_data[selected_property][selected_row, selected_col]
                    cost = data_dict['cost_function'][selected_row, selected_col]
                else:
                    flat_idx = selected_row * original_shape[1] + selected_col
                    lat = data_dict['latitude'].flatten()[flat_idx]
                    lon = data_dict['longitude'].flatten()[flat_idx]
                    sza = data_dict['sza'].flatten()[flat_idx]
                    raa = data_dict['raa'].flatten()[flat_idx]
                    val = data_dict[selected_property][flat_idx]
                    cost = data_dict['cost_function'][flat_idx]
            else:
                raise ValueError(f"Unexpected original_shape dimensionality: {len(original_shape)}D")

            # Extract scalar values for sza and raa (handle multi-dimensional arrays)
            try:
                # For sza: extract first element if it's an array
                if hasattr(sza, '__len__') and not isinstance(sza, str):
                    sza_scalar = sza.flat[0] if hasattr(sza, 'flat') else sza[0]
                else:
                    sza_scalar = sza
                sza_deg = np.degrees(np.arccos(sza_scalar))
            except:
                sza_deg = None

            try:
                # For raa: extract first element if it's an array
                if hasattr(raa, '__len__') and not isinstance(raa, str):
                    raa_scalar = raa.flat[0] if hasattr(raa, 'flat') else raa[0]
                else:
                    raa_scalar = raa
            except:
                raa_scalar = None

            # Build click info with conditional formatting
            click_info_parts = [
                html.Strong("Location: "), f"Lat {lat:.4f}°, Lon {lon:.4f}°",
                html.Br()
            ]

            click_info = html.Div(click_info_parts)

            # Use compact version for consistent styling
            properties_table = create_properties_table_compact(filtered_data, selected_row, selected_col, selected_property)

        # Control panel visibility: show only when file is loaded AND point is clicked
        if clicked_point_data is not None and 'row' in clicked_point_data:
            panel_style = {
                'padding': '15px',
                'border': '1px solid #bdc3c7',
                'borderRadius': '5px',
                'backgroundColor': '#ffffff',
                'display': 'block'
            }
        else:
            panel_style = {'display': 'none'}

        return (scatter_fig, combined_fig,
                clicked_point_data, click_info, properties_table, panel_style)

    # ---------------------------------------------------
    # EXPORT CALLBACK #1 (16 of 18 total)
    #   1. First callback updates the status message immediately
    # ---------------------------------------------------
    @app.callback(
        Output('export-status', 'children'),
        Input('export-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def update_export_status(n_clicks):
        print("Doing callback: update_export_status")
        if n_clicks is None or n_clicks == 0:
            return ""

        return "Processing export..."

    # ---------------------------------------------------
    # EXPORT CALLBACK #2 (17 of 18 total)
    #   2. Second handles the actual export as png (updated to also use
    #   current_file_data)
    # ---------------------------------------------------
    @app.callback(
        [Output('download-image', 'data'),
         Output('export-status', 'children', allow_duplicate=True)],
        Input('export-button', 'n_clicks'),
        [State('property-selector', 'value'),
         State('cost-input', 'value'),
         State('clicked-point-store', 'data'),
         State('current-file-data', 'data')],
        prevent_initial_call=True
    )
    def generate_image_download(n_clicks, selected_property, max_cost, clicked_point_data, current_file_data):
        print("Doing callback: generate_image_download")
        if n_clicks is None or n_clicks == 0:
            return no_update, no_update

        try:
            # Get current file path
            file_path = current_file_data.get('file_path')

            # Read data from current file
            data_dict, sorted_variables, display_names, variable_metadata = \
                load_retrieval_file(file_path)

            # Filter data and get indices
            filtered_data, original_indices = filter_by_cost(data_dict, max_cost)

            # Get selected point if any
            point_idx = None
            if clicked_point_data is not None and 'original_idx' in clicked_point_data:
                point_idx = clicked_point_data['original_idx']

            # Generate the figure
            fig = create_export_figure(
                filtered_data, selected_property, original_indices, point_idx, max_cost
            )

            # Create image bytes
            img_bytes = pio.to_image(fig, format="png", width=1550, height=680, scale=2)

            # Encode as base64 for json serialization
            encoded_image = base64.b64encode(img_bytes).decode('ascii')

            # Prepare download data and success message
            return {
                'content': encoded_image,
                'filename': f"pace_mapp_idx{point_idx}_{selected_property}.png",
                'type': 'image/png',
                'base64': True
            }, "Export complete! Download should start automatically."

        except Exception as e:
            print(f"Export error: {str(e)}")
            traceback.print_exc()
            return no_update, f"Error: {str(e)}"

    # ---------------------------------------------------
    # EXPORT CALLBACK #3 (18 of 18 total)
    #   -export as kml callback
    # ---------------------------------------------------
    @app.callback(
        Output('download-image', 'data', allow_duplicate=True),
        Input('export-kml-button', 'n_clicks'),
        [State('property-selector', 'value'),
         State('cost-input', 'value'),
         State('clicked-point-store', 'data'),
         State('current-file-data', 'data')],
        prevent_initial_call=True
    )
    def export_kml(n_clicks, selected_property, max_cost, clicked_point_data, current_file_data):
        print("Doing callback: export_klm")
        if n_clicks is None or n_clicks == 0:
            return no_update

        try:
            # Get current file path
            file_path = current_file_data.get('file_path')

            # Read data from current file
            data_dict, sorted_variables, display_names, variable_metadata = \
                load_retrieval_file(file_path)

            # Filter data and get indices
            filtered_data, original_indices = filter_by_cost(data_dict, max_cost)

            # Create KML content
            kml_content = create_simple_kml_content(filtered_data, selected_property, original_indices)

            # Return data for download
            return {
                'content': kml_content,
                'filename': f'pace_mapp_{selected_property}.kml',
                'type': 'application/vnd.google-earth.kml+xml',
                'base64': False
            }

        except Exception as e:
            import traceback
            print(f"KML Export error: {str(e)}")
            traceback.print_exc()
            return no_update

    # Run the app
    # Modified the below from run_server to run.app for compatibility
    # app.run_server(debug=True, port=8050)
    app.run(debug=True, port=8050)


# Run the application
if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(
            description="Plot PACE-MAPP retrieved variables with Plotly."
            )
    parser.add_argument(
            "--directory",
            type=str,
            required=True,
            help="Path to the directory containing PACE-MAPP output HDF5 and NC files."
            )

    # Parse arguments
    args = parser.parse_args()

    # Set file path from args
    directory_path = args.directory

    # Check if directory exists
    if os.path.isdir(directory_path):
        if debug > 1:
            print(f'Directory {directory_path} found!')
        # Scan directory for files
        try:
            files = scan_directory_for_files(directory_path)
            if debug > 1:
                print(f'Found {len(files)} .h5 and .nc files')

            # Run app with the first file and the list of all files
            run_app(files[0], directory_path)
        except ValueError as e:
            print(f'Error: {e}')

    else:
        print(f'Directory {directory_path} not found!')
