import numpy as np
import h5py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
# from dash.dependencies import Input, Output, State
import colorsys
import argparse
import glob
import os
import io
import base64
import traceback
import sys


PLOT_WIDTH = 1550  # in pixels
debug = 1


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def scan_directory_for_files(directory_path):
    """
    Scan specified directory for specified .h5 and .nc files.

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
    Determine which instruments were used in the retrieval.

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


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================
def get_wavelength_instrument_mapping():
    """
    Returns wavelength-instrument mapping with vza counts.
    Modify this to add new instruments/wavelengths.

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


def get_instrument_for_wavelength(wavelength, wavelength_mapping):
    """
    Get instrument type for given wavelength.

    Args:
        wavelength: wavelength value (int or float)
        wavelength_mapping: list of tuples (wavelength, instrument, n_vza)

    Returns:
        instrument (str): Instrument name ('SPEX', 'HARP', 'OCI')
    """
    for wl, instrument, _ in wavelength_mapping:
        if wl == int(wavelength):
            return instrument
    return None


# =============================================================================
# DATA I/O AND PROCESSING FUNCTIONS
# =============================================================================
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

            return data_dict, sorted_variables, display_names, variable_metadata

    except Exception as e:
        print(f"Error reading file: {str(e)}")
        raise


def filter_by_cost(data_dict, max_cost=None):
    """
    Filter data by cost function, updated to handle 2d arrays and NaN values
    and preserve dimension. Sets points that fial the filtering to infinity.
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

    # Get the cost function and ensure it's 2D
    if 'cost_function' in data_dict:
        cost_2d = data_dict['cost_function']
        if cost_2d.ndim == 1:
            # reshape if it's somehow flattened
            cost_2d = cost_2d.reshape(original_shape)
    else:
        raise ValueError("cost_function not found in data_dict")

    # Create cost mask
    if max_cost is None:
        # If no max_max cost, all finite values pass (we keep NaN: see above)
        cost_mask_2d = ~np.isinf(cost_2d)
    else:
        # Points pass if non NaN AND cost <= max_cost
        # Points that are NaN stay NaN and are filtered before plotting
        # Points that fail xost filtering set to infinity
        cost_mask_2d = (~np.isnan(cost_2d)) & (cost_2d <= max_cost)

    # Create filtered dict
    filtered_dict = {}

    for key, value in data_dict.items():
        if key in ['file_path', 'original_shape', 'wavelengths', 'output_channels']:
            # Keep metadata as is
            filtered_dict[key] = value

        elif isinstance(value, np.ndarray):
            if value.ndim == 2 and value.shape == original_shape:
                # 2d arrays: set cost filtered points to inf
                filtered_value = value.copy().astype(float)
                filtered_value[~cost_mask_2d] = np.inf
                filtered_dict[key] = filtered_value
            elif value.ndim == 3 and value.shape[:2] == original_shape:
                # 3d arrays (ymvec, fvec, vza): set cost filtered points to inf
                filtered_value = value.copy().astype(float)
                filtered_value[~cost_mask_2d, :] = np.inf
                filtered_dict[key] = filtered_value
            elif value.ndim == 1 and len(value) == (original_shape[0] * original_shape[1]):
                # 1d arrays that should be 2d: reshape and apply cost filter
                value_2d = value.reshape(original_shape).astype(float)
                value_2d[~cost_mask_2d] = np.inf
                filtered_dict[key] = value_2d
            else:
                # Arrays with differend dimension, keep as is
                filtered_dict[key] = value
        else:
            # Non array values keep as is
            filtered_dict[key] = value

    # Return original_indices (all points) for compatibility with existing code
    # The plotting function can determing valid points using np.isfinite
    total_points = original_shape[0] * original_shape[1]
    original_indices = np.arange(total_points)

    # Debug
    valid_count = cost_mask_2d.sum()
    nan_count = np.isnan(cost_2d).sum()
    cost_filtered_count = total_points - valid_count - nan_count

    if debug > 1:
        print("Cost filter results:")
        print(f"  {valid_count} points passed cost filter")
        print(f"  {cost_filtered_count} points failed cost filter (set to infinity)")
        print(f"  {nan_count} points have NaN (retrieval failure)")
        print(f"  Total: {total_points} points")
        print(f"Cost range: {np.nanmin(cost_2d):.3f} to {np.nanmax(cost_2d):.3f}")
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
    wavelength_mapping = get_wavelength_instrument_mapping()

    # Use order from output_channels or wavelengths in file
    output_channels = data_dict.get('wavelengths', None)
    channel_ranges, metadata = build_channel_ranges(wavelength_mapping, output_channels)

    total_intensity_angles = metadata['total_intensity_angles']
    total_dolp_angles = metadata['total_dolp_angles']

    # Extract data for the specific spatial location [row_idx, col_idx]
    if ymvec.ndim == 3 and ymvec.shape[:2] == original_shape:
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
    else:
        raise ValueError("Unexpected shape for ymvec: {}, expected shape starting with {}".format(
            ymvec.shape, original_shape))

    # Extract angular data for this point
    if vza.ndim == 3 and vza.shape[:2] == original_shape:
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
    """Create properly formatted dropdown options"""
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

    # Highlight selected point
    if clicked_point_data is not None and 'row' in clicked_point_data:
        selected_row = clicked_point_data['row']
        selected_col = clicked_point_data['col']
        lat = data_dict['latitude'][selected_row, selected_col]
        lon = data_dict['longitude'][selected_row, selected_col]

        fig.add_trace(
            go.Scattermap(
                lon=[lon], lat=[lat],
                mode='markers',
                marker=dict(size=10, color='red', symbol='circle'),
                showlegend=False,
                hoverinfo='skip'
            )
        )

    center_lat = np.mean(lat_valid) if len(lat_valid) > 0 else 34
    center_lon = np.mean(lon_valid) if len(lon_valid) > 0 else -121

    fig.update_layout(
        title="",
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=4.35
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


def create_combined_intensity_dolp_plot(intensity_data, dolp_data, wavelengths, wl_colors):
    """
    Create a single plot with intensity and DoLP as subplots
    with legend between them
    """

    DEBUG_PLOTTING = True

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Intensity vs VZA", "DoLP vs VZA"),
        vertical_spacing=0.20,  # spacing between plots
        shared_xaxes=True,
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
        height=800,
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
                text="<b>Wavelengths </b>" + "(— Solid: Measured, - - Dashed: Modeled)</b>",
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
        wl_mapping = get_wavelength_instrument_mapping()
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


def create_properties_table(filtered_data, selected_row, selected_col):
    """Create a table showing aerosol properties by mode"""

    # Define the properties and their display names (NOTE: we only
    # print values at 556 nm for now)
    properties_config = [
        ('optical_depth', 'Optical Depth', '556'),
        ('ssa', 'Single Scattering Albedo', '556'),
        ('real', 'Real Refractive Index', '556'),
        ('imag', 'Imaginary Refractive Index', '556'),
        ('asymmetry', 'Asymmetry Parameter', '556'),
        ('absorption_coefficient', 'Absorption Coefficient', '556'),
        ('cross_section', 'Cross Section', '556'),
        ('extinction_coefficient', 'Extinction Coefficient', '556'),
        ('number_concentration', 'Number Concentration', '556'),
        ('scattering_coefficient', 'Scattering Coefficient', '556'),
        ('reff', 'Effective Radius', ''),
        ('veff', 'Effective Variance', ''),
    ]

    modes = ['fine', 'coarse', 'dust']
    mode_colors = {
        'fine': '#3498db',
        'coarse': '#e74c3c',
        'dust': '#f39c12'
    }

    # Create table header with updated property column title
    header = html.Tr([
        html.Th("Property (* 556 nm)", style={'textAlign': 'left', 'padding': '8px', 'borderBottom': '2px solid #34495e'}),
        *[html.Th(mode.title(),
                  style={'textAlign': 'center', 'padding': '8px', 'borderBottom': '2px solid #34495e',
                         'color': mode_colors[mode], 'fontWeight': 'bold'})
          for mode in modes]
    ])

    # Create table rows
    table_rows = [header]

    for prop_base, prop_display, wavelength in properties_config:
        # Build the property keys for each mode
        mode_values = {}

        for mode in modes:
            if wavelength:
                prop_key = f"{prop_base}_{mode}_{wavelength}"
            else:
                prop_key = f"{prop_base}_{mode}"

            if prop_key in filtered_data:
                try:
                    value = filtered_data[prop_key][selected_row, selected_col]
                    if np.isfinite(value):
                        mode_values[mode] = f"{value:.3f}"
                    else:
                        mode_values[mode] = "N/A"
                except:
                    mode_values[mode] = "N/A"
            else:
                mode_values[mode] = "—"

        # Create row if at least one mode has data
        if any(val not in ["—", "N/A"] for val in mode_values.values()):
            # Use * for 556nm properties, show wavelength for others
            if wavelength == '556':
                property_label = f"{prop_display}*"
            elif wavelength:
                property_label = f"{prop_display} ({wavelength}nm)"
            else:
                property_label = prop_display

            row = html.Tr([
                html.Td(property_label,
                        style={'padding': '8px', 'borderBottom': '1px solid #ecf0f1', 'fontWeight': '500'}),
                *[html.Td(mode_values.get(mode, "—"),
                          style={'textAlign': 'center', 'padding': '8px',
                                 'borderBottom': '1px solid #ecf0f1',
                                 'color': mode_colors[mode] if mode_values.get(mode, "—") not in ["—", "N/A"] else '#95a5a6'}) 
                  for mode in modes]
            ])
            table_rows.append(row)

    return html.Table(table_rows, style={'width': '100%', 'borderCollapse': 'collapse'})


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
            # old format: convert flat idx to spatial coors
            original_shape = data_dict['original_shape']
            row_idx = clicked_point_data // original_shape[1]
            col_idx = clicked_point_data % original_shape[1]
            # print(f"DEBUG: Converted old format {clicked_point_data} to [{row_idx}, {col_idx}]")
        else:
            # print(f"DEBUG: unexpected clicked_point_data format: {type(clicked_point_data)}")
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
        title=f"PACE-MAPP Aerosol Properties (with {instrument}): {selected_property} (Showing points with Cost ≤ {formatted_cost})",
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
        # Convert flat index back to 2D coordinates for naming
        flat_idx = indices_sample[i]
        row = flat_idx // original_shape[1]
        col = flat_idx % original_shape[1]

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
      <name>📊 Legend</name>
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

    # Read the initial HDF5 file
    try:
        data_dict, sorted_variables, display_names, variable_metadata = \
                read_hdf5_variables(initial_file_path)

        # If we successfully got data, create the app
        if len(data_dict['latitude']) > 0:
            # Get maximum cost function value for slider range
            # (updated to nanmax, nanmin)
            max_cost_value = np.nanmax(data_dict['cost_function'])
            min_cost_value = np.nanmin(data_dict['cost_function'])

            # Create the Dash app
            app = Dash(__name__, suppress_callback_exceptions=True)

            # Select a default variable to display - prefer fine AOD (556)
            # Will soon change to total AOD?
            default_var = None
            for var in sorted_variables:
                if 'optical_depth' in var and ('556' in var or '_556' in var) and 'fine' in var:
                    default_var = var
                    break

            if default_var is None and sorted_variables:
                default_var = sorted_variables[0]

            # Filter initial data and get original indices
            filtered_data, original_indices = filter_by_cost(data_dict, max_cost_value)

            # Group variables for the dropdown
            dropdown_options = create_dropdown_options(sorted_variables, display_names, variable_metadata)

            # Updated app layout with new 3 column design
            app.layout = html.Div([
                # Add download component and stores
                dcc.Download(id='download-image'),
                dcc.Store(id='current-file-data', data={
                    'file_path': initial_file_path,
                    'max_cost_value': max_cost_value,
                    'default_var': default_var
                }),
                dcc.Store(id='clicked-point-store'),

                # Page header
                html.H1("PACE-MAPP Aerosol Properties Interactive Visualization",
                        style={
                            'textAlign': 'center',
                            'marginBottom': '20px',
                            'marginTop': '10px',
                            'color': '#2c3e50'
                        }),

                # Main three-column layout
                html.Div([

                    # LEFT COLUMN - Controls and Properties
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
                                    value=initial_file_path,
                                    style={
                                        'marginBottom': '15px',
                                        'fontSize': '12px'
                                    }
                                ),
                            ]),

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
                                    options=dropdown_options,
                                    value=default_var,
                                    style={
                                        'marginBottom': '25px',
                                        'height': '24px',
                                        'fontSize': '16px'
                                    }
                                ),
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
                                            'padding': '4px 8px'  # internal padding (vert, horiz)
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
                                            'padding': '4px 8px'  # internal padding (vert, horiz)
                                        }
                                    ),
                                ], style={'display': 'flex', 'marginBottom': '10px'}),
                                html.Button('Find Closest Point', id='find-point-button', n_clicks=0,
                                            style={'width': '100%', 'padding': '8px', 'backgroundColor': '#3498db',
                                                   'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                                   'cursor': 'pointer', 'marginBottom': '15px'}),
                            ]),

                            # Cost selector
                            html.Div([
                                html.Label(f"Cost Filter (Min={min_cost_value:.3f}/Max={max_cost_value:.3f}):",
                                           style={
                                               'fontWeight': 'bold',
                                               'marginBottom': '5px',
                                               'display': 'block',
                                               'fontSize': '15px'
                                            }),
                                html.Div([
                                    dcc.Input(
                                        id='cost-input',
                                        type='number',
                                        min=0,
                                        max=max_cost_value,
                                        value=max_cost_value,
                                        step=0.1,
                                        style={
                                            'width': '65%',
                                            'height': '24px',
                                            'fontSize': '12px',
                                            'marginRight': '5%'  # internal padding (vert, horiz)
                                        }
                                    ),
                                    html.Button('Apply', id='apply-cost-button', n_clicks=0,
                                                style={'width': '30%', 'padding': '8px', 'backgroundColor': '#27ae60',
                                                       'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                                       'cursor': 'pointer'}),
                                ], style={'display': 'flex', 'marginBottom': '30px'}),
                                html.Div(id='cost-input-message', style={'fontSize': '12px', 'color': '#7f8c8d'}),
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

                        # Clicked point properties section
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
                        ], style={
                            'padding': '15px',
                            'border': '1px solid #bdc3c7',
                            'borderRadius': '5px',
                            'backgroundColor': '#ffffff'
                        }),

                    ], style={
                        'flex': '0 0 25%',  # Fixed width, no grow/shrink
                        'marginRight': '1%'
                    }),

                    # MIDDLE COLUMN - Scatter Plot
                    html.Div([
                        dcc.Graph(
                            id='aerosol-plot',
                            figure=create_scatter_plot_only(filtered_data, default_var, original_indices),
                            style={'height': '800px', 'border': '1px solid #bdc3c7', 'borderRadius': '5px'}
                        ),
                    ], style={
                        'flex': '0 0 31%',  # Back to original size
                        'marginRight': '1%'
                    }),

                    # RIGHT COLUMN - Combined Intensity and DoLP plot
                    html.Div([
                        # Single combined plot with both intensity and DoLP as subplots
                        html.Div([
                            dcc.Graph(
                                id='combined-plot',
                                style={'height': '800px', 'border': '1px solid #bdc3c7', 'borderRadius': '5px'}  # Full height
                            ),
                        ]),

                    ], style={
                        'flex': '0 0 42%'
                    }),
                ], style={
                    'display': 'flex',
                    'flexDirection': 'row',
                    'padding': '0 20px',
                    'gap': '0px'  # No gap since we're using margins
                }),

            ], style={'backgroundColor': '#ecf0f1', 'minHeight': '100vh', 'padding': '20px 0'})
            # END of APP LAYOUT

            # Begin Callbacks
            # 1. DATA/FILE MANAGEMENT CALLBACKS (highest level)
            # File selector callback
            @app.callback(
                [Output('property-selector', 'options'),
                 Output('property-selector', 'value'),
                 Output('current-file-data', 'data'),
                 Output('cost-input', 'max'),
                 Output('cost-input', 'value'),
                 Output('clicked-point-store', 'data')],
                [Input('file-selector', 'value')]
            )
            def update_file_selection(selected_file_path):
                # Read new file
                try:
                    new_data_dict, new_sorted_variables, new_display_names, new_variable_metadata = read_hdf5_variables(selected_file_path)

                    # Get new max cost value
                    new_max_cost_value = np.nanmax(new_data_dict['cost_function'])

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
                    return (
                        new_dropdown_options,
                        new_default_var,
                        new_file_data,
                        new_max_cost_value,
                        new_max_cost_value,
                        None
                    )

                except Exception as e:
                    print(f"Error loading file {selected_file_path}: {str(e)}")
                    # If error, return current values
                    raise dash.exceptions.PreventUpdate

            # INPUT VALIDATION CALLBACKS
            # Cost function filter callback
            @app.callback(
                [Output('cost-input', 'value', allow_duplicate=True),
                 Output('cost-input-message', 'children')],
                [Input('apply-cost-button', 'n_clicks')],
                [State('cost-input', 'value'),
                 State('current-file-data', 'data')],
                prevent_initial_call='initial_duplicate'
            )
            def validate_cost_input(n_clicks, input_value, current_file_data):
                # Debugging
                if debug > 1:
                    print("Current file data:", current_file_data)

                if n_clicks == 0:
                    return no_update, ""

                # Safer to use default value if max_cost_value can't be found
                max_cost_value = 200.0
                if current_file_data is not None:
                    # Get current max cost val from store
                    max_cost_value = current_file_data.get('max_cost_value', 10.0)

                if debug > 1:
                    print("Max cost value:", max_cost_value)

                if input_value is None:
                    return max_cost_value, "Using maximum cost value"

                # Ensure cost val is within bounds
                if input_value < 0:
                    return 0, "Input was less than 0. Using minimum value (0)."

                if input_value > max_cost_value:
                    return max_cost_value, f"Input exceeded maximum. Using maximum value ({max_cost_value:.2f})."

                return input_value, f"Using cost threshold: {input_value:.2f}"

            # 3. UI SYNCHRONIZATION CALLBACKS
            @app.callback(
                    [Output('latitude-input', 'value'),
                     Output('longitude-input', 'value')],
                    [Input('aerosol-plot', 'clickData')],
                    [State('clicked-point-store', 'data'),
                     State('current-file-data', 'data')]
                    )
            def update_latlon_inputs(clickData, stored_point_data, current_file_data):
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
                            data_dict, _, _, _ = read_hdf5_variables(file_path)

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

            # MAIN VISUALIZATION CALLBACK (core functionality)
            @app.callback(
                [Output('aerosol-plot', 'figure'),
                 Output('combined-plot', 'figure'),
                 Output('clicked-point-store', 'data', allow_duplicate=True),
                 Output('click-info', 'children'),
                 Output('panel-properties-table', 'children')],
                [Input('property-selector', 'value'),
                 Input('cost-input', 'value'),
                 Input('aerosol-plot', 'clickData'),
                 Input('find-point-button', 'n_clicks'),
                 Input('current-file-data', 'data')],
                [State('latitude-input', 'value'),
                 State('longitude-input', 'value'),
                 State('clicked-point-store', 'data')],
                prevent_initial_call='initial_duplicate'
            )
            def update_all_plots(selected_property, max_cost, clickData, find_button_clicks,
                                 current_file_data, input_lat, input_lon, stored_point_data):

                # Determine which input triggered callback
                ctx = callback_context
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

                # Use the current file data instead of global data_dict
                if current_file_data is None:
                    empty_fig = go.Figure()
                    return (empty_fig, empty_fig, empty_fig, "No file selected", None, "No file selected", "")

                # Get current file path from store and read data
                file_path = current_file_data.get('file_path')
                if file_path is None:
                    empty_fig = go.Figure()
                    return (empty_fig, empty_fig, empty_fig, "No file path found", None, "No file path found", "")

                # Read data from current file path
                data_dict, sorted_variables, display_names, variable_metadata = \
                    read_hdf5_variables(file_path)

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
                            selected_row = original_flat_idx // original_shape[1]
                            selected_col = original_flat_idx % original_shape[1]
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
                        selected_row = clicked_original_flat_idx // original_shape[1]
                        selected_col = clicked_original_flat_idx % original_shape[1]
                        selected_point_idx = valid_original_indices[pointNumber]
                        clicked_point_data = {'row': selected_row, 'col': selected_col, 'original_idx': int(selected_point_idx)}
                    except Exception as e:
                        print(f"Error processing click: {e}")
                        clicked_point_data = stored_point_data
                else:
                    clicked_point_data = stored_point_data

                # Create the main scatter plot
                scatter_fig = create_scatter_plot_only(filtered_data, selected_property, original_indices, clicked_point_data, max_cost)

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
                        combined_fig = create_combined_intensity_dolp_plot(
                            intensity_data, dolp_data, wavelengths, wl_colors
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
                    lat = data_dict['latitude'][selected_row, selected_col]
                    lon = data_dict['longitude'][selected_row, selected_col]

                    if filtered_data[selected_property].ndim == 2:
                        val = filtered_data[selected_property][selected_row, selected_col]
                    else:
                        flat_idx = selected_row * original_shape[1] + selected_col
                        val = data_dict[selected_property][flat_idx]

                    if filtered_data['cost_function'].ndim == 2:
                        cost = data_dict['cost_function'][selected_row, selected_col]
                    else:
                        flat_idx = selected_row * original_shape[1] + selected_col
                        cost = data_dict['cost_function'][flat_idx]

                    click_info = html.Div([
                        html.Strong("Location: "), f"Lat {lat:.4f}°, Lon {lon:.4f}°",
                        html.Br(),
                        html.Strong("Selected: "), f"{selected_property} = {val:.3f}",
                        html.Br(),
                        html.Strong("Cost: "), f"{cost:.3f}"
                    ])

                    properties_table = create_properties_table(filtered_data, selected_row, selected_col)

                return (scatter_fig, combined_fig,
                        clicked_point_data, click_info, properties_table)

            # 5. EXPORT CALLBACKS
            # 1. First callback updates the status message immediately
            @app.callback(
                Output('export-status', 'children'),
                Input('export-button', 'n_clicks'),
                prevent_initial_call=True
            )
            def update_export_status(n_clicks):
                if n_clicks is None or n_clicks == 0:
                    return ""

                return "Processing export... ⏳"

            # 2. Second handles the actual export as png (updated to also use
            # current_file_data)
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
                if n_clicks is None or n_clicks == 0:
                    return no_update, no_update

                try:
                    # Get current file path
                    file_path = current_file_data.get('file_path')

                    # Read data from current file
                    data_dict, sorted_variables, display_names, variable_metadata = \
                        read_hdf5_variables(file_path)

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

            # Export as kml callback
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
                if n_clicks is None or n_clicks == 0:
                    return no_update

                try:
                    # Get current file path
                    file_path = current_file_data.get('file_path')

                    # Read data from current file
                    data_dict, sorted_variables, display_names, variable_metadata = \
                        read_hdf5_variables(file_path)

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
            app.run_server(debug=True, port=8050)

        else:
            print("No data available for plotting.")

    except Exception as e:
        print(f"Error setting up application: {str(e)}")


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
