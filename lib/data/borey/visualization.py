# add "import cartopy" to the top of your jupyter notebook,
# before using these functions, or visualizations will fail

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib import pyplot as plt


def fix_quiver_bug(field, lat):
    ufield, vfield = field
    old_magnitude = np.sqrt(ufield ** 2 + vfield ** 2)
    ufield_fixed = ufield / np.cos(np.radians(lat))
    new_magnitude = np.sqrt(ufield_fixed ** 2 + vfield ** 2)
    field_fixed = np.stack([ufield_fixed, vfield]) * old_magnitude / new_magnitude.clip(min=1e-6)
    return field_fixed


def create_cartopy():
    fig, ax = plt.subplots(
        figsize=(12, 12),
        subplot_kw={
            'projection': ccrs.NorthPolarStereo(central_longitude=45.0),
        }
    )

    # ax.add_feature(cfeature.OCEAN, zorder=0)  # significantly slows down the rendering
    ax.set_facecolor(cfeature.COLORS['water'])  # essentially the same, but fast
    ax.add_feature(cfeature.LAND, edgecolor='black', zorder=0)
    ax.gridlines(draw_labels=True, color='gray', zorder=9)
    return fig, ax


def visualize_scalar_field(ax, grid, field, vmin=None, vmax=None):
    layer = ax.pcolormesh(
        grid.lon,
        grid.lat,
        field,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        alpha=None,
    )
    plt.colorbar(layer, shrink=0.75)


def visualize_vector_field(ax, grid, field, key_length=50, key_units='cm/s', key_color='black'):
    field_fixed = fix_quiver_bug(field, grid.lat)  # fix bug in quiver vector plot interpretation
    layer = ax.quiver(
        grid.lon,
        grid.lat,
        field_fixed[0],
        field_fixed[1],
        transform=ccrs.PlateCarree(),
        color=key_color,
    )
    ax.quiverkey(layer, X=0.82, Y=0.2, U=key_length, label=f'{key_length} {key_units}',
                 labelpos='E', coordinates='figure')


def show_validation_table(rows, columns, data, title):
    # Calculate the mean values of each row
    row_means = np.mean(np.array([
        [data[i, j] for j in range(data.shape[1]) if i != j]
        for i in range(data.shape[0])
    ]), axis=1).reshape(-1, 1)

    # Create a new data array with an extra column for the means
    # Add a large negative value as a separator (will not be shown)
    separator = np.full((data.shape[0], 1), -np.inf)
    extended_data = np.hstack((data, separator, row_means))

    fig, ax = plt.subplots(figsize=(8, 7))  # Slightly wider figure to accommodate the extra column
    vmin = max(0, min(data[i, j] for i in range(len(rows)) for j in range(len(columns)) if i != j))
    vmax = max(data[i, j] for i in range(len(rows)) for j in range(len(columns)) if i != j)
    ax.imshow(extended_data, vmin=vmin, vmax=vmax, cmap='viridis')  # Use a colormap that highlights the data well

    # Adjust tick labels to include the mean column
    ax.set_xticks(np.arange(len(columns) + 2))  # +2 for the separator and mean column
    ax.set_xticklabels(columns + [''] + ['Mean'])  # Empty label for the separator column
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Annotations for the main data
    for i in range(len(rows)):
        for j in range(len(columns)):
            ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center', color='r')

    # Annotations for the mean values
    mean_column_index = len(columns) + 1  # Index of the mean column
    for i, mean_value in enumerate(row_means):
        ax.text(mean_column_index, i, f'{mean_value[0]:.1f}', ha='center', va='center', color='r')

    ax.set_title(title)
    fig.tight_layout()
