import matplotlib.patches as patches

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def zoom_in_rectangle(img, ax, zoom, rectangle_xy, rectangle_width, rectangle_height, **kwargs):
    """
    Parameters:
    -----------
        img: array-like
            The image data.
        ax: Axes
            Axes to place the inset axes.
        zoom: float
            Scaling factor of the data axes. zoom > 1 will enlargen the coordinates (i.e., "zoomed in"),
                while zoom < 1 will shrink the coordinates (i.e., "zoomed out").
        rectangle_xy: (float or int, float or int)
            The anchor point of the rectangle to be zoomed.
        rectangle_width: float or int
            Rectangle to be zoomed width.
        rectangle_height: float or int
            Rectangle to be zoomed height.

    Other Parameters:
    -----------------
        cmap: str or Colormap, default 'gray'
            The Colormap instance or registered colormap name used to map scalar data to colors.
        zoomed_inset_loc: int or str, default: 'upper right'
            Location to place the inset axes.
        zoomed_inset_lw: float or None, default 1
            Zoomed inset axes linewidth.
        zoomed_inset_col: float or None, default black
            Zoomed inset axes color.
        mark_inset_loc1: int or str, default is 1
            First location to place line connecting box and inset axes.
        mark_inset_loc2:  int or str, default is 3
            Second location to place line connecting box and inset axes.
        mark_inset_lw: float or None, default None
            Linewidth of lines connecting box and inset axes.
        mark_inset_ec: color or None
            Color of lines connecting box and inset axes.

    """
    axins = zoomed_inset_axes(ax, zoom, loc=kwargs.get("zoomed_inset_loc", 1))

    rect = patches.Rectangle(xy=rectangle_xy, width=rectangle_width, height=rectangle_height)
    x1, x2 = rect.get_x(), rect.get_x() + rect.get_width()
    y1, y2 = rect.get_y(), rect.get_y() + rect.get_height()

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    mark_inset(
        ax,
        axins,
        loc1=kwargs.get("mark_inset_loc1", 1),
        loc2=kwargs.get("mark_inset_loc2", 3),
        lw=kwargs.get("mark_inset_lw", None),
        ec=kwargs.get("mark_inset_ec", "1.0"),
    )

    axins.imshow(
        img,
        cmap=kwargs.get("cmap", "gray"),
        origin="lower",
        vmin=kwargs.get("vmin", None),
        vmax=kwargs.get("vmax", None),
    )

    for axis in ["top", "bottom", "left", "right"]:
        axins.spines[axis].set_linewidth(kwargs.get("zoomed_inset_lw", 1))
        axins.spines[axis].set_color(kwargs.get("zoomed_inset_col", "k"))

    axins.set_xticklabels([])
    axins.set_yticklabels([])
