import logging
import matplotlib.patches as patches
from typing import List

# Setup logger for the package
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler("candlestick_plot.log", mode="w")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def add_rectangle(rects: List[patches.Rectangle], x: int, y1: float, y2: float, width: int = 1, **rect_params):
    """Utility to create a rectangle patch for matplotlib."""
    try:
        rects.append(
            patches.Rectangle(
                (x - 0.5, min(y1, y2)),
                width=width,
                height=abs(y2 - y1),
                facecolor=rect_params.get('fill_color', 'none'),
                edgecolor=rect_params.get('edge_color', 'black'),
                alpha=rect_params.get('alpha', 0.1),
                linewidth=rect_params.get('linewidth', 1)
            )
        )
    except Exception as e:
        logger.error(f"Error adding rectangle: {e}")
