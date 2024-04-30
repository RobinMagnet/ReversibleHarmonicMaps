import sys
from pathlib import Path

DENSEMAPS_PATH = str(Path(__file__).parents[1] / "ScalableDenseMaps")
if DENSEMAPS_PATH not in sys.path:
    sys.path.append(DENSEMAPS_PATH)