import sys
from pathlib import Path
import os
DENSEMAPS_PATH = os.path.abspath(str(Path(__file__).parents[1] / "ScalableDenseMaps"))
if DENSEMAPS_PATH not in sys.path:
    sys.path.append(DENSEMAPS_PATH)