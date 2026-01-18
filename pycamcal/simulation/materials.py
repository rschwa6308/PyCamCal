import numpy as np
import open3d


# TODO: Revisit this idea
# class Material:
#     color: tuple[float, float, float]
#     transparent: bool
#     mirror: bool


############ COLORS ############

BLACK   = [0.0, 0.0, 0.0]
WHITE   = [1.0, 1.0, 1.0]

RED     = [1.0, 0.0, 0.0]
GREEN   = [0.0, 1.0, 0.0]
BLUE    = [0.0, 0.0, 1.0]

YELLOW  = [1.0, 1.0, 0.0]
MAGENTA = [1.0, 0.0, 1.0]
CYAN    = [0.0, 0.0, 1.0]

GRAY_LIGHT  = [0.75, 0.75, 0.75]
GRAY_MEDIUM = [0.50, 0.50, 0.50]
GRAY_DARK   = [0.25, 0.25, 0.25]


ALL_COLORS = [BLACK, WHITE, RED, GREEN, BLUE, YELLOW, MAGENTA, CYAN, GRAY_LIGHT, GRAY_MEDIUM, GRAY_DARK]


############ MATERIALS ############

# NOTE: a material ID's must match its index in `ALL_COLORS` (to enable fast lookup)

# Basic materials
MAT_BLACK       = 0
MAT_WHITE       = 1
MAT_RED         = 2
MAT_GREEN       = 3
MAT_BLUE        = 4
MAT_YELLOW      = 5
MAT_MAGENTA     = 6
MAT_CYAN        = 7
MAT_GRAY_LIGHT  = 8
MAT_GRAY_MEDIUM = 9
MAT_GRAY_DARK   = 10

# Special materials
MAT_MIRROR      = 98
MAT_TRANSPARENT = 99


def lookup_material_color(mat_ids: np.ndarray):
    return np.array(ALL_COLORS)[mat_ids]

