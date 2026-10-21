
# Auto-generated incremental update
import numpy as np

def calculate_evi(nir, red, blue, g=2.5, c1=6.0, c2=7.5, l=1.0):
    denom = nir + c1 * red - c2 * blue + l
    return np.where(denom != 0, g * (nir - red) / denom, 0.0)


# Auto-generated incremental update
import numpy as np

def calculate_evi(nir, red, blue, g=2.5, c1=6.0, c2=7.5, l=1.0):
    denom = nir + c1 * red - c2 * blue + l
    return np.where(denom != 0, g * (nir - red) / denom, 0.0)

