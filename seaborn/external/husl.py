import operator
import math
__version__ = '2.1.0'
m = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.204, 1.057]]
m_inv = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
refX = 0.95047
refY = 1.0
refZ = 1.08883
refU = 0.19784
refV = 0.46834
lab_e = 0.008856
lab_k = 903.3