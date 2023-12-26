"""
This will contain experimentation with the Clifford package.
"""

import clifford as cf
import math

layout, blades = cf.Cl(4)

B = blades['e1'] * blades['e2'] + blades['e3'] * blades['e4']

print(B)

print(math.e**(-B * math.pi / 4))

