import numpy as np
import matplotlib.pyplot as plt
from qutip import Bloch, rand_ket

# Número de qubits
num_qubits = 1

# aleatory state of qubit
state = rand_ket(2**num_qubits)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sphere = Bloch(fig=fig, axes=ax)

sphere.add_states(state)

sphere.show()
plt.show()
