#!/usr/bin/env python
# coding: utf-8

# # Comparison between quantum computing and Classical computing.
# 
# ## Quantum computing
# 1. Quantum computing calculates qubits which can represent 0 and 1 at same time.
# 2. Power increases in proportion.
# 3. It have higher error rates 
# 4. It is high speed parallel computer based on quantum mechanics.
# 5. Information processing is carried out by Quantum logic gates.
# 6. Circuit behaviour is governed explicitly by quantum mechanics.
# 7. Severe restrictions exist on copying and measuring signals.
# 
# ## Classical computing
# 1. Classical computing calculates with the transistors which can represent 0 and 1.
# 2. Power increases in 1:1 relationship.
# 3. It have low error rates.
# 4. It is large scale integrated multi-purpose computer.
# 5. Information processing is carried out by logic gates e.g. NOT, AND, OR etc.
# 6. Circuit behaviour is governed by classical physics.
# 7. No restrictions exist on copying or measuring signals.

# # Deutsch’s algorithm
# ### Deutsch-Jozsa algorithm is one of the first quantum algorithms with nice speedup over its classical counterpart.
# ### The Deutsch–Jozsa problem is specifically designed to be easy for a quantum algorithm and hard for any deterministic classical algorithm. The motivation is to show a black box problem that can be solved efficiently by a quantum computer with no error, whereas a deterministic classical computer would need a large number of queries to the black box to solve the problem.
# ### The Deutsch-Jozsa algorithm was the first to show a separation between the quantum and classical difficulty of a problem. This algorithm demonstrates the significance of allowing quantum amplitudes to take both positive and negative values, as opposed to classical probabilities that are always non-negative.
# ### It can determine whether or not a function has a certain property (being balanced). The algorithm achieves this by requiring that the function (more precisely, a derivation of the function) need only be called once with a quantum algorithm instead of twice with a classical algorithm. When the function is very 'expensive', e.g., in terms of computational resources, it can be very beneficial if you have to compute this function only once instead of twice.Although the speed-up of this specific algorithm is only a factor of 2, other quantum algorithms, using the same quantum mechanical effects, can achieve a polynomial or even an exponential speed-up compared to classical algorithms.

# # An explanation of Deutsch’s algorithm and code simulating it using qiskit.

# In[14]:


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ, BasicAer
from qiskit.visualization import plot_bloch_multivector,plot_bloch_vector, plot_histogram
from qiskit.quantum_info import Statevector
import numpy as np 
import matplotlib


# In[15]:


backend = BasicAer.get_backend('qasm_simulator')
shots = 1024


# In[16]:


style = {'backgroundcolor': 'lightyellow'} # Style of the circuits


# In[17]:


qreg1 = QuantumRegister(2) # The quantum register of the qubits, in this case 2 qubits
register1 = ClassicalRegister(1) 

qc = QuantumCircuit(qreg1, register1)


# # Initial state

# In[18]:


qc.x(1)
qc.barrier()
qc.draw(output='mpl', style=style) 


# # Apply hadamard gates 

# In[19]:


qc.h(0)
qc.h(1)
qc.barrier()
qc.draw(output='mpl', style=style)


# In[20]:


qc.cx(0,1)
qc.barrier()
qc.draw(output='mpl', style=style)


# # Apply the Hadamard gate to the first (A) qubit.

# In[21]:


qc.h(0)
qc.draw(output='mpl', style=style) 


# In[22]:


qc.measure(qreg1[0],register1)
qc.draw(output='mpl', style=style) 


# # Execute and get counts.

# In[23]:


results = execute(qc, backend=backend, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)


# In[ ]:




