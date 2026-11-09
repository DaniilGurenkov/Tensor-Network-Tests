from qiskit import QuantumCircuit, transpile
from qiskit.circuit import QuantumRegister
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer

from math import sqrt, asin, floor, pi
import matplotlib.pyplot as plt

"""
Our goal is to find two binary numbers of length 3, whose digits each sum up to the same value.
For our purposes, we will encode the two numbers as just one bitstring by concatenating them.
Examples: 101011 -> 1 + 0 + 1 == 0 + 1 + 1 -> True
          010001 -> 0 + 1 + 0 == 0 + 0 + 1 -> True
          001110 -> 0 + 0 + 1 != 1 + 1 + 0 -> False
"""

def partial_oracle():
    """
    The "forward" oracle: determine if our six bits evaluate to true for the given condition.
    DO NOT build in phase kickback, and don't bother to clear any ancilla bits you use.
    Return a QuantumCircuit.
    """
    # ---- create registers and circuit ----
    processing_register = QuantumRegister(6, name='p')
    sum_1 = QuantumRegister(2, name = 's1')
    sum_2 = QuantumRegister(2, name = 's2')
    aux_register = QuantumRegister(1, name = 'a')
    qc = QuantumCircuit(processing_register, sum_1, sum_2, aux_register)

    # ---- assign variables to make circuit construction easier ----
    q0, q1, q2 = processing_register[0], processing_register[1], processing_register[2]
    q3, q4, q5 = processing_register[3], processing_register[4], processing_register[5]
    s0, s1 = sum_1[0], sum_1[1]
    s2, s3 = sum_2[0], sum_2[1]
    a0 = aux_register[0]

    # ---- add gates, construct circuit ----

    # sum first three qubits into sum register
    qc.cx(q0, s0)
    qc.cx(q1, s0)
    qc.cx(q2, s0)

    qc.ccx(q0, q1, s1)
    qc.ccx(q1, q2, s1)
    qc.ccx(q0, q2,  s1)

    # sum next three qubits into sum register
    qc.cx(q3, s2)
    qc.cx(q4, s2)
    qc.cx(q5, s2)

    qc.ccx(q3, q4, s3)
    qc.ccx(q4, q5, s3)
    qc.ccx(q3, q5, s3)

    # check for equality of sums
    qc.cx(s0, s2)
    qc.cx(s1, s3)
    qc.x(s2)
    qc.x(s3)
    qc.ccx(s2, s3, a0)
    return qc



def oracle():
    """
    Use the partial oracle and create the full oracle; i.e. a circuit U such that
    U|x> = (-1)^(f(x)) |x>
    
    Make sure to clear your ancilla bits!
    Return a QuantumCircuit.
    """
    # get the partial oracle
    # TODO
    partial = partial_oracle()
    # get the inverse of the partial oracle to clear ancilla bits
    # TODO
    inv = partial.inverse()

    # convert the above to gates, attach labels for clarity
    # TODO
    partial_gate = partial.to_gate(label = "partial oracle")
    inv_gate = inv.to_gate(label = "inverse partial oracle")

    # create the full oracle circuit
    # notice the extra bit and the CCX gate used for phase kickback
    # TODO
    full_oracle = QuantumCircuit(12, name = "oracle")
    full_oracle.append(partial_gate, list(range(11)))
    full_oracle.x(11)
    full_oracle.h(11)
    full_oracle.cx(10, 11)
    full_oracle.append(inv_gate, list(range(11)))
    return full_oracle

pc = oracle()
pc.draw(output = "mpl")
plt.show()

def diffuser(n: int):
    """
    Create the diffusion operation.
    """
    # define circuit and helpful registers
    # TODO

    # apply hadamard and x-gates
    # TODO

    # apply multi-controlled z gate
    # TODO

    # apply x-gates and hadamards
    # TODO

def num_iter(num_sol: int, total: int):
    """
    Find the optimal number of iterations to apply the Grover's operation.
    """
    # TODO

def grover_circ():
    """
    Create the entire circuit.
    """
    # get oracle and diffuser circuits
    # optionally convert them to gates, easier to see when printing
    # TODO

    # create circuit, initialize qubit states 
    # TODO

    # apply grover and diffuser operators the appropriate number of times
    # TODO

def histogram(qc: QuantumCircuit, n: int):
    """
    Create a histogram of the given circuit qc for n shots.
    """
    backend = Aer.get_backend("aer_simulator")
    tqc = transpile(qc, backend)
    counts = backend.run(tqc, shots = n).result().get_counts()
    return counts

def sample(qc: QuantumCircuit):
    """
    Sample once from the final state of the circuit.
    """
    backend = Aer.get_backend("aer_simulator")
    tqc = transpile(qc, backend)
    counts = backend.run(tqc, shots = 1).result().get_counts()
    val = list(counts.keys())[0]
    return val

# circ = grover_circ()
# circ.draw(output="mpl")
# plot_histogram(histogram(circ, 1000))
# plt.show()
