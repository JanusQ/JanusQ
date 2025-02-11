import random

from qiskit import QuantumCircuit, QuantumRegister

random.seed(555)


def gen_cc(qc, qr, nCoins):
    indexOfFalseCoin = random.randint(0, nCoins - 1)

    for i in range(nCoins):
        qc.h(qr[i])
    for i in range(nCoins):
        qc.cx(qr[i], qr[nCoins])
    qc.measure(qr[nCoins], cr[nCoins])

    qc.x(qr[nCoins]).c_if(cr, 0)
    qc.h(qr[nCoins]).c_if(cr, 0)

    for i in range(nCoins):
        qc.h(qr[i]).c_if(cr, 2 ** nCoins)
    qc.barrier()

    qc.cx(qr[indexOfFalseCoin], qr[nCoins]).c_if(cr, 0)
    qc.barrier()

    for i in range(nCoins):
        qc.h(qr[i]).c_if(cr, 0)

    # for i in range(nCoins):
    #     qc.measure(qr[i], cr[i])

def get_circuit(nCoins):
    n_qubits = nCoins + 1
    qr = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr)

    gen_cc(qc, qr, cr, nCoins)
    return qc