"""
give the definition of the operators of quantum observables, and the operators of quantum operations
"""
import numpy as np
__all__ = ['pauliX','pauliY','pauliZ','hadamard']

class Operator:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __matmul__(self, other):
        if not isinstance(other,Operator):
            return NotImplemented
        if self.value.shape!= other.value.shape:
            return NotImplemented
        tensor = np.tensordot(self.value,other.value,axes=0)
        ## reshape the tensor to a 2D matrix
        tensor = tensor.reshape(self.value.shape[0]*other.value.shape[0],self.value.shape[1]*other.value.shape[1])
        return Operator(self.name+"@"+other.name,tensor)
    def __mul__(self, other):
        return NotImplemented
    
    
class PauliX(Operator):
    def __init__(self):
        super().__init__('PauliX',np.array([[0,1],[1,0]]))
    
    def __mul__(self, other):
        if isinstance(other,PauliX):
            return np.eye(2)
        if isinstance(other,PauliY):
            return PauliZ()
        if isinstance(other,PauliZ):
            return -PauliY()
        if isinstance(other,Hadamard):
            return PauliZ()
        return NotImplemented

class PauliY(Operator):
    def __init__(self):
        super().__init__('PauliY',np.array([[0,-1j],[1j,0]]))

class PauliZ(Operator):
    def __init__(self):
        super().__init__('PauliZ',np.array([[1,0],[0,-1]]))

class Hadamard(Operator):
    def __init__(self):
        super().__init__('Hadamard',1/np.sqrt(2)*np.array([[1,1],[1,-1]]))


pauliY = PauliY()
pauliX = PauliX()
pauliZ = PauliZ()
hadamard = Hadamard()



if __name__ == '__main__':

    print(np.array([[1,0],[0,-1]]).dot(PauliX().value))
    

