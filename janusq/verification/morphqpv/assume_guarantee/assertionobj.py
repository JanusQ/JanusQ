
from typing import Any
from .operator import Operator
import jax.numpy as jnp
class AssertionObject:
    def __init__(self) -> None:
        pass

class State(AssertionObject):
    def __init__(self, value: jnp.array) -> None:
        self.value = value

class Expectation(AssertionObject):
    def __init__(self, operator: Operator) -> None:
        self.operator = operator
    def __call__(self, _value: State) -> Any:
        if isinstance(_value, float):
            return Scalar(_value)
        if len(_value.shape) > 1:
            return Scalar(DensityMatrix(_value).__expectation__(self.operator))
        else:
            return Scalar(StateVector(_value).__expectation__(self.operator))
    def __eq__(self, __value: object) -> bool:
        return 

class Distribution(AssertionObject):
    def __init__(self, assertion_value: list) -> None:
        self.value = assertion_value
    def __call__(self, state: State) -> bool:
        return state.value in self.value

class Scalar(AssertionObject):
    def __init__(self, value: float) -> None:
        self.value = value
    def __eq__(self, other: object) -> bool:
        return jnp.abs(self.value - other.value)


import jax.numpy as jnp

class StateVector(State):
    def __init__(self,value: jnp.array) -> None:
        assert len(value.shape) == 1, 'The dimension of statevector must be 1'
        super().__init__(value)
    def __ne__(self, __value: object) -> bool:
        return jnp.abs(jnp.mean(jnp.abs(self.value-__value.value))- 1)
    def __expectation__(self, operator: object) -> object:
        return self._to_density_matrix().__expectation__(operator)
    def __trace__(self) -> object:
        return self._to_density_matrix().__trace__()
    def __eq__(self, state: object) -> bool:
        return jnp.mean(jnp.abs(self.value-state.value))
    def __ispure__(self) -> bool:
        return jnp.abs(self.value @ self.value.conj().T - 1)
    def __isvalid__(self) -> bool:
        return self.__ispure__()
    def _to_density_matrix(self):
        return DensityMatrix(jnp.outer(self.value,self.value.conj()))

class DensityMatrix(State):
    def __init__(self,value: jnp.array) -> None:
        assert len(value.shape) == 2, 'The dimension of density matrix must be 2'
        super().__init__(value)
    def __expectation__(self, operator: object) -> object:
        return jnp.trace(self.value @ operator.value)
    def __trace__(self) -> object:
        return jnp.trace(self.value)
    def __eq__(self, state: object) -> bool:
        return jnp.mean(jnp.abs(self.value-state.value))
    def __ispure__(self) -> bool:
        return (jnp.mean(jnp.abs(self.value-self.value@self.value)) + jnp.abs(jnp.trace(self.value)-1))
    def __isvalid__(self) -> bool:
        return jnp.abs(jnp.trace(self.value)-1)
