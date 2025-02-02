
from .assertionobj import AssertionObject,State,StateVector

__all__ = ['IsPure','Equal','NotEqual','Contains','GreaterThan','LessThan','GreaterThanOrEqual','LessThanOrEqual']
class Predicate:
    ''' A constraint '''
    def __init__(self):
        pass
    def __repr__(self):
        """representation of the predicate"""
        return f'{self.Literal}'

class IsPure(Predicate):
    Literal = 'IsPure'
    def __init__(self, assertion_object: AssertionObject = StateVector) -> None:
        self.object = assertion_object
    def __call__(self, state: AssertionObject) -> bool:
        state = self.object(state)
        return state.__ispure__()
    
class Equal(Predicate):
    Literal = 'Equal'
    def __init__(self, assertion_object: AssertionObject = StateVector) -> None:
        self.object = assertion_object
    def __call__(self, ObjectA: AssertionObject, ObjectB: AssertionObject) -> bool:
        ObjectA = self.object(ObjectA)
        ObjectB = self.object(ObjectB)
        return ObjectA == ObjectB

class NotEqual(Predicate):
    Literal = 'NotEqual'
    def __init__(self, assertion_object: AssertionObject = StateVector ) -> None:
        self.object = assertion_object
    def __call__(self, ObjectA: AssertionObject, ObjectB: AssertionObject) -> bool:
        ObjectA = self.object(ObjectA)
        ObjectB = self.object(ObjectB)
        return ObjectA != ObjectB

class Contains(Predicate):
    Literal = 'Contains'
    def __init__(self, assertion_object: AssertionObject) -> None:
        self.object = assertion_object
    def __call__(self, Object: AssertionObject, ObjectList: list) -> bool:
        return Object in ObjectList

class GreaterThan(Predicate):
    Literal = 'GreaterThan'
    def __init__(self, assertion_object: AssertionObject) -> None:
        self.object = assertion_object
    def __call__(self, ObjectA: AssertionObject, ObjectB: AssertionObject) -> bool:
        return ObjectA > ObjectB
    

class LessThan(Predicate):
    Literal = 'LessThan'
    def __init__(self, assertion_object: AssertionObject) -> None:
        self.object = assertion_object
    def __call__(self, ObjectA: AssertionObject, ObjectB: AssertionObject) -> bool:
        return ObjectA < ObjectB
    
class GreaterThanOrEqual(Predicate):
    Literal = 'Geq'
    def __init__(self, assertion_object: AssertionObject) -> None:
        self.object = assertion_object
    def __call__(self, ObjectA: AssertionObject, ObjectB: AssertionObject) -> bool:
        return ObjectA >= ObjectB
    
class LessThanOrEqual(Predicate):
    Literal = 'Leq'
    def __init__(self, assertion_object: AssertionObject) -> None:
        self.object = assertion_object
    def __call__(self, ObjectA: AssertionObject, ObjectB: AssertionObject) -> bool:
        return ObjectA <= ObjectB