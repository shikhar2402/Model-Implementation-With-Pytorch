from torch import nn
nn.RNN()

from enum import Enum

class test(Enum):
    PLAYING= 1

obj = test.PLAYING
print(test.PLAYING)
print(test.PLAYING.value)
print(test.PLAYING.name)
print(test)

# __repr__ vs __str__ demonstration
class Demo:
    def __init__(self, value):
        self.value = value
        
    # def __repr__(self):
    #     # __repr__ is meant for developers - should be detailed and unambiguous
    #     # Ideally should be a valid Python expression to recreate the object
    #     return f"Demo(value={self.value})"
        
    # def __str__(self):
    #     # __str__ is meant for end users - should be readable and concise
    #     # Used when printing object or converting to string
    #     return f"Value is {self.value}"

# Example usage
d = Demo(42)
print(d)
print(f"__str__: {str(d)}")  # Uses __str__
print(f"__repr__: {repr(d)}") # Uses __repr__

# If __str__ is not defined, Python will use __repr__ as fallback
# If __repr__ is not defined, default implementation shows object memory address
# Actually, having __repr__ is still valuable even if __str__ uses it as fallback:

# 1. Explicit is better than implicit - having both methods clearly shows intent
# 2. __repr__ should aim to be unambiguous and show full object state
# 3. __str__ can be more user-friendly/concise even if currently missing
# 4. Different use cases: print() vs debugging/development
# 5. Following the principle that __repr__ should ideally allow recreating the object

# So while they may return the same thing as a fallback,
# it's good practice to implement both with their distinct purposes
