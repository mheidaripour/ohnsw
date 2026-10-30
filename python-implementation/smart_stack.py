from single_access_machine import alloc, read, write, return_available_address
from typing import Tuple, Any

"""
SmartStack - stack built on SAM accesses
"""

class SmartStack:
    # uses SAM addresses when True
    stack_osam = True  

    # label for tracking which object incurs SAM accesses
    structure = "SmartStack"

    @staticmethod
    def init_stack() -> None:
        """Initialize a SAM-built stack"""
        top = None # stack is currently empty
        return top

    @staticmethod
    def push(top: int | None, value: Any) -> int:
        """Push value onto stack and returns updated top"""
        new_top = alloc(SmartStack.stack_osam, SmartStack.structure)
        write(new_top, (value, top), SmartStack.structure)
        return new_top

    @staticmethod
    def pop(top: int | None) -> Tuple[Any, int | None]: 
        """Pop last value pushed"""
        # trying to pop from an empty stack
        if top is None:
            return None, None

        next_entry = read(top, SmartStack.structure)
        return_available_address(top)

        if isinstance(next_entry, tuple):
            return next_entry

        else:
            return None, None




def test_stack() -> None:
    top = SmartStack.init_stack()
    print("Initial Stack:", top)

    top = SmartStack.push(top, "cat")
    print("Push:", top)

    top = SmartStack.push(top, "dog")
    print("Push:", top)

    value, top = SmartStack.pop(top)
    print("Pop:", top, value)

    top = SmartStack.push(top, "horse")
    print("Push:", top)

    value, top  = SmartStack.pop(top)
    print("Pop:", top, value)

    value, top  = SmartStack.pop(top)
    print("Pop:", top, value)

    value, top  = SmartStack.pop(top)
    print("Pop:", top, value)

if __name__ == "__main__":
    test_stack()    
