from single_access_machine import alloc, read, write, get_global_counter
import copy
import random
from attribute_at_positions import AttributeAtPositions
from typing import List, Dict, Tuple, Set, FrozenSet, Type, Any

"""
PlaintextPointer - Pointer class that does incur SAM stats
"""

class PlaintextPointer:
    # pointer label  
    label = "plaintext"

    # do not use SAM addresses
    pointer_osam = False
    
    # label for tracking which object incurs SAM accesses
    structure = "PlaintextPointer"

    # simulate reference count single addresses can be shared
    reference_counts: Dict[int, int] = dict()

    def __init__(self, head: int | Type["PlaintextPointer"] | None) -> None:
        """Initialize PlaintextPointer from different origins"""
        if isinstance(head, int) or head is None:
            self.head = head
        elif isinstance(head, PlaintextPointer):
            self.head = head.head
        else:
            raise ValueError(f"Invalid type for PlaintextPointer initialization: {type(head)}")

    def __repr__(self) -> str:
        """String representation of PlaintextPointer"""
        return str(self.head)

    @staticmethod
    def new(value: Any = None) -> int:
        """Create a reference to a new pointee"""
        addr = alloc(PlaintextPointer.pointer_osam, PlaintextPointer.structure)
        write(addr, value, PlaintextPointer.structure)
        if addr in PlaintextPointer.reference_counts:
            PlaintextPointer.reference_counts[addr] += 1
        else:
            PlaintextPointer.reference_counts[addr] = 1
        return addr

    @staticmethod
    def is_single_reference(p: "PlaintextPointer") -> bool:
        """Checks if p is the only reference to its pointee"""
        if isinstance(p, PlaintextPointer) and isinstance(p.head, int):
            return PlaintextPointer.reference_counts[p.head] == 1
        return False

    @staticmethod
    def __smart_copy_pointer(p0: "PlaintextPointer") -> "PlaintextPointer":
        """Create a Smart Copy p1 that points to p0's pointee"""
        if p0.head is None:
            return PlaintextPointer(None)
        PlaintextPointer.reference_counts[p0.head] += 1
        p1 = PlaintextPointer(p0.head)
        return p1

     # @staticmethod
    # def copy(p: Any, temp_copy: bool = False) -> Any:
    #     """Handles copies of any object"""
    #     if isinstance(p, PlaintextPointer):
    #         return PlaintextPointer.__smart_copy_pointer(p)
    #     elif hasattr(p, "smart_copy") and callable(p.smart_copy):
    #         return p.smart_copy(p, temp_copy)
    #     else:
    #         return copy.deepcopy(p)

    @staticmethod
    def copy(content: Any, temp_copy: bool = False) -> Any:
        """Handles copies of any object"""
        def copy_list(struct1: List[Any] | Tuple[Any] | Set[Any] | FrozenSet[Any]) -> List[Any]:
            """Helper function that smart copies a list"""
            struct2 = []
            for elem in struct1:
                struct2.append(PlaintextPointer.copy(elem, temp_copy))
            return struct2

        def copy_dict(struct1: Dict[Any, Any]) -> Dict[Any, Any]:
            """Helper function that smart copies a dict"""
            struct2: Dict[Any, Any] = dict()
            for key, value in struct1.items():
                struct2[key] = PlaintextPointer.copy(value, temp_copy)
            return struct2

        if isinstance(content, list):
            return copy_list(content)
        elif isinstance(content, tuple):
            return tuple(copy_list(content))
        elif isinstance(content, set):
            return set(copy_list(content))
        elif isinstance(content, frozenset):
            return frozenset(copy_list(content))
        elif isinstance(content, dict):
            return copy_dict(content)
        elif isinstance(content, PlaintextPointer):
            return PlaintextPointer.__smart_copy_pointer(content)
        elif hasattr(content, "smart_copy") and callable(content.smart_copy):
            return content.smart_copy(content, temp_copy)
        elif content is not None:
            return copy.deepcopy(content)

    @staticmethod
    def get(p: "PlaintextPointer") -> Any:
        """Return p's pointee"""
        assert isinstance(p.head, int)
        return read(p.head, PlaintextPointer.structure)
    
    @staticmethod 
    def get_and_copy(p: "PlaintextPointer") -> Any:
        """Return a copy of p's pointee"""
        value = PlaintextPointer.get(p)
        return PlaintextPointer.copy(value)

    @staticmethod
    def get_attr(   
            p: "PlaintextPointer", 
            attributes: AttributeAtPositions | List[str] | str,
            make_copies: bool = True
        ) -> Any:
        """Retrieve specified attributes of a pointee object"""
        value = PlaintextPointer.get(p)
        content: Dict[str, Any] | str | None

        # get an attribute that is a list at specific positions only
        if isinstance(attributes, AttributeAtPositions):
            content = dict()
            try: 
                list_attr = getattr(value, attributes.attribute)
                sublist = []
                for pos in attributes.positions:
                    elem = list_attr[pos]
                    if make_copies: elem = PlaintextPointer.copy(elem, temp_copy=make_copies)
                    sublist.append(elem)
                content[attributes.attribute] = sublist
            except:
                content[attributes.attribute] = None

        # get a list of attributes
        elif isinstance(attributes, list):
            content = dict()
            for attr in attributes:
                try:
                    elem = getattr(value, attr)
                    if make_copies: elem = PlaintextPointer.copy(elem, temp_copy=make_copies)
                    content[attr] = elem
                except:
                    content[attr] = None

        # get one attribute
        elif type(attributes) == str:
            try:
                content = getattr(value, attributes)
                if make_copies: content = PlaintextPointer.copy(content, temp_copy=make_copies)
            except:
                content = None
        
        else:
            raise TypeError("Attributes must be a list, a string, or a AttributeAtPositions instance.")

        return content

    @staticmethod
    def put(p: "PlaintextPointer", value: Any, delete_old: bool = True) -> None:
        """Put new pointee value at p"""
        if delete_old:
            old_value = PlaintextPointer.get(p)
            PlaintextPointer.delete(old_value)
        # write a copy instead of the original
        assert isinstance(p.head, int)
        write(p.head, PlaintextPointer.copy(value), PlaintextPointer.structure)
    
    @staticmethod
    def put_attr(p: "PlaintextPointer", attributes: Dict[str, Any], delete_old: bool = False) -> None:
        """Update specified attributes at a pointee object"""
        value = PlaintextPointer.get(p)
        for attr in attributes.keys():
            if delete_old and hasattr(value, attr):
                PlaintextPointer.delete(getattr(value, attr))
            setattr(value, attr, attributes[attr])
        assert isinstance(p.head, int)
        write(p.head, value, PlaintextPointer.structure)
    
    @staticmethod
    def delete(p: Any) -> None:
        """Handles deletes for PlaintextPointer and other objects"""
        if isinstance(p, PlaintextPointer):
            if p.head is not None:
                PlaintextPointer.reference_counts[p.head] -= 1
                p.head = None
        elif type(p) in [list, tuple, set, frozenset]:
            for elem in p:
                PlaintextPointer.delete(elem)
        elif isinstance(p, dict):
            for key, value in p.items():
                PlaintextPointer.delete(key)
                PlaintextPointer.delete(value)
        elif hasattr(p, "smart_delete"):
            p.smart_delete()
    
    @staticmethod
    def get_count(p: "PlaintextPointer") -> int:
        """Return number of references to p's pointee"""
        assert isinstance(p.head, int)
        return PlaintextPointer.reference_counts[p.head]
    
def smart_pointer_test_1() -> None:
    num_ops = get_global_counter()

    p1 = PlaintextPointer(PlaintextPointer.new("cat"))
    print("Values initial no cp,",p1.head)
    p2 = PlaintextPointer.copy(p1)
    print("First copy,", get_global_counter())
    num_ops = get_global_counter()
    # print("Post copy values ",p1.head, p2.head, p3.head)
    print(PlaintextPointer.get(p1) == "cat")

    assert(PlaintextPointer.get(p1) == "cat")
    print("First get,", get_global_counter())
    #Check if you can get the same value twice
    assert(PlaintextPointer.get(p1) == "cat")
    print("Second get,", get_global_counter())
    num_ops = get_global_counter()
    p3 = PlaintextPointer.copy(p1)
    print("Second copy,", get_global_counter())
    num_ops = get_global_counter()
    assert(PlaintextPointer.get(p1) == "cat")
    print("Third get,", get_global_counter())
    num_ops = get_global_counter()
    PlaintextPointer.put(p1, "dog")
    print("Put,", get_global_counter())
    num_ops = get_global_counter()
    for i in range(10):
        assert(PlaintextPointer.get(p1) == "dog")
        print("Get,", get_global_counter())
        num_ops = get_global_counter()
        assert(PlaintextPointer.get(p2) == "dog")
        print("Get,", get_global_counter())
        num_ops = get_global_counter()
    p4 = PlaintextPointer(PlaintextPointer.new(p1.get(p2)))
    assert(PlaintextPointer.get(p4) == "dog")
    print("Get three copies,", get_global_counter())
    PlaintextPointer.put(p1, "horse")
    assert(PlaintextPointer.get(p2) == "horse")
    print("Get three copies,", get_global_counter())
    assert(PlaintextPointer.get(p4) == "dog")
    print("Get p4 unrelated,", get_global_counter())
    PlaintextPointer.delete(p3)
    print("Get post delete p3,", get_global_counter())
    assert(PlaintextPointer.get(p2) == "horse")
    print("Get two copies,", get_global_counter())
    PlaintextPointer.delete(p2)
    print("Get post delete p2,", get_global_counter())
    assert(PlaintextPointer.get(p1) == "horse")
    print("Get post delete p2,", get_global_counter())
    PlaintextPointer.delete(p1)
    PlaintextPointer.delete(p4)
    print("Test 1 completed successfully.")

def smart_pointer_test_2() -> None:
    animal_list = ["cow", "cat", "dog", "horse", "sheep", "pig", "goat", "chicken"]
    smart_pointer_list = []
    for animal in animal_list:
        sp = PlaintextPointer(PlaintextPointer.new(animal))
        smart_pointer_list.append(sp)
    print("Loaded smart pointers with animals:", get_global_counter())

    pointer_to_pointer = PlaintextPointer(PlaintextPointer.new(smart_pointer_list))
    print("Loaded pointer to pointers:", get_global_counter())
    list_of_ptrs = PlaintextPointer.get(pointer_to_pointer)
    i = 0
    for ptr in list_of_ptrs:
        assert(PlaintextPointer.get(ptr)==animal_list[i])
        i += 1
    list_of_ptrs= PlaintextPointer.get_and_copy(pointer_to_pointer)
    i = 0
    for ptr in list_of_ptrs:
        assert(PlaintextPointer.get(ptr)==animal_list[i])
        i += 1

    print("Test 2 completed successfully with animals:",get_global_counter())

if __name__ == "__main__":
    print("Allocs, Reads, Writes")
    smart_pointer_test_1()
    smart_pointer_test_2()
