from single_access_machine import alloc, read, write, get_global_counter, set_max_reads, set_max_writes
import copy
import random
from sys import maxsize
from attribute_at_positions import AttributeAtPositions
from typing import List, Dict, Tuple, Set, FrozenSet, Type, Any

"""
RecursivePointer - Pointer class built on multi read / multi write SAM
"""

class RecursivePointer:
    # pointer label  
    label = "recursive"

    # uses SAM addresses when True
    pointer_osam = True

    # label for tracking which object incurs SAM accesses
    structure = "RecursivePointer"

    # simulate reference count single addresses can be shared
    reference_counts: Dict[int, int] = dict()
    
    def __init__(self, head: int | Type["RecursivePointer"] | None) -> None:
        """Initialize RecursivePointer from different origins"""
        if isinstance(head, int) or head is None:
            self.head = head
        elif isinstance(head, RecursivePointer):
            self.head = head.head
        else:
            raise ValueError(f"Invalid type for RecursivePointer initialization: {type(head)}")

    def __repr__(self) -> str:
        """String representation of RecursivePointer"""
        return str(self.head)

    @staticmethod
    def new(value: Any = None) -> int:
        """Create a reference to a new pointee"""
        addr = alloc(RecursivePointer.pointer_osam, RecursivePointer.structure)
        write(addr, value, RecursivePointer.structure)
        if addr in RecursivePointer.reference_counts:
            RecursivePointer.reference_counts[addr] += 1
        else:
            RecursivePointer.reference_counts[addr] = 1
        return addr

    @staticmethod
    def is_single_reference(p: "RecursivePointer") -> bool:
        """Checks if p is the only reference to its pointee"""
        if isinstance(p, RecursivePointer) and isinstance(p.head, int):
            return RecursivePointer.reference_counts[p.head] == 1
        return False

    @staticmethod
    def __smart_copy_pointer(p0: "RecursivePointer") -> "RecursivePointer":
        """Create a Smart Copy p1 that points to p0's pointee"""
        if p0.head is None:
            return RecursivePointer(None)
        RecursivePointer.reference_counts[p0.head] += 1
        p1 = RecursivePointer(p0.head)
        return p1

    # @staticmethod
    # def copy(p: Any, temp_copy: bool = False) -> Any:
    #     """Handles copies of any object"""
    #     if isinstance(p, RecursivePointer):
    #         return RecursivePointer.__smart_copy_pointer(p)
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
                struct2.append(RecursivePointer.copy(elem, temp_copy))
            return struct2

        def copy_dict(struct1: Dict[Any, Any]) -> Dict[Any, Any]:
            """Helper function that smart copies a dict"""
            struct2: Dict[Any, Any] = dict()
            for key, value in struct1.items():
                struct2[key] = RecursivePointer.copy(value, temp_copy)
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
        elif isinstance(content, RecursivePointer):
            return RecursivePointer.__smart_copy_pointer(content)
        elif hasattr(content, "smart_copy") and callable(content.smart_copy):
            return content.smart_copy(content, temp_copy)
        elif content is not None:
            return copy.deepcopy(content)

    @staticmethod
    def get(p: "RecursivePointer") -> Any:
        """Return p's pointee"""
        assert isinstance(p.head, int)
        return read(p.head, RecursivePointer.structure)
    
    @staticmethod 
    def get_and_copy(p: "RecursivePointer") -> Any:
        """Return a copy of p's pointee"""
        value = RecursivePointer.get(p)
        return RecursivePointer.copy(value)

    @staticmethod
    def get_attr(   
            p: "RecursivePointer", 
            attributes: AttributeAtPositions | List[str] | str,
            make_copies: bool = True
        ) -> Any:
        """Retrieve specified attributes of a pointee object"""
        value = RecursivePointer.get(p)
        content: Dict[str, Any] | str | None

        # get an attribute that is a list at specific positions only
        if isinstance(attributes, AttributeAtPositions):
            content = dict()
            try: 
                list_attr = getattr(value, attributes.attribute)
                sublist = []
                for pos in attributes.positions:
                    elem = list_attr[pos]
                    if make_copies: elem = RecursivePointer.copy(elem, temp_copy=make_copies)
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
                    if make_copies: elem = RecursivePointer.copy(elem, temp_copy=make_copies)
                    content[attr] = elem
                except:
                    content[attr] = None

        # get one attribute
        elif type(attributes) == str:
            try:
                content = getattr(value, attributes)
                if make_copies: content = RecursivePointer.copy(content, temp_copy=make_copies)
            except:
                content = None
        
        else:
            raise TypeError("Attributes must be a list, a string, or a AttributeAtPositions instance.")

        return content

    @staticmethod
    def put(p: "RecursivePointer", value: Any, delete_old: bool = True) -> None:
        """Put new pointee value at p"""
        if delete_old:
            old_value = RecursivePointer.get(p)
            RecursivePointer.delete(old_value)
        # write a copy instead of the original
        assert isinstance(p.head, int)
        write(p.head, RecursivePointer.copy(value), RecursivePointer.structure)
    
    @staticmethod
    def put_attr(p: "RecursivePointer", attributes: Dict[str, Any], delete_old: bool = False) -> None:
        """Update specified attributes at a pointee object"""
        value = RecursivePointer.get(p)
        for attr in attributes.keys():
            if delete_old and hasattr(value, attr):
                RecursivePointer.delete(getattr(value, attr))
            setattr(value, attr, attributes[attr])
        assert isinstance(p.head, int)
        write(p.head, value, RecursivePointer.structure)
    
    @staticmethod
    def delete(p: Any) -> None:
        """Handles deletes for RecursivePointer and other objects"""
        if isinstance(p, RecursivePointer):
            if p.head is not None:
                RecursivePointer.reference_counts[p.head] -= 1
                p.head = None
        elif type(p) in [list, tuple, set, frozenset]:
            for elem in p:
                RecursivePointer.delete(elem)
        elif isinstance(p, dict):
            for key, value in p.items():
                RecursivePointer.delete(key)
                RecursivePointer.delete(value)
        elif hasattr(p, "smart_delete"):
            p.smart_delete()
    
    @staticmethod
    def get_count(p: "RecursivePointer") -> int:
        """Return number of references to p's pointee"""
        assert isinstance(p.head, int)
        return RecursivePointer.reference_counts[p.head]




def smart_pointer_test_1() -> None:
    num_ops = get_global_counter()

    p1 = RecursivePointer(RecursivePointer.new("cat"))
    print("Values initial no cp,",p1.head)
    p2 = RecursivePointer.copy(p1)
    print("First copy,", get_global_counter())
    num_ops = get_global_counter()
    # print("Post copy values ",p1.head, p2.head, p3.head)
    print(RecursivePointer.get(p1) == "cat")

    assert(RecursivePointer.get(p1) == "cat")
    print("First get,", get_global_counter())
    #Check if you can get the same value twice
    assert(RecursivePointer.get(p1) == "cat")
    print("Second get,", get_global_counter())
    num_ops = get_global_counter()
    p3 = RecursivePointer.copy(p1)
    print("Second copy,", get_global_counter())
    num_ops = get_global_counter()
    assert(RecursivePointer.get(p1) == "cat")
    print("Third get,", get_global_counter())
    num_ops = get_global_counter()
    RecursivePointer.put(p1, "dog")
    print("Put,", get_global_counter())
    num_ops = get_global_counter()
    for i in range(10):
        assert(RecursivePointer.get(p1) == "dog")
        print("Get,", get_global_counter())
        num_ops = get_global_counter()
        assert(RecursivePointer.get(p2) == "dog")
        print("Get,", get_global_counter())
        num_ops = get_global_counter()
    p4 = RecursivePointer(RecursivePointer.new(p1.get(p2)))
    assert(RecursivePointer.get(p4) == "dog")
    print("Get three copies,", get_global_counter())
    RecursivePointer.put(p1, "horse")
    assert(RecursivePointer.get(p2) == "horse")
    print("Get three copies,", get_global_counter())
    assert(RecursivePointer.get(p4) == "dog")
    print("Get p4 unrelated,", get_global_counter())
    RecursivePointer.delete(p3)
    print("Get post delete p3,", get_global_counter())
    assert(RecursivePointer.get(p2) == "horse")
    print("Get two copies,", get_global_counter())
    RecursivePointer.delete(p2)
    print("Get post delete p2,", get_global_counter())
    assert(RecursivePointer.get(p1) == "horse")
    print("Get post delete p2,", get_global_counter())
    RecursivePointer.delete(p1)
    RecursivePointer.delete(p4)
    print("Test 1 completed successfully.")

def smart_pointer_test_2() -> None:
    animal_list = ["cow", "cat", "dog", "horse", "sheep", "pig", "goat", "chicken"]
    smart_pointer_list = []
    for animal in animal_list:
        sp = RecursivePointer(RecursivePointer.new(animal))
        smart_pointer_list.append(sp)
    print("Loaded smart pointers with animals:", get_global_counter())

    pointer_to_pointer = RecursivePointer(RecursivePointer.new(smart_pointer_list))
    print("Loaded pointer to pointers:", get_global_counter())
    list_of_ptrs = RecursivePointer.get(pointer_to_pointer)
    i = 0
    for ptr in list_of_ptrs:
        assert(RecursivePointer.get(ptr)==animal_list[i])
        i += 1
    list_of_ptrs= RecursivePointer.get_and_copy(pointer_to_pointer)
    i = 0
    for ptr in list_of_ptrs:
        assert(RecursivePointer.get(ptr)==animal_list[i])
        i += 1

    print("Test 2 completed successfully with animals:",get_global_counter())

if __name__ == "__main__":
    print("Allocs, Reads, Writes")
    set_max_reads(maxsize)
    set_max_writes(maxsize)
    smart_pointer_test_1()
    smart_pointer_test_2()

