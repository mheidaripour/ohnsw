from smart_pointer_original import SmartPointerOriginal
from smart_pointer_multi_write import SmartPointerMultiWrite
from smart_pointer_multi_write_rary import SmartPointerMultiWriteRary
from recursive_pointer import RecursivePointer
from plaintext_pointer import PlaintextPointer
from single_access_machine import get_global_counter, set_max_reads, set_max_writes, clear_available_addresses
from sys import maxsize
from typing import List, Set, Dict, Tuple, FrozenSet, Any
from attribute_at_positions import AttributeAtPositions
import copy

"""
SmartPointer - Wrapper class that abstracts multiple pointer types
"""

class SmartPointer:
    # pointer label  
    label = "multiwrite"
    
    def __init__(   
            self, 
            head: int | SmartPointerOriginal | SmartPointerMultiWrite | RecursivePointer | SmartPointerMultiWriteRary | PlaintextPointer | None,
            temp_copy: bool = False
        ) -> None:
        """Intialize a SmartPointer wrapper object"""
        # somtimes we create temporary pointers during an algorithm
        # and would like to delete them afterward
        # the temp_copy flag controls this
        self.temp_copy = temp_copy

        match head:
            case SmartPointerOriginal() | SmartPointerMultiWrite() | RecursivePointer() | SmartPointerMultiWriteRary() | PlaintextPointer():
                self.p = head
            case _:
                match SmartPointer.label:
                    case "original":
                        self.p = SmartPointerOriginal(head)
                    case "multiwrite":
                        self.p = SmartPointerMultiWrite(head)
                    case "recursive":
                        self.p = RecursivePointer(head)
                    case "multiwriterary":
                        self.p = SmartPointerMultiWriteRary(head)
                    case "plaintext":
                        self.p = PlaintextPointer(head)
                    case _:
                        raise RuntimeError(f"Init: Label {SmartPointer.label} is not recognizable.")

    def __repr__(self) -> str:
        """String representation of a SmartPointer"""
        return str(self.p)

    def __eq__(self, other: object) -> bool:
        """Determine equality of two Smart Pointers"""
        if isinstance(other, SmartPointer):
            return id(self) == id(other)
            # return self.p.head == other.p.head
        return False

    def get_label(self) -> str:
        """Return label pointer belongs to"""
        return self.p.label

    def smart_delete(self) -> None:
        """Generic function to delete different Graph Objects"""
        SmartPointer.delete(self)

    @staticmethod
    def set_smart_pointer(label: str, clear: bool = True) -> None:
        """Change the pointer architecture to use under the hood"""
        match label:
            case "original":
                set_max_reads(1)
                set_max_writes(1)
            case "multiwrite" | "multiwriterary":
                set_max_reads(1)
                set_max_writes(maxsize)
            case "recursive":
                set_max_reads(maxsize)
                set_max_writes(maxsize)
            case "plaintext":
                pass
            case _:
                raise RuntimeError(f"Set Smart Pointer: Label {label} does not correspond to any pointer type.")

        SmartPointer.label = label
        if clear:
            clear_available_addresses()

    @staticmethod
    # type hint should be: sp: "SmartPointer" | None
    # but this throws a TypeError
    def is_temp_copy(sp: Any) -> bool:
        """Checks if a pointer is a temporary copy"""
        if isinstance(sp, SmartPointer) and hasattr(sp, "temp_copy"):
            return sp.temp_copy   
        return False

    @staticmethod
    def new(value: Any = None) -> int:
        """Create a reference to a new pointee"""
        match SmartPointer.label:
            case "original": 
                return SmartPointerOriginal.new(value)
            case "multiwrite":
                return SmartPointerMultiWrite.new(value)
            case "multiwriterary":
                return SmartPointerMultiWriteRary.new(value)
            case "recursive":
                return RecursivePointer.new(value)
            case "plaintext":
                return PlaintextPointer.new(value)
            case _:
                raise RuntimeError(f"New: Label {SmartPointer.label} is not recognizable.")

    @staticmethod
    def set_branching_factor(bf: int) -> None:
        """Set the branching factor for multiwriterary pointers"""
        if SmartPointer.label != "multiwriterary":
            raise RuntimeError(f"Set Branching Factor: Label {SmartPointer.label} is not multiwriterary.")
        SmartPointerMultiWriteRary.set_branching_factor(bf)

    @staticmethod
    def is_single_reference(sp: "SmartPointer") -> bool:
        """Checks if sp is the only reference to its pointee"""
        p = sp.p
        if p is None:
            return False

        match p:
            case SmartPointerOriginal():
                return SmartPointerOriginal.is_single_reference(p)
            case SmartPointerMultiWrite():
                return SmartPointerMultiWrite.is_single_reference(p)
            case SmartPointerMultiWriteRary():
                return SmartPointerMultiWriteRary.is_single_reference(p)
            case RecursivePointer():
                return RecursivePointer.is_single_reference(p)
            case PlaintextPointer():
                return PlaintextPointer.is_single_reference(p)
            case _:
                raise RuntimeError(f"Is Single Reference: Pointer type {type(p)} could not be matched.") 

    @staticmethod
    def copy(
            sp: Any,
            temp_copy: bool = False,
            num_copies: int = 1,
        ) -> Any:
        """Copy an object, with batched support for multi-write pointers."""
        if isinstance(sp, SmartPointer):
            p = sp.p
            match p:
                case SmartPointerOriginal():
                    if num_copies != 1:
                        raise NotImplementedError(
                            "num_copies is only supported by multi-write pointers"
                        )
                    new_copy = SmartPointerOriginal.copy(p, temp_copy)
                case SmartPointerMultiWrite():
                    new_copy = SmartPointerMultiWrite.copy(
                        p, temp_copy, num_copies
                    )
                case SmartPointerMultiWriteRary():
                    new_copy = SmartPointerMultiWriteRary.copy(
                        p, temp_copy, num_copies
                    )
                case RecursivePointer():
                    if num_copies != 1:
                        raise NotImplementedError(
                            "num_copies is only supported by multi-write pointers"
                        )
                    new_copy = RecursivePointer.copy(p, temp_copy)
                case PlaintextPointer():
                    if num_copies != 1:
                        raise NotImplementedError(
                            "num_copies is only supported by multi-write pointers"
                        )
                    new_copy = PlaintextPointer.copy(p, temp_copy)
                case _:
                    raise RuntimeError(f"Copy: Pointer type {type(p)} could not be matched.")

            if isinstance(new_copy, list) and isinstance(
                p, (SmartPointerMultiWrite, SmartPointerMultiWriteRary)
            ):
                return [SmartPointer(item, temp_copy) for item in new_copy]
            match new_copy:
                case SmartPointerOriginal() | SmartPointerMultiWrite() | SmartPointerMultiWriteRary() | RecursivePointer() | PlaintextPointer():
                    return SmartPointer(new_copy, temp_copy)
                case _:
                    return new_copy
        else:
            match SmartPointer.label:
                case "original":
                    if num_copies != 1:
                        raise NotImplementedError(
                            "num_copies is only supported by multi-write pointers"
                        )
                    return SmartPointerOriginal.copy(sp, temp_copy)
                case "multiwrite":
                    return SmartPointerMultiWrite.copy(
                        sp, temp_copy, num_copies
                    )
                case "multiwriterary":
                    return SmartPointerMultiWriteRary.copy(
                        sp, temp_copy, num_copies
                    )
                case "recursive":
                    if num_copies != 1:
                        raise NotImplementedError(
                            "num_copies is only supported by multi-write pointers"
                        )
                    return RecursivePointer.copy(sp, temp_copy)
                case "plaintext":
                    if num_copies != 1:
                        raise NotImplementedError(
                            "num_copies is only supported by multi-write pointers"
                        )
                    return PlaintextPointer.copy(sp, temp_copy)
                case _:
                    raise RuntimeError(f"Copy: Label {SmartPointer.label} does not correspond to any pointer type.")

    @staticmethod
    def smart_copy(
            sp: "SmartPointer",
            temp_copy: bool = False,
            num_copies: int = 1,
        ) -> Any:
        return SmartPointer.copy(sp, temp_copy, num_copies)

    @staticmethod
    def get(sp: "SmartPointer") -> Any:
        """Return p's pointee"""
        p = sp.p
        match p:
            case SmartPointerOriginal():
                return SmartPointerOriginal.get(p)
            case SmartPointerMultiWrite():
                return SmartPointerMultiWrite.get(p)
            case SmartPointerMultiWriteRary():
                return SmartPointerMultiWriteRary.get(p)
            case RecursivePointer():
                return RecursivePointer.get(p)
            case PlaintextPointer():
                return PlaintextPointer.get(p)
            case _:
                raise RuntimeError(f"Get: Pointer type {type(p)} could not be matched.") 

    @staticmethod 
    def get_and_copy(sp: "SmartPointer") -> Any:
        """Return a copy of p's pointee"""
        p = sp.p
        match p:
            case SmartPointerOriginal():
                return SmartPointerOriginal.get_and_copy(p)
            case SmartPointerMultiWrite():
                return SmartPointerMultiWrite.get_and_copy(p)
            case SmartPointerMultiWriteRary():
                return SmartPointerMultiWriteRary.get_and_copy(p)
            case RecursivePointer():
                return RecursivePointer.get_and_copy(p)
            case PlaintextPointer():
                return PlaintextPointer.get_and_copy(p)
            case _:
                raise RuntimeError(f"Get and Copy: Pointer type {type(p)} could not be matched.") 

    @staticmethod
    def get_attr(
            sp: "SmartPointer", 
            attributes: AttributeAtPositions | List[str] | str,
            make_copies: bool = True
        ) -> Any:
        """Retrieve specified attributes of a pointee object"""
        if isinstance(sp, SmartPointer):
            p = sp.p
        
            match p:
                case SmartPointerOriginal():
                    return SmartPointerOriginal.get_attr(p, attributes, make_copies)
                case SmartPointerMultiWrite():
                    return SmartPointerMultiWrite.get_attr(p, attributes, make_copies)
                case SmartPointerMultiWriteRary():
                    return SmartPointerMultiWriteRary.get_attr(p, attributes, make_copies)
                case RecursivePointer():
                    return RecursivePointer.get_attr(p, attributes, make_copies)
                case PlaintextPointer():
                    return PlaintextPointer.get_attr(p, attributes, make_copies)
                case _:
                    raise RuntimeError(f"Get Attr: Pointer type {type(p)} could not be matched.") 

    @staticmethod
    def put(sp: "SmartPointer", value: Any, delete_old: bool = True) -> None:
        """Put new pointee value at sp"""
        p = sp.p
        # ensure any other SmartPointer objects that are stored in 
        # sp's pointee data are not listed as temporary copies
        # and get deleted by accident
        if hasattr(value, "out_children"):
            i: Any
            for i in value.out_children:
                if hasattr(i, "temp_copy"):
                    i.temp_copy = False

        if hasattr(value, "children"):
            for i in value.children:
                if hasattr(i, "temp_copy"):
                    i.temp_copy = False

        if hasattr(value, "ogE") and hasattr(value.ogE, "temp_copy"):
            value.ogE.temp_copy = False

        match p:
            case SmartPointerOriginal():
                SmartPointerOriginal.put(p, value)
            case SmartPointerMultiWrite():
                SmartPointerMultiWrite.put(p, value, delete_old)
            case SmartPointerMultiWriteRary():
                SmartPointerMultiWriteRary.put(p, value, delete_old)
            case RecursivePointer():
                RecursivePointer.put(p, value, delete_old)
            case PlaintextPointer():
                PlaintextPointer.put(p, value, delete_old)
            case _:
                raise RuntimeError(f"Put: Pointer type {type(p)} could not be matched.") 

    @staticmethod
    def put_attr(sp: "SmartPointer", attributes: Dict[str, Any], delete_old: bool = False) -> None:
        """Update specified attributes at a pointee object"""
        p = sp.p

        # ensure any other SmartPointer objects that are stored in 
        # sp's pointee data are not listed as temporary copies
        # and get deleted by accident
        for value in attributes.values():
            if hasattr(value, "temp_copy"):
                value.temp_copy = False
            elif isinstance(value, list):
                for item in value:
                    if hasattr(item, "temp_copy"):
                        item.temp_copy = False
            elif isinstance(value, dict):
                for key in value.keys():
                    if hasattr(value[key], "temp_copy"):
                        value[key].temp_copy = False

        match p:
            case SmartPointerOriginal():
                SmartPointerOriginal.put_attr(p, attributes, delete_old)
            case SmartPointerMultiWrite():
                SmartPointerMultiWrite.put_attr(p, attributes, delete_old)
            case SmartPointerMultiWriteRary():
                SmartPointerMultiWriteRary.put_attr(p, attributes, delete_old)
            case RecursivePointer():
                RecursivePointer.put_attr(p, attributes, delete_old)
            case PlaintextPointer():
                PlaintextPointer.put_attr(p, attributes, delete_old)
            case _:
                raise RuntimeError(f"Put Attr: Pointer type {type(p)} could not be matched.")        

    def delete(sp: Any) -> None:
        """Handles deletes for SmartPointer and other objects"""
        if isinstance(sp, SmartPointer):
            p = sp.p

            match p:
                case SmartPointerOriginal():
                    SmartPointerOriginal.delete(p)
                case SmartPointerMultiWrite():
                    SmartPointerMultiWrite.delete(p)
                case SmartPointerMultiWriteRary():
                    SmartPointerMultiWriteRary.delete(p)
                case RecursivePointer():
                    RecursivePointer.delete(p)
                case PlaintextPointer():
                    PlaintextPointer.delete(p)
                case None:
                    return
                case _:
                    raise RuntimeError(f"Delete: Pointer type {type(p)} could not be matched.") 

        elif type(sp) in [list, tuple, set, frozenset]:
            for elem in sp:
                SmartPointer.delete(elem)
        elif isinstance(sp, dict):
            for key, value in sp.items():
                SmartPointer.delete(key)
                SmartPointer.delete(value)
        elif hasattr(sp, "smart_delete") and callable(sp.smart_delete):
            sp.smart_delete() 

    @staticmethod
    # type hint should be: sp: "SmartPointer" | None
    # but this throws a TypeError
    def delete_temp_copy(sp: Any) -> None:
        """Delete temporary copy"""
        if isinstance(sp, SmartPointer) and SmartPointer.is_temp_copy(sp):
            SmartPointer.delete(sp)
