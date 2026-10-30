from single_access_machine import (
    alloc,
    get_global_counter,
    read,
    reset,
    return_available_address,
    sam,
    set_max_reads,
    set_max_writes,
    write,
)
import copy
import random
from sys import maxsize
from attribute_at_positions import AttributeAtPositions
from typing import List, Dict, Tuple, Set, FrozenSet, Type, Any

"""
SmartPointerMultiWrite - Pointer class built on single read / multi write SAM
"""

class SmartPointerMultiWrite:
    # pointer label  
    label = "multiwrite"

    # uses SAM addresses when True
    pointer_osam = True

    # label for tracking which object incurs SAM accesses
    structure = "SmartPointerMultiWrite"

    def __init__(self, head: int | Type["SmartPointerMultiWrite"] | None) -> None:
        """Initialize SmartPointerMultiWrite from different origins"""
        if isinstance(head, int) or head is None:
            self.head = head
        elif isinstance(head, SmartPointerMultiWrite):
            new_left, new_right = SmartPointerMultiWrite.copy_raw_address(head.head)
            self.head = new_left
            head.head = new_right
        else:
            raise ValueError(f"Invalid type for SmartPointerMultiWrite initialization: {type(head)}")

    def __repr__(self) -> str:
        """String representation of SmartPointerMultiWrite"""
        return str(self.head)

    @staticmethod
    def link(left: int, right: int, parent: int) -> None:
        """Arranges siblings left and right so that they both point to parent"""
        write(left, ("INNER", parent, right), SmartPointerMultiWrite.structure)
        write(right, ("INNER", parent, left), SmartPointerMultiWrite.structure)

    @staticmethod
    def deref(head: int) -> Tuple[int, int, Tuple[str, int | None, Any]]:
        """Dereference a pointer to get the pointee and a writeback address"""
        t, p_val, s = read(head, SmartPointerMultiWrite.structure)
        return_available_address(head)
        p_new = alloc(SmartPointerMultiWrite.pointer_osam, SmartPointerMultiWrite.structure)
        if t == "ROOT":
            return (p_new, p_new, (t, p_val, s))
        elif t == "INNER":
            new_root, tree_node = SmartPointerMultiWrite.splay(p_new, s, p_val)
            return (p_new, new_root, tree_node)
        else:
            raise ValueError("Invalid address dereference")
        
    @staticmethod
    def splay(a: int, b: int, x: int) -> Tuple[int, Tuple[str, None, Any]]:
        """Balances traversal path when a pointer is dereferenced"""
        while True:
            tx, y, c = read(x, SmartPointerMultiWrite.structure)
            if tx == "ROOT":
                x_ = alloc(SmartPointerMultiWrite.pointer_osam, SmartPointerMultiWrite.structure)
                SmartPointerMultiWrite.link(a, b, x_)
                return (x_, (tx, y, c))
            elif tx == "INNER":
                ty, z, d = read(y, SmartPointerMultiWrite.structure)
                if ty == "ROOT":
                    x_ = alloc(SmartPointerMultiWrite.pointer_osam, SmartPointerMultiWrite.structure)
                    y_ = alloc(SmartPointerMultiWrite.pointer_osam, SmartPointerMultiWrite.structure)
                    SmartPointerMultiWrite.link(b, c, y_)
                    SmartPointerMultiWrite.link(a, y_, x_)
                    return (x_, (ty, z, d))
                elif ty == "INNER":
                    x_ = z
                    y_ = alloc(SmartPointerMultiWrite.pointer_osam, SmartPointerMultiWrite.structure)
                    z_ = alloc(SmartPointerMultiWrite.pointer_osam, SmartPointerMultiWrite.structure)
                    SmartPointerMultiWrite.link(a, b, y_)
                    SmartPointerMultiWrite.link(c, d, z_)
                    a = y_
                    b = z_
                    x = x_
                else:
                    raise ValueError("Invalid address dereference")
            else:
                raise ValueError("Invalid address dereference")

    @staticmethod
    def new(value: Any = None) -> int:
        """Create a reference to a new pointee"""
        root = alloc(SmartPointerMultiWrite.pointer_osam, SmartPointerMultiWrite.structure)
        write(root, ("ROOT", None, value), SmartPointerMultiWrite.structure)
        return root

    @staticmethod
    def is_single_reference(p: "SmartPointerMultiWrite") -> bool:
        """Checks if p is the only reference to its pointee"""
        assert isinstance(p.head, int)
        new_head, root, tree_node = SmartPointerMultiWrite.deref(p.head)
        write(root, tree_node, SmartPointerMultiWrite.structure)
        p.head = new_head
        return new_head == root

    @staticmethod
    def __smart_copy_pointer(
            p0: "SmartPointerMultiWrite",
            num_copies: int = 1,
        ) -> "SmartPointerMultiWrite" | List["SmartPointerMultiWrite"]:
        """Create aliases below p0's head, balanced when making many copies."""
        if isinstance(num_copies, bool) or not isinstance(num_copies, int):
            raise TypeError("num_copies must be an integer")
        if num_copies < 1:
            raise ValueError(f"num_copies must be >= 1, got {num_copies}")

        if p0.head is None:
            copies = [SmartPointerMultiWrite(None) for _ in range(num_copies)]
            return copies[0] if num_copies == 1 else copies

        if num_copies == 1:
            new_copy, new_orig = SmartPointerMultiWrite.copy_raw_address(p0.head)
            p0.head = new_orig
            return SmartPointerMultiWrite(new_copy)

        heads: List[int] = []
        SmartPointerMultiWrite._build_balanced_subtree(
            p0.head, num_copies + 1, heads
        )
        p0.head = heads[0]
        return [SmartPointerMultiWrite(head) for head in heads[1:]]

    @staticmethod
    def _build_balanced_subtree(
            parent: int,
            num_leaves: int,
            leaves: List[int],
        ) -> None:
        """Install a balanced binary subtree with `num_leaves` below parent."""
        if num_leaves < 2:
            raise ValueError(f"num_leaves must be >= 2, got {num_leaves}")

        left_count = num_leaves // 2
        right_count = num_leaves - left_count
        left = alloc(
            SmartPointerMultiWrite.pointer_osam,
            SmartPointerMultiWrite.structure,
        )
        right = alloc(
            SmartPointerMultiWrite.pointer_osam,
            SmartPointerMultiWrite.structure,
        )
        SmartPointerMultiWrite.link(left, right, parent)

        if left_count == 1:
            leaves.append(left)
        else:
            SmartPointerMultiWrite._build_balanced_subtree(
                left, left_count, leaves
            )

        if right_count == 1:
            leaves.append(right)
        else:
            SmartPointerMultiWrite._build_balanced_subtree(
                right, right_count, leaves
            )

    @staticmethod
    def copy_raw_address(new_parent: int) -> Tuple[int, int]:
        """Output two new children that point to new_parent"""
        new_left = alloc(SmartPointerMultiWrite.pointer_osam, SmartPointerMultiWrite.structure)
        new_right = alloc(SmartPointerMultiWrite.pointer_osam, SmartPointerMultiWrite.structure)
        SmartPointerMultiWrite.link(new_left, new_right, new_parent)
        return new_left, new_right
    
    # @staticmethod
    # def copy(p: Any, temp_copy: bool = False) -> Any:
    #     """Handles copies of any object"""
    #     if isinstance(p, SmartPointerMultiWrite):
    #         return SmartPointerMultiWrite.__smart_copy_pointer(p)
    #     elif hasattr(p, "smart_copy") and callable(p.smart_copy):
    #         return p.smart_copy(p, temp_copy)
    #     else:
    #         return copy.deepcopy(p)

    @staticmethod
    def copy(
            content: Any,
            temp_copy: bool = False,
            num_copies: int = 1,
        ) -> Any:
        """Copy an object, balancing batched SmartPointerMultiWrite aliases."""
        if isinstance(num_copies, bool) or not isinstance(num_copies, int):
            raise TypeError("num_copies must be an integer")
        if num_copies < 1:
            raise ValueError(f"num_copies must be >= 1, got {num_copies}")

        if isinstance(content, SmartPointerMultiWrite):
            return SmartPointerMultiWrite.__smart_copy_pointer(
                content, num_copies
            )
        if hasattr(content, "smart_copy") and callable(content.smart_copy):
            if num_copies > 1:
                return content.smart_copy(content, temp_copy, num_copies)
            return content.smart_copy(content, temp_copy)
        if num_copies > 1:
            return [
                SmartPointerMultiWrite.copy(content, temp_copy)
                for _ in range(num_copies)
            ]

        def copy_list(struct1: List[Any] | Tuple[Any] | Set[Any] | FrozenSet[Any]) -> List[Any]:
            """Helper function that smart copies a list"""
            struct2 = []
            for elem in struct1:
                struct2.append(SmartPointerMultiWrite.copy(elem, temp_copy))
            return struct2

        def copy_dict(struct1: Dict[Any, Any]) -> Dict[Any, Any]:
            """Helper function that smart copies a dict"""
            struct2: Dict[Any, Any] = dict()
            for key, value in struct1.items():
                struct2[key] = SmartPointerMultiWrite.copy(value, temp_copy)
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
        elif content is not None:
            return copy.deepcopy(content)

    @staticmethod
    def get(p: "SmartPointerMultiWrite") -> Any:
        """Return p's pointee"""
        assert isinstance(p.head, int)
        new_address, root, tree_node = SmartPointerMultiWrite.deref(p.head)
        assert (tree_node[1] is None)
        p.head = new_address
        write(root, ("ROOT", None, tree_node[2]), SmartPointerMultiWrite.structure)
        return tree_node[2]
    
    @staticmethod 
    def get_and_copy(p: "SmartPointerMultiWrite") -> Any:
        """Return a copy of p's pointee"""
        value = SmartPointerMultiWrite.get(p)
        return SmartPointerMultiWrite.copy(value) 

    @staticmethod
    def get_attr(   
            p: "SmartPointerMultiWrite", 
            attributes: AttributeAtPositions | List[str] | str,
            make_copies: bool = True
        ) -> Any:
        """Retrieve specified attributes of a pointee object"""
        value = SmartPointerMultiWrite.get(p)
        content: Dict[str, Any] | str | None

        # get an attribute that is a list at specific positions only
        if isinstance(attributes, AttributeAtPositions):
            content = dict()
            try: 
                list_attr = getattr(value, attributes.attribute)
                sublist = []
                for pos in attributes.positions:
                    elem = list_attr[pos]
                    if make_copies: elem = SmartPointerMultiWrite.copy(elem, temp_copy=make_copies)
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
                    if make_copies: elem = SmartPointerMultiWrite.copy(elem, temp_copy=make_copies)
                    content[attr] = elem
                except:
                    content[attr] = None

        # get one attribute
        elif type(attributes) == str:
            try:
                content = getattr(value, attributes)
                if make_copies: content = SmartPointerMultiWrite.copy(content, temp_copy=make_copies)
            except:
                content = None
        
        else:
            raise TypeError("Attributes must be a list, a string, or a AttributeAtPositions instance.")

        return content

    @staticmethod
    def put(p: "SmartPointerMultiWrite", value: Any, delete_old: bool = True) -> None:
        """Put new pointee value at p"""
        assert isinstance(p.head, int)
        new_head, root, tree_node = SmartPointerMultiWrite.deref(p.head)
        assert tree_node[1] is None
        p.head = new_head
        if delete_old:
            SmartPointerMultiWrite.delete(tree_node[2])
        # save a copy with put
        write(root, ("ROOT", None, SmartPointerMultiWrite.copy(value)), SmartPointerMultiWrite.structure)

    @staticmethod
    def put_attr(p: "SmartPointerMultiWrite", attributes: Dict[str, Any], delete_old: bool = False) -> None:
        """Update specified attributes at a pointee object"""
        assert isinstance(p.head, int)
        new_head, root, tree_node = SmartPointerMultiWrite.deref(p.head)
        assert tree_node[1] is None 
        for attr in attributes.keys():
            # delete original object
            if delete_old and hasattr(tree_node[2], attr):
                SmartPointerMultiWrite.delete(getattr(tree_node[2], attr))
            setattr(tree_node[2], attr, attributes[attr])
        p.head = new_head
        write(root, ("ROOT", None, tree_node[2]), SmartPointerMultiWrite.structure)

    @staticmethod
    def delete(p: Any) -> None:
        """Handles deletes for SmartPointerMultiWrite and other objects"""
        if isinstance(p, SmartPointerMultiWrite):
            if p.head is None:
                return
            t1, p1, s1 = read(p.head, SmartPointerMultiWrite.structure)
            if t1 == "INNER":
                t2, p2, s2 = read(p1, SmartPointerMultiWrite.structure)  
                if t2 == "ROOT":
                    write(s1, [t2, p2, s2], SmartPointerMultiWrite.structure)
                else:
                    SmartPointerMultiWrite.link(s1, s2, p2)
            p.head = None
        elif type(p) in [list, tuple, set, frozenset]:
            for elem in p:
                SmartPointerMultiWrite.delete(elem)
        elif isinstance(p, dict):
            for key, value in p.items():
                SmartPointerMultiWrite.delete(key)
                SmartPointerMultiWrite.delete(value)
        elif hasattr(p, "smart_delete"):
            p.smart_delete()





def test_bulk_copy_builds_balanced_tree() -> None:
    reset()
    set_max_reads(1)
    set_max_writes(maxsize)

    original = SmartPointerMultiWrite(SmartPointerMultiWrite.new("balanced"))
    copies = SmartPointerMultiWrite.copy(original, num_copies=100)

    assert isinstance(copies, list)
    assert len(copies) == 100

    def initial_depth(head: int) -> int:
        depth = 0
        node = sam.storage[head]
        while node[0] == "INNER":
            depth += 1
            node = sam.storage[node[1]]
        return depth

    depths = [initial_depth(pointer.head) for pointer in [original, *copies]]
    assert min(depths) == 6
    assert max(depths) == 7
    assert all(
        SmartPointerMultiWrite.get(pointer) == "balanced"
        for pointer in [original, *copies]
    )
    print("test_bulk_copy_builds_balanced_tree: OK")


def smart_pointer_test_1() -> None:
    num_ops = get_global_counter()
    p1 = SmartPointerMultiWrite(SmartPointerMultiWrite.new("cat"))
    assert(SmartPointerMultiWrite.is_single_reference(p1) == True)
    p2 = SmartPointerMultiWrite.copy(p1)
    print("First copy,", get_global_counter())
    num_ops = get_global_counter()
    # print("Post copy values ",p1.head, p2.head, p3.head)
    assert(SmartPointerMultiWrite.get(p1) == "cat")
    assert(SmartPointerMultiWrite.is_single_reference(p1) == False)
    print("First get,", get_global_counter())
    #Check if you can get the same value twice
    assert(SmartPointerMultiWrite.get(p1) == "cat")
    print("Second get,", get_global_counter())
    num_ops = get_global_counter()
    p3 = SmartPointerMultiWrite.copy(p1)
    assert(SmartPointerMultiWrite.is_single_reference(p1) == False)
    print("Second copy,", get_global_counter())
    num_ops = get_global_counter()
    assert(SmartPointerMultiWrite.get(p1) == "cat")
    print("Third get,", get_global_counter())
    num_ops = get_global_counter()
    SmartPointerMultiWrite.put(p1, "dog")
    print("Put,", get_global_counter())
    num_ops = get_global_counter()
    for i in range(10):
        assert(SmartPointerMultiWrite.get(p1) == "dog")
        print("Get,", get_global_counter())
        num_ops = get_global_counter()
        assert(SmartPointerMultiWrite.get(p2) == "dog")
        print("Get,", get_global_counter())
        num_ops = get_global_counter()
    
    p4 = SmartPointerMultiWrite(SmartPointerMultiWrite.new(p1.get(p2)))
    assert(SmartPointerMultiWrite.is_single_reference(p4) == True)
    assert(SmartPointerMultiWrite.get(p4) == "dog")
    print("Get three copies,", get_global_counter())
    SmartPointerMultiWrite.put(p1, "horse")
    assert(SmartPointerMultiWrite.get(p2) == "horse")
    print("Get three copies,", get_global_counter())
    assert(SmartPointerMultiWrite.get(p4) == "dog")
    print("Get p4 unrelated,", get_global_counter())
    SmartPointerMultiWrite.delete(p3)
    print("Get post delete p3,", get_global_counter())
    assert(SmartPointerMultiWrite.get(p2) == "horse")
    print("Get two copies,", get_global_counter())
    SmartPointerMultiWrite.delete(p2)
    print("Get post delete p2,", get_global_counter())
    assert(SmartPointerMultiWrite.get(p1) == "horse")
    print("Get post delete p2,", get_global_counter())
    assert(SmartPointerMultiWrite.is_single_reference(p1) == True)
    assert(SmartPointerMultiWrite.is_single_reference(p4) == True)
    SmartPointerMultiWrite.delete(p1)
    SmartPointerMultiWrite.delete(p4)
    print("Test 1 completed successfully.")
    
def smart_pointer_test_2() -> None:
    for lg_num_pointers in range(1, 5):
        num_pointers = 2**lg_num_pointers
        print("testing with ",num_pointers)

        pointers: List[None | int] = [None] * num_pointers
        pointers[0] = SmartPointerMultiWrite.new("cat")
        
        for i in range(1, num_pointers):
            a = pointers[i - 1]
            assert isinstance(a, int)
            p, q = SmartPointerMultiWrite.copy_raw_address(a)
            pointers[i] = p
            pointers[i - 1] = q

        # random.shuffle(pointers)

        for i in range(num_pointers):
            a = pointers[i]
            assert isinstance(a, int)
            new_pointer, root, value = SmartPointerMultiWrite.deref(a)
            pointers[i] = new_pointer
            write(root, ["ROOT", None, value], SmartPointerMultiWrite.structure)

        for i in range(num_pointers):
            a = pointers[i]
            assert isinstance(a, int)
            new_pointer, root, value = SmartPointerMultiWrite.deref(a)
            pointers[i] = new_pointer
            write(root, ["ROOT", None, value], SmartPointerMultiWrite.structure)

        print("After ",num_pointers, " the global counter is ",get_global_counter())

def smart_pointer_test_3() -> None:
    animal_list = ["cow", "cat", "dog", "horse", "sheep", "pig", "goat", "chicken"]
    smart_pointer_list = []
    for animal in animal_list:
        sp = SmartPointerMultiWrite(SmartPointerMultiWrite.new(animal))
        smart_pointer_list.append(sp)
    print("Loaded smart pointers with animals:",get_global_counter())

    pointer_to_pointer = SmartPointerMultiWrite(SmartPointerMultiWrite.new(smart_pointer_list))
    print("Loaded pointer to pointers:",get_global_counter())
    list_of_ptrs= SmartPointerMultiWrite.get(pointer_to_pointer)
    i = 0
    for ptr in list_of_ptrs:
        assert(SmartPointerMultiWrite.get(ptr)==animal_list[i])
        i += 1
    list_of_ptrs = SmartPointerMultiWrite.get_and_copy(pointer_to_pointer)
    i = 0
    for ptr in list_of_ptrs:
        assert(SmartPointerMultiWrite.get(ptr)==animal_list[i])
        i += 1

    print("Test 3 completed successfully with animals:",get_global_counter())

def smart_pointer_test_4() -> None:
    p_original = (SmartPointerMultiWrite(SmartPointerMultiWrite.new("cat")))
    p_copies = []
    num_copies = 100
    for i in range(num_copies):
        p_copies.append(SmartPointerMultiWrite.copy(p_original))

    total_alloc = 0
    for i in range(num_copies):
        pre_alloc = get_global_counter()[0]
        assert(SmartPointerMultiWrite.get(p_copies[i])=="cat")
        total_alloc += get_global_counter()[0] - pre_alloc

    print("Average allocs per indexed get with {} copies: {}".format(num_copies, total_alloc/num_copies))

    total_alloc = 0
    for i in range(num_copies):
        pre_alloc = get_global_counter()[0]
        assert(SmartPointerMultiWrite.get(p_copies[i])=="cat")
        total_alloc += get_global_counter()[0] - pre_alloc

    print("Average allocs per indexed get with {} copies: {}".format(num_copies, total_alloc/num_copies))


    total_alloc = 0
    for i in range(num_copies):
        pre_alloc = get_global_counter()[0]
        assert(SmartPointerMultiWrite.get(p_copies[i])=="cat")
        total_alloc += get_global_counter()[0] - pre_alloc

    print("Average allocs per indexed get with {} copies: {}".format(num_copies, total_alloc/num_copies))

    total_alloc = 0
    for i in range(num_copies):
        j = random.randint(0, num_copies -1)
        pre_alloc = get_global_counter()[0]
        assert(SmartPointerMultiWrite.get(p_copies[j])=="cat")
        total_alloc += get_global_counter()[0] - pre_alloc

    print("Average allocs per random get with {} copies: {}".format(num_copies, total_alloc/num_copies))

    total_alloc = 0
    for i in range(num_copies):
        j = random.randint(0, num_copies -1)
        pre_alloc = get_global_counter()[0]
        assert(SmartPointerMultiWrite.get(p_copies[j])=="cat")
        total_alloc += get_global_counter()[0] - pre_alloc

    print("Average allocs per random get with {} copies: {}".format(num_copies, total_alloc/num_copies))

    total_alloc = 0
    for i in range(num_copies):
        j = random.randint(0, num_copies -1)
        pre_alloc = get_global_counter()[0]
        assert(SmartPointerMultiWrite.get(p_copies[j])=="cat")
        total_alloc += get_global_counter()[0] - pre_alloc

    print("Average allocs per random get with {} copies: {}".format(num_copies, total_alloc/num_copies))


    total_alloc = 0
    for i in range(num_copies):
        pre_alloc = get_global_counter()[0]
        assert(SmartPointerMultiWrite.get(p_copies[i])=="cat")
        total_alloc += get_global_counter()[0] - pre_alloc

    print("Average allocs per indexed get with {} copies: {}".format(num_copies, total_alloc/num_copies))

    total_alloc = 0
    for i in range(num_copies):
        pre_alloc = get_global_counter()[0]
        assert(SmartPointerMultiWrite.get(p_copies[i])=="cat")
        total_alloc += get_global_counter()[0] - pre_alloc

    print("Average allocs per indexed get with {} copies: {}".format(num_copies, total_alloc/num_copies))

    total_alloc = 0
    for i in range(num_copies):
        j = random.randint(0, num_copies -1)
        pre_alloc = get_global_counter()[0]
        assert(SmartPointerMultiWrite.get(p_copies[j])=="cat")
        total_alloc += get_global_counter()[0] - pre_alloc

    print("Average allocs per random get with {} copies: {}".format(num_copies, total_alloc/num_copies))

if __name__ == "__main__":
    print("Allocs, Reads, Writes")
    set_max_reads(1)
    set_max_writes(maxsize)
    
    test_bulk_copy_builds_balanced_tree()
    smart_pointer_test_1()
    smart_pointer_test_2()
    smart_pointer_test_3()
    smart_pointer_test_4()
