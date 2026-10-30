from single_access_machine import alloc, read, write, get_global_counter, set_max_reads, set_max_writes
from address_queue import AddressQueue as queue
import copy
from attribute_at_positions import AttributeAtPositions
from typing import List, Dict, Tuple, Set, FrozenSet, Type, Any

"""
SmartPointerOriginal - Pointer class built on single read / single write SAM
"""

class SmartNode:
    """Storage for pointers built on queues""" 
    def __init__(   
            self, 
            left_tail: int | None = None, 
            right_tail: int | None = None, 
            node_head: int | None = None, 
            is_root: bool = False, 
            content: Any = None
        ) -> None:
        self.left_tail = left_tail
        self.right_tail = right_tail
        self.is_root = is_root
        self.content = content
        self.node_head = node_head
        self.count = 1

    def __str__(self) -> str:
        """String representation of SmartNode"""
        return f"Node Contents: {self.right_tail}), {self.left_tail}, {self.is_root}, {self.content}, {self.node_head}"
        
class SmartPointerOriginal:
    # pointer label  
    label = "original"

    # uses SAM addresses when True
    pointer_osam = True

    # label for tracking which object incurs SAM accesses
    structure = "SmartPointerOriginal"

    def __init__(self, head: int | None) -> None:
        """Initialize a new pointer"""
        self.head = head
        
    def __repr__(self) -> str:
        """String representation of SmartPointerOriginal"""
        return str(self.head)

    @staticmethod
    def __chase(head: int | None) -> SmartNode | None:
        """Retrieves object at the end of a queue"""
        target, latest, tail = None, None, None
        while isinstance(head, int):
            latest = target
            tail = head
            target, head = queue.dequeue(head)
            
        node = None
        if isinstance(latest, int):
            node = read(latest, SmartPointerOriginal.structure)
            if node.left_tail == tail:
                node.left_tail = None
            else:
                node.right_tail = None
        return node

    @staticmethod
    def __save_node(node: SmartNode | None) -> None:
        """
        Checks if the appropriate queues have been created
        If not, initialize the two queues
        """
        address = alloc(SmartPointerOriginal.pointer_osam, SmartPointerOriginal.structure)
        if node is not None:
            if node.left_tail is not None:
                node.left_tail = queue.enqueue(node.left_tail, address)
            if node.right_tail is not None:
                node.right_tail = queue.enqueue(node.right_tail, address)
        write(address, node, SmartPointerOriginal.structure)

    @staticmethod
    def __add_tail(node: SmartNode | None) -> int | None:
        """Begins new queue at SmartNode""" 
        if node is None:
            return None
        head, tail = queue.init_queue()
        if isinstance(node.left_tail, int) and isinstance(node.right_tail, int):
            raise ValueError("Both tails are already occupied")
        elif node.left_tail is None:
            node.left_tail = tail
        else:
            node.right_tail = tail
        return head

    @staticmethod
    def new(content: Any) -> int:
        """Create a reference to a new pointee"""
        node = SmartNode(left_tail=None, right_tail=None, content=content, is_root=True)
        head = SmartPointerOriginal.__add_tail(node)
        SmartPointerOriginal.__save_node(node)
        assert isinstance(head, int)
        return head

    @staticmethod
    def is_single_reference(p: "SmartPointerOriginal") -> bool:
        """Checks if p is the only reference to its pointee"""
        node = SmartPointerOriginal.__chase(p.head)
        p.head = SmartPointerOriginal.__add_tail(node)
        another_pointer = False
        if isinstance(node, SmartNode) and isinstance(node.left_tail, int) and isinstance(node.right_tail, int):
            another_pointer = True
        while isinstance(node, SmartNode) and (not node.is_root):
            new_node = SmartPointerOriginal.__chase(node.node_head)
            assert isinstance(new_node, SmartNode)
            node.node_head = SmartPointerOriginal.__add_tail(new_node)
            SmartPointerOriginal.__save_node(node)
            node = new_node
            if isinstance(node.left_tail, int) and isinstance(node.right_tail, int):
                another_pointer = True
        SmartPointerOriginal.__save_node(node)
        return not another_pointer

    @staticmethod
    def __smart_copy_pointer(p0: "SmartPointerOriginal") -> "SmartPointerOriginal":
        """Create a Smart Copy p1 that points to p0's pointee"""
        p1 = SmartPointerOriginal(None)
        node = SmartPointerOriginal.__chase(p0.head)
        if isinstance(node, SmartNode):
            node.count += 1
            if isinstance(node.left_tail, int) or isinstance(node.right_tail, int):
                new_node = SmartNode(node_head=SmartPointerOriginal.__add_tail(node), is_root=False, content="Branch Node")
                SmartPointerOriginal.__save_node(node)
                node = new_node

        p1.head = SmartPointerOriginal.__add_tail(node)
        p0.head = SmartPointerOriginal.__add_tail(node)
        SmartPointerOriginal.__save_node(node)
        return p1
        
    # @staticmethod
    # def copy(p: Any, temp_copy: bool = False) -> Any:
    #     """Handles copies of any object"""
    #     if isinstance(p, SmartPointerOriginal):
    #         return SmartPointerOriginal.__smart_copy_pointer(p)
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
                struct2.append(SmartPointerOriginal.copy(elem, temp_copy))
            return struct2

        def copy_dict(struct1: Dict[Any, Any]) -> Dict[Any, Any]:
            """Helper function that smart copies a dict"""
            struct2: Dict[Any, Any] = dict()
            for key, value in struct1.items():
                struct2[key] = SmartPointerOriginal.copy(value, temp_copy)
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
        elif isinstance(content, SmartPointerOriginal):
            return SmartPointerOriginal.__smart_copy_pointer(content)
        elif hasattr(content, "smart_copy") and callable(content.smart_copy):
            return content.smart_copy(content, temp_copy)
        elif content is not None:
            return copy.deepcopy(content)

    @staticmethod
    def get(p: "SmartPointerOriginal") -> Any:
        """Return p's pointee"""
        node = SmartPointerOriginal.__chase(p.head)
        p.head = SmartPointerOriginal.__add_tail(node)
        while isinstance(node, SmartNode) and (not node.is_root):
            new_node = SmartPointerOriginal.__chase(node.node_head)
            node.node_head = SmartPointerOriginal.__add_tail(new_node)
            SmartPointerOriginal.__save_node(node)
            node = new_node
        if node is not None:
            content = node.content
        else:
            content = None
        SmartPointerOriginal.__save_node(node)
        return content

    @staticmethod
    def get_and_copy(p: "SmartPointerOriginal") -> Any:
        """Return a copy of p's pointee"""
        node = SmartPointerOriginal.__chase(p.head)
        p.head = SmartPointerOriginal.__add_tail(node)
        while isinstance(node, SmartNode) and (not node.is_root):
            new_node = SmartPointerOriginal.__chase(node.node_head)
            node.node_head = SmartPointerOriginal.__add_tail(new_node)
            SmartPointerOriginal.__save_node(node)
            node = new_node
        assert isinstance(node, SmartNode)
        copy_of_data = SmartPointerOriginal.copy(node.content, temp_copy=True)
        SmartPointerOriginal.__save_node(node)
        return copy_of_data

    @staticmethod
    def get_attr(   
            p: "SmartPointerOriginal", 
            attributes: AttributeAtPositions | List[str] | str,
            make_copies: bool = True
        ) -> Any:
        """Retrieve specified attributes of a pointee object"""
        value = SmartPointerOriginal.get(p)
        content: Dict[str, Any] | str | None

        # get an attribute that is a list at specific positions only
        if isinstance(attributes, AttributeAtPositions):
            content = dict()
            try: 
                list_attr = getattr(value, attributes.attribute)
                sublist = []
                for pos in attributes.positions:
                    elem = list_attr[pos]
                    if make_copies: elem = SmartPointerOriginal.copy(elem, temp_copy=make_copies)
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
                    if make_copies: elem = SmartPointerOriginal.copy(elem, temp_copy=make_copies)
                    content[attr] = elem
                except:
                    content[attr] = None

        # get one attribute
        elif type(attributes) == str:
            try:
                content = getattr(value, attributes)
                if make_copies: content = SmartPointerOriginal.copy(content, temp_copy=make_copies)
            except:
                content = None
        
        else:
            raise TypeError("Attributes must be a list, a string, or a AttributeAtPositions instance.")

        return content

    @staticmethod
    def put(p: "SmartPointerOriginal", content: Any) -> None:
        """Put new pointee value at p and delete old pointee"""
        # delete content all the way at end of pointer
        SmartPointerOriginal.delete(SmartPointerOriginal.get(p))
        node = SmartPointerOriginal.__chase(p.head)
        assert isinstance(node, SmartNode)
        SmartPointerOriginal.delete(node.content)
        p.head = SmartPointerOriginal.__add_tail(node)
        while isinstance(node, SmartNode) and (not node.is_root):
            new_node = SmartPointerOriginal.__chase(node.node_head)
            node.node_head = SmartPointerOriginal.__add_tail(new_node)
            SmartPointerOriginal.__save_node(node)
            node = new_node
        assert isinstance(node, SmartNode)
        node.content = SmartPointerOriginal.copy(content)
        SmartPointerOriginal.__save_node(node)

    @staticmethod
    def put_attr(p: "SmartPointerOriginal", attributes: Dict[str, Any], delete_old: bool = False) -> None:
        """Update specified attributes at a pointee object"""
        node = SmartPointerOriginal.__chase(p.head)
        p.head = SmartPointerOriginal.__add_tail(node)
        while isinstance(node, SmartNode) and (not node.is_root):
            new_node = SmartPointerOriginal.__chase(node.node_head)
            node.node_head = SmartPointerOriginal.__add_tail(new_node)
            SmartPointerOriginal.__save_node(node)
            node = new_node
        assert isinstance(node, SmartNode)
        for attr in attributes:
            if delete_old and hasattr(node.content, attr):
                SmartPointerOriginal.delete(getattr(node.content, attr))
            setattr(node.content, attr, attributes[attr])
        SmartPointerOriginal.__save_node(node)    

    @staticmethod
    def delete(p: Any) -> None:
        """Handles deletes for SmartPointerOriginal and other objects"""
        if isinstance(p, SmartPointerOriginal):
            node = SmartPointerOriginal.__chase(p.head)
            if node is None:
                return 
            if node.is_root:
                if isinstance(node.left_tail, int) and isinstance(node.right_tail, int):
                    SmartPointerOriginal.delete(node.content)
                else:
                    node.count -= 1
                    SmartPointerOriginal.__save_node(node)
            else:
                if isinstance(node.left_tail, int):
                    tail = node.left_tail
                elif isinstance(node.right_tail, int):
                    tail = node.right_tail
                node = SmartPointerOriginal.__chase(node.node_head)
                assert isinstance(node, SmartNode)
                node.count -= 1
                if node.left_tail is None:
                    node.left_tail = tail
                else:
                    node.right_tail = tail
                SmartPointerOriginal.__save_node(node)
            p.head = None
        elif type(p) in [list, tuple, set, frozenset]:
            for elem in p:
                SmartPointerOriginal.delete(elem)
        elif isinstance(p, dict):
            for key, value in p.items():
                SmartPointerOriginal.delete(key)
                SmartPointerOriginal.delete(value)
        elif hasattr(p, "smart_delete"):
            p.smart_delete()

    @staticmethod
    def get_count(p: "SmartPointerOriginal") -> int:
        """Return number of references to p's pointee"""
        node = SmartPointerOriginal.__chase(p.head)
        p.head = SmartPointerOriginal.__add_tail(node)
        while isinstance(node, SmartNode) and (not node.is_root):
            new_node = SmartPointerOriginal.__chase(node.node_head)
            node.node_head = SmartPointerOriginal.__add_tail(new_node)
            SmartPointerOriginal.__save_node(node)
            node = new_node
        assert isinstance(node, SmartNode)
        count = node.count
        SmartPointerOriginal.__save_node(node)
        return count
        



def smart_pointer_test_1() -> None:
    print("Initializing pointers:")
    p1 = SmartPointerOriginal(SmartPointerOriginal.new("cat"))
    print("Values initial no cp,",p1.head)
    assert(SmartPointerOriginal.get_count(p1)==1)
    assert(SmartPointerOriginal.is_single_reference(p1)==True)
    num_ops = get_global_counter()[0]
    p2 = SmartPointerOriginal.copy(p1)
    print("Number allocs first copy,", get_global_counter()[0]-num_ops)
    assert(SmartPointerOriginal.get_count(p2)==2)
    assert(SmartPointerOriginal.is_single_reference(p1)==False)
    assert(SmartPointerOriginal.is_single_reference(p2)==False)
    num_ops = get_global_counter()[0]
    # print("Post copy values ",p1.head, p2.head, p3.head)
    assert(SmartPointerOriginal.get(p1)=="cat")
    print("Number allocs first get,", get_global_counter()[0]-num_ops)
    
    num_ops = get_global_counter()[0]
    #Check if you can get the same value twice
    assert(SmartPointerOriginal.get(p1)=="cat")
    print("Number allocs second get,", get_global_counter()[0]-num_ops)
    num_ops = get_global_counter()[0]
    p3 = SmartPointerOriginal.copy(p1)
    print("Number allocs second copy,", get_global_counter()[0]-num_ops)
    assert(SmartPointerOriginal.get_count(p3)==3)
    num_ops = get_global_counter()[0]
    assert(SmartPointerOriginal.get(p1)=="cat")
    print("Number allocs third get,", get_global_counter()[0]-num_ops)
    num_ops = get_global_counter()[0]
    SmartPointerOriginal.put(p1, "dog")
    print("Number allocs put,", get_global_counter()[0]-num_ops)
    num_ops = get_global_counter()[0]
    for i in range(10):
        assert(SmartPointerOriginal.get(p1)=="dog")
        print("Number allocs get,", get_global_counter()[0]-num_ops)
        num_ops = get_global_counter()[0]
        assert(SmartPointerOriginal.get(p2)=="dog")
        print("Number allocs get,", get_global_counter()[0]-num_ops)
        num_ops = get_global_counter()[0]
    p4 = SmartPointerOriginal(SmartPointerOriginal.new(SmartPointerOriginal.get(p2)))
    assert(SmartPointerOriginal.get(p4)=="dog")
    print("Number allocs get three copies,", get_global_counter()[0]-num_ops)
    assert(SmartPointerOriginal.get_count(p4)==1)
    assert(SmartPointerOriginal.is_single_reference(p4)==True)
    num_ops = get_global_counter()[0]
    SmartPointerOriginal.put(p1, "horse")
    assert(SmartPointerOriginal.get(p2)=="horse")
    print("Number allocs get three copies,", get_global_counter()[0]-num_ops)
    assert(SmartPointerOriginal.is_single_reference(p1)==False)
    num_ops = get_global_counter()[0]
    assert(SmartPointerOriginal.get(p4)=="dog")
    print("Number allocs get p4 unrelated,", get_global_counter()[0]-num_ops)
    num_ops = get_global_counter()[0]
    SmartPointerOriginal.delete(p3)
    assert(SmartPointerOriginal.get_count(p2)==2)
    print("Number allocs get post delete p3,", get_global_counter()[0]-num_ops)
    num_ops = get_global_counter()[0]
    assert(SmartPointerOriginal.is_single_reference(p2)==False)
    assert(SmartPointerOriginal.get(p2)=="horse")
    print("Number allocs get two copies,", get_global_counter()[0]-num_ops)
    num_ops = get_global_counter()[0]
    SmartPointerOriginal.delete(p2)
    assert(SmartPointerOriginal.is_single_reference(p1)==True)
    assert(SmartPointerOriginal.get_count(p1)==1)
    print("Number allocs get post delete,", get_global_counter()[0]-num_ops)
    num_ops = get_global_counter()[0]
    assert(SmartPointerOriginal.get(p1)=="horse")
    print("Number allocs get post delete,", get_global_counter()[0]-num_ops)
    num_ops = get_global_counter()[0]
    SmartPointerOriginal.delete(p1)
    SmartPointerOriginal.delete(p4)

if __name__ == "__main__":
    set_max_reads(1)
    set_max_writes(1)
    smart_pointer_test_1()
    
    
