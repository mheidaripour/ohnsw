from single_access_machine import read, write, alloc, return_available_address
from collections import deque
from hashlib import sha3_256
import random
import unittest
from typing import List, Set, Tuple, Dict, Deque, Any, FrozenSet

"""
SmartAVLTree - AVLTree built on top of SAM
"""

class AVLNode:
    """Storage node for SmartAVLTrees"""

    def __init__(   
            self, 
            hash_key: int | str | bytes | FrozenSet[int], 
            value: Any, 
            children: List[int | None | str] = [None, None], 
            height: int = 1, 
            left_height: int = 0, 
            right_height: int = 0, 
            balance: int = 0, 
            left_balance: int | None = None, 
            right_balance: int | None = None
        ) -> None:
        """Initialize AVLNode"""
        self.hash_key = hash_key 
        self.value = value
        self.children = children # OSAM addresses of children
        self.height = height
        self.left_height = left_height
        self.right_height = right_height
        self.balance = balance
        self.left_balance = left_balance
        self.right_balance = right_balance

    def delete(self):
        """Delete helper function"""
        self.hash_key = None 
        self.value = None
        self.children.clear()
        self.height = -1
        self.left_height = -1
        self.right_height = -1
        self.balance = 0
        self.left_balance = None
        self.right_balance = None


class SmartAVLTree:
    def __init__(self, hash_enabled: bool = True, avl_tree_osam: bool = True) -> None:
        """Initialize a SmartAVLTree"""
        self.root_address: int | None = None
        self.hash_enabled = hash_enabled # toggle hashing
        self.avl_tree_osam = avl_tree_osam
        self.structure = "SmartAVLTree"

    def __get_key(self, key: int | str | bytes | FrozenSet[int]) -> int | str | bytes | FrozenSet[int]:
        """Compute hash of key if hashing is enabled"""
        hash_key: int | str | bytes | FrozenSet[int]
        if self.hash_enabled:
            hash_key = sha3_256(str(key).encode("utf-8")).digest()
        else:
            hash_key = key

        return hash_key

    def build_tree(self, original_elements: Dict[Any, Any] | List[Any] | Set[Any] | Tuple[Any]) -> None:
        """Build entire SmartAVLTree from scratch"""
        if len(original_elements) <= 0:
            return

        # convert to dictionary for processing
        original_dict: Dict[Any, Any] = dict()
        if isinstance(original_elements, list) or isinstance(original_elements, tuple) or isinstance(original_elements, set):
            original_dict = dict()

            # unpack key, value lists / tuples
            element: List[int | str | bytes] | Tuple[int | str | bytes] | int | str | bytes
            for element in original_elements:
                if (isinstance(element, list) or isinstance(element, tuple)) and len(element) >= 2:
                    original_dict[element[0]] = element[1]
                
                else:
                    original_dict[element] = element

        elif isinstance(original_elements, dict):
            original_dict = original_elements

        # cannot build new SmartAVLTree if one already exists
        if self.root_address is not None: 
            # dynamically insert items
            for key, value in original_dict.items():
                self.insert(key, value)
            
        # build new SmartAVLTree from scratch
        else:
            elements: List[Tuple[int | str | bytes | FrozenSet[int], Any]] = []
            for key, value in original_dict.items(): 
                # hash the string representation of each key
                hash_key = self.__get_key(key)
                elements.append((hash_key, value))

            # sort by digest
            elements.sort(key=lambda map_item: map_item[0])

            # assign new root address
            self.root_address = alloc(self.avl_tree_osam, self.structure)

            # left and right bounds contain all items
            left = 0
            right = len(elements) - 1

            # recursively build tree
            self.__build_tree(elements, left, right, self.root_address)

    def __build_tree(self, elements: List[Tuple[int | str | bytes | FrozenSet[int], Any]], left: int, right: int, address: int) -> Tuple[int, int | None]:
        """Recursive function for building SmartAVLTree top to bottom"""
        # terminate if left bound exceeds right
        if left > right: 
            return 0, None
        
        # choose middle element to turn into the current node
        middle = (left + right) // 2 
        hash_key = elements[middle][0]
        value = elements[middle][1]
        children: List[int | None | str] = []

        # recursively determine left attributes (children, height, balance)
        leftree_list = elements[left:middle]
        left_height = 0
        left_balance: int | None = None
        if leftree_list: 
            # if the slice contains elements, there are more nodes to make
            left_address = alloc(self.avl_tree_osam, self.structure)
            children.append(left_address)
            left_height, left_balance = self.__build_tree(elements, left, middle-1, left_address)
        else:
            # no more children to add
            children.append(None)
            
        # recursively determine right attributes (children, height, balance)
        rightree_list = elements[middle+1:right+1]
        right_height = 0
        right_balance: int | None = None
        if rightree_list:
            right_address = alloc(self.avl_tree_osam, self.structure)
            children.append(right_address)
            right_height, right_balance = self.__build_tree(elements, middle+1, right, right_address)

        else:
            # no more children to add
            children.append(None)
        
        # compute and return balance and height information to inform parents 
        height = 1 + max(left_height, right_height)
        balance = left_height - right_height

        # create and write node
        node = AVLNode(hash_key=hash_key, value=value, children=children, height=height, left_height=left_height, right_height=right_height, balance=balance, left_balance=left_balance, right_balance=right_balance)
        write(address, node, self.structure)

        return height, balance

    def __contains__(self, key: int | str | FrozenSet[int]) -> bool:
        """Boolean if key exists inside SmartAVLTree"""
        return self.search(key) is not None

    def search(self, key: int | str | FrozenSet[int]) -> Any:
        """Retrieve the value associated with key in SmartAVLTree"""
        return self.__getitem__(key)

    def __getitem__(self, key: int | str | FrozenSet[int]) -> Any:
        """Retrieve the value associated with key in SmartAVLTree"""
        value = None
        address, node = self.__get_node(key) 
        if address is not None and node is not None:
            value = node.value
            write(address, node, self.structure) # write back final node
        return value

    def __get_node(self, key: int | str | FrozenSet[int]) -> Tuple[int | None, AVLNode | None]:
        """Returns the node associated with key and the write back address"""
        # cannot retrieve if tree does not exist
        if self.root_address is None:
            return None, None

        # client side queue 
        # enqueue newly allocated addresses that are assigned as children
        # need to process them in order to write back nodes later
        addresses: Deque[int] = deque()

        hash_key = self.__get_key(key)

        # get root from root address
        current_node = read(self.root_address, self.structure) 
        return_available_address(self.root_address) 

        # allocate new root address
        self.root_address = alloc(self.avl_tree_osam, self.structure)
        addresses.append(self.root_address)
        
        # traverse down tree until finding the matching key
        while (len(addresses) > 0):
            # pre-allocated address that the current node will be written back to
            current_address = addresses.popleft()
            current_hash_key = current_node.hash_key

            if hash_key == current_hash_key: # found target
                return current_address, current_node
            elif hash_key < current_hash_key: 
                next_child_index = 0 # get left child
            else: 
                next_child_index = 1 # get right child

            # load address of next child
            next_address = current_node.children[next_child_index]

            # if a leaf is reached
            if next_address is None:
                break

            # allocate new address that the child node will be written back to
            new_next_address = alloc(self.avl_tree_osam, self.structure) 
            
            # update child address at current node 
            current_node.children[next_child_index] = new_next_address 

            # enqueue to be written back later
            addresses.append(new_next_address)
                
            # write current node back to current address
            write(current_address, current_node, self.structure)

            # read child node
            current_node = read(next_address, self.structure)
            return_available_address(next_address) 

        # write back final node if target wasn't found
        write(current_address, current_node, self.structure) 

        return None, None
    
    def height(self, key: int | str | FrozenSet[int]) -> int | None:
        """Outputs height of node associated with key"""
        height = None
        address, node = self.__get_node(key)
        if address is not None and node is not None:
            height = node.height
            write(address, node, self.structure) # write back final node
        return height

    def insert(self, key: int | str | FrozenSet[int], value: Any) -> None:
        """
        Inserts a new node associated with (key, value), or
        Updates value at an existing node associated with key 
        """
        self.__setitem__(key, value)

    def __setitem__(self, key: int | str | FrozenSet[int], value: Any) -> None:
        """
        Inserts a new node associated with (key, value), or
        Updates value at an existing node associated with key 
        """
        hash_key = self.__get_key(key)

        # set item as root if SmartAVLTree is empty
        if self.root_address is None:
            self.root_address = alloc(self.avl_tree_osam, self.structure)
            write(self.root_address, AVLNode(hash_key=hash_key, value=value, children=[None, None], height=1, left_height=0, right_height=0, balance=0, left_balance=None, right_balance=None), self.structure)
            return 

        # read root 
        current_node = read(self.root_address, self.structure) 
        return_available_address(self.root_address)
    
        # find leaf to imsert new node below or target node to update
        nodes = [] # locally store nodes encountered on the path down
        is_insert = True # distinguish insertions and updates
        while True:
            current_hash_key = current_node.hash_key
            nodes.append(current_node)

            if hash_key == current_hash_key: # update to an existing node
                current_node.value = value
                is_insert = False 
                break
            elif hash_key < current_hash_key:
                next_child_index = 0 # get left child
            else:
                next_child_index = 1 # get right child

            next_address = current_node.children[next_child_index]

            # fill with "REPLACE" to know which children need to be rewritten
            # avoid assigning new children when nodes may need to be rebalanced
            current_node.children[next_child_index] = "REPLACE"

            if next_address is None: # leaf node reached
                # create a new node to be inserted at the bottom of the tree
                nodes.append(AVLNode(hash_key=hash_key, value=value, children=[None, None], height=1, left_height=0, right_height=0, balance=0, left_balance=None, right_balance=None))
                break

            # read next child 
            current_node = read(next_address, self.structure) 
            return_available_address(next_address) 

        # moved child address to update at a parent
        child_address = None

        if is_insert: 
            # iterate over path of nodes from bottom to top
            # lower nodes need to be rebalanced and written back first
            i = len(nodes) - 1
            
            # to ignore type errors of 
            child_hash_key: Any
            
            while (i >= 0):
                # update possible heights / balances changes of children
                # last node does not have to update left / right children stats
                if nodes[i].hash_key != hash_key: 
                    if child_hash_key < nodes[i].hash_key:
                        nodes[i].left_height = nodes[i+1].height
                        nodes[i].left_balance = nodes[i+1].balance
                    else:
                        nodes[i].right_height = nodes[i+1].height
                        nodes[i].right_balance = nodes[i+1].balance
                        
                    nodes[i].height = 1 + max(nodes[i].left_height, nodes[i].right_height)
                    nodes[i].balance = nodes[i].left_height - nodes[i].right_height

                # write back nodes that are too deep to be part of our rotation
                # all insertion rotations consider the lowest three nodes 
                if i + 2 < len(nodes) - 1:
                    child_address = self.__write_back_last_node(child_address, nodes)

                # get balance 
                balance = nodes[i].balance

                """
                Insertions require at most one rotation, so break out after performing one
                """

                # left left rotation
                # before:
                #         z
                #       / 
                #     y
                #   /
                # x
                # after:
                #     y
                #   /   \
                # x       z
                if balance > 1 and hash_key < child_hash_key:
                    # print("LL rotate")
                    self.__right_rotate(nodes[i+1], nodes[i])
                    child_address = self.__write_back_last_node(child_address, nodes) # write node_x back
                    nodes[i+1].children[0] = child_address # assign address of node_x as child to node_y
                    node_z = nodes.pop(i) # get node_z
                    child_address = alloc(self.avl_tree_osam, self.structure) 
                    nodes[i].children[1] = child_address # assign address of node_z as child to node_y
                    write(child_address, node_z, self.structure) # write node_z back
                    break

                # right right rotation
                # before:
                # x        
                #   \     
                #     y
                #       \  
                #         z
                # after:
                #     y
                #   /   \
                # x       z
                if balance < -1 and hash_key > child_hash_key:
                    # print("RR rotate")
                    self.__left_rotate(nodes[i], nodes[i+1])
                    child_address = self.__write_back_last_node(child_address, nodes) # write node_z back
                    nodes[i+1].children[1] = child_address # assign address of node_z as child to node_y
                    node_x = nodes.pop(i) # get node_x
                    child_address = alloc(self.avl_tree_osam, self.structure)
                    nodes[i].children[0] = child_address # assign address of node_x as child to node_y
                    write(child_address, node_x, self.structure) # write node_x back
                    break

                # left right rotation
                # before:
                #         z 
                #       /  
                #     /
                #   /  
                # x       
                #  \        
                #   \       
                #    \  
                #     y  
                # # after:
                #     y
                #   /   \
                # x       z  
                if balance > 1 and hash_key > child_hash_key:
                    # print("LR rotate")
                    self.__left_rotate(nodes[i+1], nodes[i+2])
                    self.__right_rotate(nodes[i+2], nodes[i])

                    # put given child address at either node_z or node_x
                    node_z = nodes.pop(i)
                    node_x = nodes.pop(i)
                    if node_z.children[0] == "REPLACE":
                        node_z.children[0] = child_address
                    else:
                        node_x.children[1] = child_address
                    
                    child_address = alloc(self.avl_tree_osam, self.structure) # get new address for node_x
                    write(child_address, node_x, self.structure) # write node_x back
                    nodes[i].children[0] = child_address # assign node_x address to node_y
                    child_address = alloc(self.avl_tree_osam, self.structure) # get new address for node_z
                    write(child_address, node_z, self.structure) # write node_x back
                    break

                # right left rotation
                # before:
                # x        
                #   \     
                #     \
                #       \  
                #         z
                #       /   
                #      /   
                #     /
                #    y  
                # after:
                #     y
                #   /   \
                # x       z
                if balance < -1 and hash_key < child_hash_key:
                    # print("RL rotate")
                    self.__right_rotate(nodes[i+2], nodes[i+1])
                    self.__left_rotate(nodes[i], nodes[i+2])

                    # put given child address at either node_z or node_x
                    node_x = nodes.pop(i)
                    node_z = nodes.pop(i)
                    if node_z.children[0] == "REPLACE":
                        node_z.children[0] = child_address
                    else:
                        node_x.children[1] = child_address

                    child_address = alloc(self.avl_tree_osam, self.structure) # get new address for node_x
                    write(child_address, node_x, self.structure) # write node_x back
                    nodes[i].children[0] = child_address # assign node_x address to node_y
                    child_address = alloc(self.avl_tree_osam, self.structure) # get new address for node_z
                    write(child_address, node_z, self.structure) # write node_x back
                    break

                child_hash_key = nodes[i].hash_key
                i -= 1

            # update right or left balance for the last node that may change
            if i - 1 >= 0:
                if nodes[i-1].hash_key < nodes[i].hash_key:
                    nodes[i-1].right_balance = nodes[i].balance
                else:
                    nodes[i-1].left_balance = nodes[i].balance

        # write back all (leftover) nodes 
        while (nodes):
            child_address = self.__write_back_last_node(child_address, nodes)

        # update root address
        self.root_address = child_address

    def __write_back_last_node(self, child_address: int | None, nodes: List[AVLNode]) -> int:
        """
        Takes the last node in the list of nodes
        Updates node with address of its moved child
        Generates a new address and writes node back
        Outputs this new address to update its parent later
        """
        # helper function that restores addresses
        node = nodes.pop()
        for j in range(len(node.children)):
            if node.children[j] == "REPLACE":
                node.children[j] = child_address
        address = alloc(self.avl_tree_osam, self.structure)
        write(address, node, self.structure)
        return address
            
    def __right_rotate(self, node_y: AVLNode, node_z: AVLNode) -> None:
        """Elevate node y (left) and lower node z (right)"""
        # before:
        #     z
        #   / 
        # y
        # after:
        # y
        #   \
        #     z
        # set right child of node being raised to left child of node being lowered
        node_z.children[0] = node_y.children[1]
        node_z.left_height = node_y.right_height
        node_z.height = 1 + max(node_z.left_height, node_z.right_height)
        node_z.left_balance = node_y.right_balance
        node_z.balance = node_z.left_height - node_z.right_height

        # update new parent
        node_y.right_height = node_z.height
        node_y.children[1] = "REPLACE"
        node_y.height = 1 + max(node_y.left_height, node_y.right_height)
        node_y.right_balance = node_z.balance
        node_y.balance = node_y.left_height - node_y.right_height

    def __left_rotate(self, node_x: AVLNode, node_y: AVLNode) -> None:
        """Lower node x (left) and elevate node y (right)"""
        # before:
        # x        
        #   \     
        #     y
        # after:
        #     y
        #   /  
        # x   
        # set left child of node being raised to right child of node being lowered
        node_x.children[1] = node_y.children[0]
        node_x.right_height = node_y.left_height
        node_x.height = 1 + max(node_x.left_height, node_x.right_height)
        node_x.right_balance = node_y.left_balance
        node_x.balance = node_x.left_height - node_x.right_height

        # update new parent
        node_y.left_height = node_x.height
        node_y.children[0] = "REPLACE"
        node_y.height = 1 + max(node_y.left_height, node_y.right_height)
        node_y.left_balance = node_x.balance
        node_y.balance = node_y.left_height - node_y.right_height

    def delete(self, key: int | str | FrozenSet[int]) -> Any:
        """Delete and return an item in the SmartAVLTree"""
        return self.__delitem__(key)

    def __delitem__(self, key: int | str | FrozenSet[int]) -> Any:
        """
        Delete an item in the SmartAVLTree
        Does not return an item when overloaded function is directly called
        """
        # cannot delete from an empty tree
        if self.root_address is None:
            return 

        hash_key = self.__get_key(key)

        # read root
        current_node = read(self.root_address, self.structure) 
        return_available_address(self.root_address)
    
        # find node to delete
        nodes = [] # locally store nodes encountered on the path down
        result = None
        found = False
        while True:
            current_hash_key = current_node.hash_key
            nodes.append(current_node)

            if hash_key == current_hash_key: # node to delete
                result = current_node.value
                found = True
                break
            elif hash_key < current_hash_key:
                next_child_index = 0 # get left child
            else:
                next_child_index = 1 # get right child

            next_address = current_node.children[next_child_index]

            # fill with "REPLACE" to know which children need to be rewritten
            # avoid assigning new children when nodes may need to be rebalanced
            current_node.children[next_child_index] = "REPLACE"

            if next_address is None: # leaf node reached
                break

            # read next child 
            current_node = read(next_address, self.structure) 
            return_available_address(next_address)

        # moved child address to update at a parent
        child_address = None

        if found:
            node = nodes.pop() # node to be deleted

            """
            Ensure children of deleted node are correctly handled
            
            - If a node has multiple children, search the left subtree 
            for the largest hash key smaller than the target. Either the 
            first left child or its farthest right leaf is

            - If a node only has one child, it must be a leaf node
            (otherwise a rotation would've occurred), so we easily 
            set the single child as the write back address
            """

            # the node to delete has two children
            if node.children[0] is not None and node.children[1] is not None:
                # put node back in path because we read down further
                nodes.append(node)

                # read left child and add to path
                nodes.append(read(node.children[0], self.structure))
                return_available_address(node.children[0]) 
                node.children[0] = "REPLACE"

                # left child has a right child, so traverse down to the right
                # until hitting a leaf to find the largest hash key smaller
                # than the target
                if nodes[-1].children[1] is not None: 
                    while nodes[-1].children[1] is not None:
                        nodes.append(read(nodes[-1].children[1], self.structure))
                        return_available_address(nodes[-1].children[1]) 
                        nodes[-2].children[1] = "REPLACE"

                # move max child stats to target node (preserves AVL invariant)
                # only need to consider max child's possible left child:
                #   - if max child is the first left child, it had no right children
                #   - otherwise, max child is the farthest right leaf 
                max_child = nodes.pop()
                node.value = max_child.value
                node.hash_key = max_child.hash_key
                child_address = max_child.children[0] 
                node = max_child

            # left child only
            elif node.children[0] is not None:
                child_address = node.children[0]
                # child_address = node.children[0]

            # right child only
            elif node.children[1] is not None:
                child_address = node.children[1]

            # delete node
            node.delete()
            
            i = len(nodes) - 1
            while (i >= 0):
                # update self and child heights and balances
                if i == len(nodes) - 1:
                    if child_address is None:
                        child_height = 0
                        child_balance = None
                    else:
                        child_height = 1
                        child_balance = 0
                else:
                    child_height = nodes[i+1].height
                    child_balance = nodes[i+1].balance
                
                if nodes[i].children[0] == "REPLACE":
                    nodes[i].left_height = child_height
                    nodes[i].left_balance = child_balance
                else:
                    nodes[i].right_height = child_height
                    nodes[i].right_balance = child_balance
                        
                # update height and balance
                nodes[i].height = 1 + max(nodes[i].left_height, nodes[i].right_height)
                nodes[i].balance = nodes[i].left_height - nodes[i].right_height
                
                # stats for comparison
                balance = nodes[i].balance
                left_balance = nodes[i].left_balance
                right_balance = nodes[i].right_balance

                # write back nodes that are too deep to be part of our rotation
                if i + 2 < len(nodes) - 1:
                    child_address = self.__write_back_last_node(child_address, nodes)

                # left left rotation
                # before:
                #         z
                #       / 
                #     y
                #   /
                # x
                # after:
                #     y
                #   /   \
                # x       z
                if balance > 1 and left_balance is not None and left_balance >= 0:
                    # print("LL rotate")

                    # write back possibly dangling third node
                    if len(nodes) - 1 - i == 2:
                        child_address = self.__write_back_last_node(child_address, nodes)

                    # if nodes contains an i+1th node
                    left_child = None
                    if len(nodes) - 1 - i == 1:
                        # if i+1th node will be rotated (it is considered node_y in the diagram)
                        if nodes[i+1].hash_key < nodes[i].hash_key:
                            left_child = nodes.pop() # remove node_y
                            left_child.children[0] = child_address # assign node_x address as left child of node_y
                        else: 
                            # i+1th node will not be rotated and can be written back
                            child_address = self.__write_back_last_node(child_address, nodes)

                    # if nodes does not contain a child node to rotate
                    if left_child is None:
                        left_child = read(nodes[i].children[0], self.structure)
                        return_available_address(nodes[i].children[0]) 
                        nodes[i].children[0] = "REPLACE"
                        nodes[i].children[1] = child_address # assign address of i+1th node as right child of node_z

                    self.__right_rotate(left_child, nodes[i])

                    # write back node_z
                    child_address = alloc(self.avl_tree_osam, self.structure) 
                    write(child_address, nodes[i], self.structure)
                    nodes.pop()
                    nodes.append(left_child)

                # right right rotation
                # before:
                # x        
                #   \     
                #     y
                #       \  
                #         z
                # after:
                #     y
                #   /   \
                # x       z
                elif balance < -1 and right_balance is not None and right_balance <= 0:
                    # print("RR rotate")
                    
                    # write back possibly dangling third node
                    if len(nodes) - 1 - i == 2:
                        child_address = self.__write_back_last_node(child_address, nodes)

                    # if nodes contains an i+1th node
                    right_child = None
                    if len(nodes) - 1 - i == 1:
                        # if i+1th node will be rotated (it is considered node_y in the diagram)
                        if nodes[i+1].hash_key > nodes[i].hash_key:
                            right_child = nodes.pop() # remove node_y
                            right_child.children[1] = child_address # assign node_z address as right child of node_y
                        else: 
                            # i+1th node will not be rotated and can be written back
                            child_address = self.__write_back_last_node(child_address, nodes)

                    # if nodes does not contain a child node to rotate
                    if right_child is None:
                        right_child = read(nodes[i].children[1], self.structure)
                        return_available_address(nodes[i].children[1]) 
                        nodes[i].children[1] = "REPLACE"
                        nodes[i].children[0] = child_address # assign address of i+1th node as left child of node_x

                    self.__left_rotate(nodes[i], right_child)
                    
                    # write back node_x
                    child_address = alloc(self.avl_tree_osam, self.structure) 
                    write(child_address, nodes[i], self.structure)
                    nodes.pop()
                    nodes.append(right_child)

                # left right rotation
                # before:
                #         z 
                #       /  
                #     /
                #   /  
                # x       
                #  \        
                #   \       
                #    \  
                #     y  
                # after:
                #     y
                #   /   \
                # x       z
                elif balance > 1 and left_balance is not None and left_balance <= 0:
                    # print("LR rotate")

                    # if nodes contains i+1th and i+2th nodes
                    left_child = right_child = None
                    # nodes may contain node_x and node_y, only node_x, or neither
                    if len(nodes) - 1 - i == 2:
                        # nodes contains both
                        if nodes[i+1].hash_key < nodes[i].hash_key and nodes[i+1].hash_key < nodes[i+2].hash_key:
                            right_child = nodes.pop() # remove node_y
                            # assign child address correctly
                            if right_child.children[0] == "REPLACE": 
                                right_child.children[0] = child_address
                            else: 
                                right_child.children[1] = child_address
                            left_child = nodes.pop() # remove node_x
                        
                        # contains node_x
                        elif nodes[i+1].hash_key < nodes[i].hash_key:
                            # write back node that won't be rotated
                            child_address = self.__write_back_last_node(child_address, nodes)
                            left_child = nodes.pop() # remove node_x
                            left_child.children[0] = child_address
                        
                        # contains neither
                        else:
                            # i+1th and 1+2th nodes will not be rotated and can be written back
                            for j in range(2):
                                child_address = self.__write_back_last_node(child_address, nodes)
                    
                    # nodes contains only node_x or neither
                    elif len(nodes) - 1 - i == 1:
                        if nodes[i+1].hash_key < nodes[i].hash_key:
                            left_child = nodes.pop() # remove node_x
                            left_child.children[0] = child_address
                        else: 
                            child_address = self.__write_back_last_node(child_address, nodes)
               
                    # if left child needs to be read
                    if left_child is None:
                        left_child = read(nodes[i].children[0], self.structure)
                        return_available_address(nodes[i].children[0]) 
                        nodes[i].children[0] = "REPLACE"
                        nodes[i].children[1] = child_address 

                    # if right child needs to be read
                    if right_child is None:
                        right_child = read(left_child.children[1], self.structure)
                        return_available_address(left_child.children[1]) 
                        left_child.children[1] = "REPLACE"

                    self.__left_rotate(left_child, right_child)
                    self.__right_rotate(right_child, nodes[i])
                    
                    # write back node_z
                    child_address = alloc(self.avl_tree_osam, self.structure) 
                    write(child_address, nodes[i], self.structure)
                    right_child.children[1] = child_address
                    nodes.pop()

                    # write back node_x
                    child_address = alloc(self.avl_tree_osam, self.structure) 
                    write(child_address, left_child, self.structure)

                    nodes.append(right_child)

                # right left rotation
                # before:
                # x        
                #   \     
                #     \
                #       \  
                #         z
                #       /   
                #      /   
                #     /
                #    y 
                # after:
                #     y
                #   /   \
                # x       z
                elif balance < -1 and right_balance is not None and right_balance > 0:
                    # print("RL rotate")

                    # if nodes contains i+1th and i+2th nodes
                    left_child = right_child = None
                    # nodes may contain node_x and node_y, only node_x, or neither
                    if len(nodes) - 1 - i == 2:
                        # nodes contains both
                        if nodes[i+1].hash_key > nodes[i].hash_key and nodes[i+1].hash_key > nodes[i+2].hash_key:
                            left_child = nodes.pop() # remove node_y
                            # assign child address correctly
                            if left_child.children[0] == "REPLACE": left_child.children[0] = child_address
                            else: left_child.children[1] = child_address
                            right_child = nodes.pop() # remove node_x
                        
                        # contains node_x
                        elif nodes[i+1].hash_key > nodes[i].hash_key:
                            # write back node that won't be rotated
                            child_address = self.__write_back_last_node(child_address, nodes)
                            right_child = nodes.pop() # remove node_x
                            right_child.children[1] = child_address
                        
                        # contains neither
                        else:
                            # i+1th and 1+2th nodes will not be rotated and can be written back
                            for j in range(2):
                                child_address = self.__write_back_last_node(child_address, nodes)
                    
                    # nodes contains only node_x or neither
                    elif len(nodes) - 1 - i == 1:
                        if nodes[i+1].hash_key > nodes[i].hash_key:
                            right_child = nodes.pop() # remove node_x
                            right_child.children[1] = child_address
                        else: 
                            child_address = self.__write_back_last_node(child_address, nodes)
                    
                    # if right child needs to be read
                    if right_child is None:
                        right_child = read(nodes[i].children[1], self.structure)
                        return_available_address(nodes[i].children[1]) 
                        nodes[i].children[1] = "REPLACE"
                        nodes[i].children[0] = child_address

                    # if right child needs to be read
                    if left_child is None:
                        left_child = read(right_child.children[0], self.structure)
                        return_available_address(right_child.children[0]) 
                        right_child.children[0] = "REPLACE"

                    self.__right_rotate(left_child, right_child)
                    self.__left_rotate(nodes[i], left_child)
                    
                    # write back node_x
                    child_address = alloc(self.avl_tree_osam, self.structure) 
                    write(child_address, nodes[i], self.structure)
                    left_child.children[0] = child_address
                    nodes.pop()

                    # write back node_z
                    child_address = alloc(self.avl_tree_osam, self.structure) 
                    write(child_address, right_child, self.structure)

                    nodes.append(left_child)
            
                i -= 1

        # write back final nodes
        while (nodes):
            child_address = self.__write_back_last_node(child_address, nodes)

        self.root_address = child_address
        return result
    
    def bfs(self, key: int | str | bytes | FrozenSet[int] | None = None) -> List[Any]:
        """
        Performs bfs down the SmartAVLTree and collects all values
        If specified, only collects values from nodes below key
        """

        values: List[Any] = []

        # there are no nodes to visit if there is no root
        if self.root_address is None:
            return values
        
        if key is not None:
            hash_key = self.__get_key(key)
        else:
            hash_key = None

        # queues for bfs
        addresses: Deque[int] = deque()
        nodes: Deque[AVLNode] = deque()

        # process root
        nodes.append(read(self.root_address, self.structure)) 
        return_available_address(self.root_address) 
        self.root_address = alloc(self.avl_tree_osam, self.structure) 
        addresses.append(self.root_address)
        
        # disable collecting until the desired key has been found
        # automatically enabled otherwise
        can_collect_values = True if key is None else False

        # traverse down SmartAVLTree
        while (len(addresses) > 0 and len(nodes) > 0):
            current_node = nodes.popleft()
            current_address = addresses.popleft()

            # detect specified key
            if hash_key is not None and current_node.hash_key == hash_key:
                can_collect_values = True

                # all other queued nodes are along other paths
                # so empty the queue and write everything back
                while len(nodes):
                    write_back_node = nodes.popleft()
                    write_back_address = addresses.popleft()
                    write(write_back_address, write_back_node, self.structure)

            if can_collect_values:
                values.append(current_node.value)

            # enqueue all children
            # allocate new addresses to write back later
            for i in range(len(current_node.children)):
                address = current_node.children[i]
                if isinstance(address, int):
                    nodes.append(read(address, self.structure))
                    return_available_address(address) 
                    address = alloc(self.avl_tree_osam, self.structure)
                    addresses.append(address)
                    current_node.children[i] = address

            # write back current node
            write(current_address, current_node, self.structure)
        
        return values
    
    def dfs(self, key: int | str | bytes | FrozenSet[int] | None = None) -> List[Any]:
        """
        Performs dfs down the SmartAVLTree and collects all values
        If specified, only collects values from nodes below key
        """

        values: List[Any] = []

        # there are no nodes to visit if there is no root
        if self.root_address is None:
            return values
        
        if key is not None:
            hash_key = self.__get_key(key)
        else:
            hash_key = None

        # stacks for dfs
        addresses = []
        nodes: List[AVLNode] = []

        # process root
        nodes.append(read(self.root_address, self.structure)) 
        return_available_address(self.root_address)
        self.root_address = alloc(self.avl_tree_osam, self.structure) 
        addresses.append(self.root_address)
        
        # disable collecting until the desired key has been found
        # automatically enabled otherwise
        can_collect_values = True if key is None else False

        # traverse down SmartAVLTree
        while (len(addresses) > 0 and len(nodes) > 0):
            current_node = nodes.pop()
            current_address = addresses.pop()

            # detect specified key
            if current_node.hash_key == hash_key:
                can_collect_values = True

                # all other queued nodes are along other paths
                # so empty the queue and write everything back
                while len(nodes):
                    write_back_node = nodes.pop()
                    write_back_address = addresses.pop()
                    write(write_back_address, write_back_node, self.structure)

            if can_collect_values:
                values.append(current_node.value)

            # push all children onto stack
            # allocate new addresses to write back later
            # process children right to left so we go deep left first
            for i in range(1, -1, -1):
                address = current_node.children[i]
                if isinstance(address, int):
                    nodes.append(read(address, self.structure))
                    return_available_address(address) 
                    address = alloc(self.avl_tree_osam, self.structure)
                    addresses.append(address)
                    current_node.children[i] = address

            # write back current node
            write(current_address, current_node, self.structure)
        
        return values

    def clear(self) -> None:
        """Clear SmartAVLTree by reading every address"""
        if self.root_address is None:
            return 
        
        # queue nodes for deleting
        nodes: Deque[AVLNode] = deque()
        nodes.append(read(self.root_address, self.structure)) 
        return_available_address(self.root_address) 
        self.root_address = None

        # traverse down tree and delete nodes
        while (len(nodes) > 0):
            current_node = nodes.popleft()
            for i in range(len(current_node.children)):
                address = current_node.children[i]
                if isinstance(address, int):
                    nodes.append(read(address, self.structure))
                    return_available_address(address) 
                    current_node.children[i] = None

            # delete node
            current_node.delete()




class test_smart_avl_tree(unittest.TestCase):
    """Class for testing different SmartAVLTree functions"""

    def setUp(self) -> None:
        self.item_list = []
        self.item_dict = dict()
        for i in range(1, 16):
            self.item_list.append(i)
            self.item_dict[i] = i
        
        # test building from lists and dicts on trees with and w/o hashing
        self.tree_list = SmartAVLTree()
        self.tree_list_no_hash = SmartAVLTree(hash_enabled=False)
        self.tree_dict = SmartAVLTree()
        self.tree_dict_no_hash = SmartAVLTree(hash_enabled=False)

        # build trees
        self.tree_list.build_tree(original_elements=self.item_list)
        self.tree_list_no_hash.build_tree(original_elements=self.item_list)
        self.tree_dict.build_tree(original_elements=self.item_dict)
        self.tree_dict_no_hash.build_tree(original_elements=self.item_dict)

    def test_search(self) -> None:
        for i in range(1, 16):
            self.assertEqual(self.tree_list[i], i)
            self.assertEqual(self.tree_list_no_hash[i], i)

            self.assertEqual(self.tree_dict.search(i), i)
            self.assertEqual(self.tree_dict_no_hash.search(i), i)

    def test_height(self) -> None:
        # can only test heights on deterministic non-hashed trees
        for i in range(1, 16, 2):
            self.assertEqual(self.tree_list_no_hash.height(i), 1)
            self.assertEqual(self.tree_dict_no_hash.height(i), 1)

        for i in range(2, 15, 4):
            self.assertEqual(self.tree_list_no_hash.height(i), 2)
            self.assertEqual(self.tree_dict_no_hash.height(i), 2)

        for i in range(4, 12, 8):
            self.assertEqual(self.tree_list_no_hash.height(i), 3)
            self.assertEqual(self.tree_dict_no_hash.height(i), 3)
        
        self.assertEqual(self.tree_list_no_hash.height(8), 4)
        self.assertEqual(self.tree_dict_no_hash.height(8), 4)

    def test_bfs(self) -> None:
        # test exact order given by bfs of unhashed
        # cannot test order for hashed due to randomness
        bfs_order1 = [8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15]

        self.assertEqual(self.tree_list_no_hash.bfs(), bfs_order1)
        self.assertEqual(self.tree_dict_no_hash.bfs(), bfs_order1)

        self.assertEqual(self.tree_list_no_hash.bfs(8), bfs_order1)
        self.assertEqual(self.tree_dict_no_hash.bfs(8), bfs_order1)

        bfs_order2 = [12, 10, 14, 9, 11, 13, 15]

        self.assertEqual(self.tree_list_no_hash.bfs(12), bfs_order2)
        self.assertEqual(self.tree_dict_no_hash.bfs(12), bfs_order2)

        bfs_order3 = [6, 5, 7]

        self.assertEqual(self.tree_list_no_hash.bfs(6), bfs_order3)
        self.assertEqual(self.tree_dict_no_hash.bfs(6), bfs_order3)

        bfs_order4 = [3]

        self.assertEqual(self.tree_list_no_hash.bfs(3), bfs_order4)
        self.assertEqual(self.tree_dict_no_hash.bfs(3), bfs_order4)

        # confirm bfs returns correct values
        tree_list_bfs = self.tree_list.bfs()
        tree_dict_bfs = self.tree_dict.bfs()

        for i in range(1, 16):
            self.assertIn(i, tree_list_bfs)
            self.assertIn(i, tree_dict_bfs)

    def test_dfs(self) -> None:
        # test exact order given by dfs of unhashed
        # cannot test order for hashed due to randomness
        unhashed_order1 = [8, 4, 2, 1, 3, 6, 5, 7, 12, 10, 9, 11, 14, 13, 15]

        self.assertEqual(self.tree_list_no_hash.dfs(), unhashed_order1)
        self.assertEqual(self.tree_dict_no_hash.dfs(), unhashed_order1)

        self.assertEqual(self.tree_list_no_hash.dfs(8), unhashed_order1)
        self.assertEqual(self.tree_dict_no_hash.dfs(8), unhashed_order1)

        unhashed_order2 = [4, 2, 1, 3, 6, 5, 7]

        self.assertEqual(self.tree_list_no_hash.dfs(4), unhashed_order2)
        self.assertEqual(self.tree_dict_no_hash.dfs(4), unhashed_order2)

        unhashed_order3 = [10, 9, 11]

        self.assertEqual(self.tree_list_no_hash.dfs(10), unhashed_order3)
        self.assertEqual(self.tree_dict_no_hash.dfs(10), unhashed_order3)

        unhashed_order4 = [15]

        self.assertEqual(self.tree_list_no_hash.bfs(15), unhashed_order4)
        self.assertEqual(self.tree_dict_no_hash.bfs(15), unhashed_order4)

        # confirm dfs returns correct values
        tree_list_dfs = self.tree_list.dfs()
        tree_dict_dfs = self.tree_dict.dfs()

        for i in range(1, 16):
            self.assertIn(i, tree_list_dfs)
            self.assertIn(i, tree_dict_dfs)

    def test_update(self) -> None:
        # use insert to update current items only
        for i in range(1, 16):
            self.tree_list[i] = str(i)
            self.tree_list_no_hash[i] = str(i)
            self.tree_dict[i] = str(i)
            self.tree_dict_no_hash[i] = str(i)

        for i in range(1, 16):
            self.assertEqual(self.tree_list[i], str(i))
            self.assertEqual(self.tree_list_no_hash[i], str(i))
            self.assertEqual(self.tree_dict[i], str(i))
            self.assertEqual(self.tree_dict_no_hash[i], str(i))

        for i in range(1, 16):
            self.tree_list.insert(i, i)
            self.tree_list_no_hash.insert(i, i)
            self.tree_dict.insert(i, i)
            self.tree_dict_no_hash.insert(i, i)

        for i in range(1, 16):
            self.assertEqual(self.tree_list[i], i)
            self.assertEqual(self.tree_list_no_hash[i], i)
            self.assertEqual(self.tree_dict[i], i)
            self.assertEqual(self.tree_dict_no_hash[i], i)
            
    def test_insert(self) -> None:
        # add new items
        for i in range(16, 32):
            self.tree_list[i] = i
            self.tree_list_no_hash[i] = i
            self.tree_dict[i] = i
            self.tree_dict_no_hash[i] = i

        for i in range(16, 32):
            self.assertEqual(self.tree_list[i], i)
            self.assertEqual(self.tree_list_no_hash[i], i)
            self.assertEqual(self.tree_dict[i], i)
            self.assertEqual(self.tree_dict_no_hash[i], i)

        bfs = [16, 8, 24, 4, 12, 20, 28, 2, 6, 10, 14, 18, 22, 26, 30, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

        self.assertEqual(self.tree_list_no_hash.bfs(), bfs)
        self.assertEqual(self.tree_dict_no_hash.bfs(), bfs)

        dfs = [16, 8, 4, 2, 1, 3, 6, 5, 7, 12, 10, 9, 11, 14, 13, 15, 24, 20, 18, 17, 19, 22, 21, 23, 28, 26, 25, 27, 30, 29, 31]

        self.assertEqual(self.tree_list_no_hash.dfs(), dfs)
        self.assertEqual(self.tree_dict_no_hash.dfs(), dfs)

    def test_delete(self) -> None:
        # add new items
        for i in range(16, 32):
            self.tree_list.delete(i)
            self.tree_list_no_hash.delete(i)
            del self.tree_dict[i]
            del self.tree_dict_no_hash[i] 

        for i in range(16, 32):
            self.assertEqual(self.tree_list[i], None)
            self.assertEqual(self.tree_list_no_hash[i], None)
            self.assertEqual(self.tree_dict[i], None)
            self.assertEqual(self.tree_dict_no_hash[i], None)

        bfs = [8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15]

        self.assertEqual(self.tree_list_no_hash.bfs(), bfs)
        self.assertEqual(self.tree_dict_no_hash.bfs(), bfs)

        dfs = [8, 4, 2, 1, 3, 6, 5, 7, 12, 10, 9, 11, 14, 13, 15]

        self.assertEqual(self.tree_list_no_hash.dfs(), dfs)
        self.assertEqual(self.tree_dict_no_hash.dfs(), dfs)

    def test_clear(self) -> None:
        self.tree_list.clear()
        self.tree_list_no_hash.clear()
        self.tree_dict.clear()
        self.tree_dict_no_hash.clear()

        for i in range(1, 16):
            self.assertEqual(self.tree_list[i], None)
            self.assertEqual(self.tree_list_no_hash[i], None)
            self.assertEqual(self.tree_dict[i], None)
            self.assertEqual(self.tree_dict_no_hash[i], None)

    def test_rotations(self) -> None:
        # this specific order tests all 4 possible rotations during insertions and deletions
        order = [25, 45, 16, 24, 21, 30, 41, 9, 19, 33, 43, 17, 46, 34, 4, 11, 42, 32, 14, 27, 47, 48, 22, 2, 3, 49, 20, 18, 39, 28, 6, 36, 12, 15, 38, 1, 8, 37, 26, 13, 23, 5, 7, 35, 10, 44, 31, 0, 29, 40]

        for i in order:
            self.tree_list_no_hash[i] = i
        
        for i in order:
            self.assertEqual(self.tree_list_no_hash[i], i)

        for i in range(len(order)):
            del self.tree_list_no_hash[order[i]]
            bfs = self.tree_list_no_hash.bfs()
            for j in order[i+1:]:
                self.assertIn(j, bfs)

if __name__ == "__main__":   
    unittest.main(exit=False)
