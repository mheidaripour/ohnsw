from single_access_machine import alloc, read, write, return_available_address
from typing import Tuple, Any

"""
AddressQueue - original queue built on SAM accesses
not used in oblivious graph traversal
used in SmartPointerOriginal
"""

class AddressQueue:
    # uses SAM addresses when True
    queue_osam = True

    # label for tracking which object incurs SAM accesses
    structure = "AddressQueue"
    
    @staticmethod
    def init_queue() -> Tuple[int, int]:
        """Initialize a SAM-built queue"""
        head = alloc(AddressQueue.queue_osam, AddressQueue.structure)
        tail = head
        return head, tail

    @staticmethod
    def enqueue(tail: int, value: Any) -> int:
        """Add item to end of queue"""
        new_tail = alloc(AddressQueue.queue_osam, AddressQueue.structure)
        write(tail, (value, new_tail), AddressQueue.structure)
        tail = new_tail
        return tail

    @staticmethod
    def dequeue(head: int | None) -> Tuple[Any, None | int]: 
        """Get next entry in queue order"""
        # trying to read from empty queue
        if head is None:
            return None, None

        next_entry = read(head, AddressQueue.structure)
        return_available_address(head)

        if isinstance(next_entry, tuple):
            return next_entry

        else:
            return None, None



def test_queue() -> None:
    head: int | None
    tail: int | None
    head, tail = AddressQueue.init_queue()
    print("Initial Queue:", head, tail)

    tail = AddressQueue.enqueue(tail, "cat")
    print("Enqueue:", head, tail)

    tail = AddressQueue.enqueue(tail, "dog")
    print("Enqueue:", head, tail)

    value, head = AddressQueue.dequeue(head)
    print("Dequeue:", head, tail, value)

    tail = AddressQueue.enqueue(tail, "horse")
    print("Enqueue:", head, tail)

    value, head = AddressQueue.dequeue(head)
    print("Dequeue:", head, tail, value)

    value, head = AddressQueue.dequeue(head)
    print("Dequeue:", head, tail, value)

    value, head = AddressQueue.dequeue(head)
    print("Dequeue:", head, tail, value)

if __name__ == "__main__":
    test_queue()    
