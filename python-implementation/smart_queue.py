from single_access_machine import alloc, read, write, return_available_address
from typing import Any

"""
SmartQueue - new queue built on SAM accesses
used in oblivious graph traversal
not used in SmartPointerOriginal
"""

class SmartQueue:
    # uses SAM addresses when True
    queue_osam = True

    # label for tracking which object incurs SAM accesses
    structure = "SmartQueue"
    
    def __init__(self, head: int | None, tail: int | None) -> None:
        """Initialize a SAM-built queue"""
        self.head = head
        self.tail = tail

    @staticmethod
    def init_queue() -> "SmartQueue":
        """Initialize a fresh SAM-built queue"""
        head = alloc(SmartQueue.queue_osam, SmartQueue.structure)
        tail = head
        queue = SmartQueue(head=head, tail=tail)
        return queue

    @staticmethod
    def enqueue(queue: "SmartQueue", value: Any) -> None:
        """Add item to end of queue"""
        new_tail = alloc(SmartQueue.queue_osam, SmartQueue.structure)
        assert isinstance(queue.tail, int)
        write(queue.tail, (new_tail, value), SmartQueue.structure)
        queue.tail = new_tail

    @staticmethod
    def dequeue(queue: "SmartQueue") -> Any:
        """Get next entry in queue order"""
        # if head equals tail, we know the queue is empty without reading head
        if queue.head == queue.tail:
            return_available_address(queue.head)
            queue.head = None
            queue.tail = None
            return None

        # trying to read from empty queue
        if queue.head is None:
            return None
            
        next_entry = read(queue.head, SmartQueue.structure)
        return_available_address(queue.head)

        if isinstance(next_entry, tuple):
            queue.head, value = next_entry
            return value
        else:
            queue.head, value = next_entry, None
            return value
    


    
def test_queue() -> None:
    queue = SmartQueue.init_queue()
    print("Initial Queue:", queue.head, queue.tail)

    SmartQueue.enqueue(queue, "cat")
    print("Enqueue:", queue.head, queue.tail)

    SmartQueue.enqueue(queue, "dog")
    print("Enqueue:", queue.head, queue.tail)

    value = SmartQueue.dequeue(queue)
    print("Dequeue:", queue.head, queue.tail)

    SmartQueue.enqueue(queue, "horse")
    print("Enqueue:", queue.head, queue.tail)

    value = SmartQueue.dequeue(queue)
    print("Dequeue:", queue.head, queue.tail, value)

    value = SmartQueue.dequeue(queue)
    print("Dequeue:", queue.head, queue.tail, value)

    value = SmartQueue.dequeue(queue)
    print("Dequeue:", queue.head, queue.tail, value)

    value = SmartQueue.dequeue(queue)
    print("Dequeue:", queue.head, queue.tail, value)

if __name__ == "__main__":
    test_queue()
