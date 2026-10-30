from typing import List

"""
AttributeAtPositions - helper class for copying a  
list of pointers at select positions only
"""

class AttributeAtPositions:
    def __init__(self, attribute: str, positions: List[int] = []) -> None:
        """
        Initialize AttributeAtPositions object
        attribute: string name of pointer attribute to return
        positions: list of integer indices to copy
        """
        self.attribute = attribute
        self.positions = positions
