from __future__ import annotations
from bisect import bisect

from dataclasses import dataclass

@dataclass
class Node:
    coord: tuple[int,int] | tuple[float,float]
    real_cost: float
    heuristic: float
    depth: int
    parent: Node|None
    neighbors: list[tuple[int,int]] | None

# https://stackoverflow.com/a/39501468
class NodeListWrapper():
    """Wraps Node for bisect comparison given key"""
    def __init__(self,it:list,key):
        self._it = it
        self._key = key
    
    def __getitem__(self,idx):
        return self._key(self._it[idx])
    
    def __len__(self):
        return self._it.__len__()
    
    def insert(self,idx,__n:Node):
        self._it.insert(idx,__n)

class OpenList:
    def __init__(self):
        self._list = [] # Ordered list
        self._dict = {} # Index dict
    
    def pop(self) -> Node:
        n = self._list.pop(0)
        self._dict = {k:v-1 for k,v in self._dict.items() if k != n.coord}
        return n

    def insert(self, __n: Node) -> None:
        key = lambda n: n.real_cost + n.heuristic
        
        if __n.coord in self._dict:
            old_idx = self._dict[__n.coord]
            old_node = self._list[old_idx]
            if __n.real_cost < old_node.real_cost:  # Replace if needed
                self._list.pop(old_idx)
                del self._dict[__n.coord]
                self._dict = {k:v if v < old_idx else v-1 for k,v in self._dict.items()}
            else:
                return

        idx = bisect(NodeListWrapper(self._list, key), key(__n))
        self._list.insert(idx, __n)
        self._dict = {k:v if v < idx else v+1 for k,v in self._dict.items()}
        self._dict[__n.coord] = idx

    def __contains__(self, __n: Node) -> bool:
        return __n.coord in self._dict
    
    def __len__(self) -> int:
        return self._list.__len__()

    def __bool__(self) -> bool:
        return self._list.__len__() != 0

class VisitedList(set):
    """Wraps set to allow lists"""
    def add(self,__o:list|tuple):
        if isinstance(__o,list):
            __o = tuple(__o)
        super(VisitedList,self).add(__o)

    def __contains__(self, __o: object) -> bool:
        if isinstance(__o,list):
            __o = tuple(__o)
        return super().__contains__(__o)
