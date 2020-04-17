import heapq


class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.len = 0

    def push(self, x: int) -> None:
        self.stack.append(x)
        self.len += 1

    def pop(self) -> None:
        self.len -= 1
        return self.stack.pop()

    def top(self) -> int:
        if self.len > 0:
            return self.stack[self.len - 1]
        else:
            pass

    def getMin(self) -> int:
        cpy = self.stack.copy()
        cpy.sort()
        return cpy[0]
# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
