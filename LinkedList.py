class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def appendToTail(node, d):
    new_end = ListNode(d)
    n = node
    while n.next is not None:
        n = n.next

    n.next = new_end


def linkedListToString(node):
    print('\n')
    while node is not None:
        print(node.val, end=' -> ')
        node = node.next

def mergeTwoSortedLists(l1, l2):
    # Store a reference to the head of list (to return)
    prehead = ListNode(-1)
    # Keep a pointer to the node we are considering changing it's next node
    prev = prehead
    # Iterate through each element of l1 and l2
    while l1 and l2:
        # Check for the minimum value between the two current nodes
        if l1.val <= l2.val:
            # Point the next to l1
            prev.next = l1
            # Move l1 over
            l1 = l1.next
        else:
            # Point the next to l2
            prev.next = l2
            # Move l2 over
            l2 = l2.next
        # Move prev over
        prev = prev.next
    # At most one of l1 and l2 can be non-null here.
    # Connect the non-null list to the end of the merged list
    prev.next = l1 or l2
    return prehead.next
