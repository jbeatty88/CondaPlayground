from LinkedList import ListNode, linkedListToString, appendToTail


def test_appendTo_tail():
    a = ListNode(5, ListNode(10, ListNode(2)))
    linkedListToString(a)
    appendToTail(a, 100)
    linkedListToString(a)
