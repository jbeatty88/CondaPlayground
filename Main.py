from ByuCompetetiveProgramming import BC
from GoogleCodeJam2019 import CodeJam
from LeetCode import LeetCode, DfsTreeNode, TreeNode
from AdventOfCode2020 import AdventOfCode2020
from LinkedList import ListNode

if __name__ == '__main__':
    codeJam = CodeJam()
    lc = LeetCode()
    bc = BC()
    aoc = AdventOfCode2020()

    # print(aoc.p2_1PasswordPhilosophy())
    # print(lc.findDiagonalOrder([[1,2,3],[4,5,6],[7,8,9]]))
    l1 = ListNode(1, ListNode(2, ListNode(4)))
    l2 = ListNode(1, ListNode(3, ListNode(4)))
    print(lc.mergeTwoLists(l1, l2))