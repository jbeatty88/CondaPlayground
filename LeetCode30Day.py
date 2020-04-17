import collections
import math
from typing import List


class LeetCode:
    # Definition for singly-linked list.
    class ListNode:
        def __init__(self, x):
            self.val = x
            self.next = None

    # Definition for a binary tree node.
    class TreeNode:
        def __init__(self, x):
            self.val = x
            self.left = None
            self.right = None

    def maxSubArray(self, nums: List[int]) -> int:
        l = [0] * len(nums)  # List of contiquous subseq
        m = {}  # Dictionary to hold sums and subseq
        s = []

        if len(nums) == 1:
            return nums[0]

        l[0] = nums[0]
        s.append(nums[0])
        for i in range(1, len(nums)):
            l[i] = max(l[i - 1] + nums[i], nums[i])
            if l[i] == nums[i]:
                m[l[i - 1]] = s
                s.clear()
            s.append(nums[i])
            if i == len(nums) - 1:
                m[l[i - 1]] = s

        return max(l)

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams = {}
        for word in strs:
            chars = list(word)
            chars.sort()
            key = ''.join(chars)
            if key not in anagrams:
                anagrams[key] = []
            anagrams[key].append(word)

        result = []
        for values in anagrams.values():
            result.append(values)
        return result

    def countElements(self, arr: List[int]) -> int:
        hs = set(arr)
        count = 0
        for n in arr:
            seek = n + 1
            instances = 0 if arr.count(seek) < 1 else 1
            count += instances
        return count

    def middleNode(self, head: ListNode) -> ListNode:
        n_nodes = 1
        root_node = head

        while head.next is not None:
            n_nodes = n_nodes + 1
            head = head.next

        print(n_nodes)

        middle = math.floor(n_nodes / 2)
        if middle % 2 is not 0:
            middle = middle

        print(middle)
        while middle > 0:
            root_node = root_node.next
            middle = middle - 1

        return root_node

    def backspaceCompare(self, S: str, T: str) -> bool:
        # Break up the strings into chars
        s_chars = list(S)
        t_chars = list(T)

        while '#' in s_chars:
            idx = s_chars.index('#')
            if idx > 0:
                s_chars.pop(idx - 1)
            s_chars.remove('#')

        while '#' in t_chars:
            idx = t_chars.index('#')
            if idx > 0:
                t_chars.pop(idx - 1)
            t_chars.remove('#')

        return True if t_chars == s_chars else False

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        def recursive_depth(node: LeetCode.TreeNode) -> int:
            if node is None:
                return 0
            return 1 + max(recursive_depth(node.left), recursive_depth(node.right))

        try:
            # Start here at the root
            cur_node = root
            left_depth = recursive_depth(cur_node.left)
            right_depth = recursive_depth(cur_node.right)

            left_diameter = self.diameterOfBinaryTree(cur_node.left)
            right_diameter = self.diameterOfBinaryTree(cur_node.right)

            longest_path = max(left_depth + right_depth + 1, max(left_diameter, right_diameter))

        except:
            longest_path = 0

        return longest_path

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        sol = []
        for idx, num in enumerate(nums):
            for idx2 in range(idx + 1, len(nums)):
                if nums[idx] + nums[idx2] == target:
                    return [idx, idx2]
        return sol

    def tribonacci(self, n: int) -> int:
        trib = [0] * 100
        trib[0] = 0
        trib[1] = 1
        trib[2] = 1
        for i in range(n):
            trib[i + 3] = trib[i] + trib[i + 1] + trib[i + 2]

        return trib[n]

    def gardenNoAdj(self, N: int, paths: List[List[int]]) -> List[int]:
        answer = [0] * (N + 1)
        flowerTypes = {1, 2, 3, 4}
        neighbors = collections.defaultdict(list)

        for idx in paths:
            neighbors[idx[0]].append(idx[1])
            neighbors[idx[1]].append(idx[0])

        for idx in range(1, N + 1):
            answer[idx] = (flowerTypes - {answer[j] for j in neighbors[idx]}).pop()
        return answer[1:]

    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        def get_actual_k(ppl, pos, h):
            pos_count = 0
            k_count = 0
            for p in ppl:
                if pos_count >= pos:
                    break
                else:
                    if p[0] >= h:
                        k_count += 1
                    pos_count += 1
            return k_count

        # Sort by K
        people.sort(key=lambda x: x[1])
        # Check each individual person against the queue
        for pos, person in enumerate(people):
            h = person[0]
            desired_k = person[1]
            # Get the actual k
            actual_k = get_actual_k(people, pos, h)
            # Move person forward actual k spots
            if actual_k is not desired_k:
                new_pos = pos - (actual_k - desired_k)
                # new_pos = pos - offset
                people.pop(pos)
                people.insert(new_pos, person)

        return people

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # Set so that the elements are unique
        solutions = []
        partial_solutions = []

        for n in candidates:
            if n < target:
                partial_solutions.append([n])
            elif n == target:
                solutions.append([n])

        while len(partial_solutions) > 0:
            cur_ps = partial_solutions.pop(0)
            for n in candidates:
                tmp = cur_ps.copy()
                if sum(tmp) + n < target:
                    tmp.append(n)
                    partial_solutions.append(tmp)
                elif sum(tmp) + n == target:
                    tmp.append(n)
                    tmp.sort()
                    solutions.append(tmp)

        # print(partial_solutions)
        # print(solutions)

        unique_sol = [list(x) for x in set(tuple(x) for x in solutions)]
        return list(unique_sol)

    def lastStoneWeight(self, stones: List[int]) -> int:
        while len(stones) > 1:
            y = max(stones)
            stones.remove(y)
            x = max(stones)
            stones.remove(x)
            smash = y - x
            if smash > 0:
                stones.append(smash)
        try:
            return stones[0]
        except:
            return 0

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # Copy of our triangle to store minimum paths
        triangle_cpy = triangle.copy()
        # Start from the level above base of triangle
        for lvl in range(len(triangle) - 2, -1, -1):
            # Find the minimum path from each position with the level below
            for pos in range(lvl + 1):
                nxt_lvl = triangle[lvl + 1]
                # Find min path from ea. pos down to next level and next level adjacent
                triangle_cpy[lvl][pos] += min(nxt_lvl[pos], nxt_lvl[pos + 1])
        # Store result at top of the triangle
        return triangle_cpy[0][0]

    def maxArea(self, height: List[int]) -> int:
        # Initially, find the highest container wall with the lowest idx.
        max_v_bssf = 0
        farthest_left = height[0]
        farthest_right = height[len(height) - 1]
        farthest_left_idx = height.index(farthest_left)
        farthest_right_idx = height.index(farthest_right)
        d = abs(farthest_left_idx - farthest_right_idx)
        h = min(farthest_left, farthest_right)
        max_v_bssf = d * h

        # BRUTE APPROACH
        for idx1, wall1 in enumerate(height):
            for idx2, wall2 in enumerate(height):
                # Get the distance to the next wall
                d = abs(idx2 - idx1)
                h = min(wall1, wall2)

                max_v_bssf = d * h if d * h > max_v_bssf else max_v_bssf

        return max_v_bssf

    def findMaxLength(self, nums: List[int]) -> int:
        max_len = 0
        hash_table = {}
        curr_sum = 0

        for i in range(len(nums)):
            if nums[i] == 0:
                nums[i] = -1
            else:
                nums[i] = 1

        for i in range(len(nums)):

            curr_sum = curr_sum + nums[i]

            if curr_sum == 0:
                max_len = i + 1

                # If this sum is seen before,
            if curr_sum in hash_table:
                if max_len < i - hash_table[curr_sum]:
                    max_len = i - hash_table[curr_sum]
            else:
                hash_table[curr_sum] = i

        for i in range(len(nums)):
            if nums[i] == -1:
                nums[i] = 0
            else:
                nums[i] = 1

        return max_len

    def checkValidString(self, s: str) -> bool:
        l_count = r_count = 0
        n = len(s)
        for i in range(n):
            if s[i] in "(*":
                l_count += 1
            else:
                l_count -= 1
            if s[n - i - 1] in "*)":
                r_count += 1
            else:
                r_count -= 1
            if l_count < 0 or r_count < 0:
                return False
        return True

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        product = [1] * len(nums)

        l = 1
        r = 1

        for i in range(len(nums)):
            product[i] *= l
            product[~i] *= r
            l *= nums[i]
            r *= nums[~i]

        return product

    def stringShift(self, s: str, shift: List[List[int]]) -> str:
        partial_sol = collections.deque(s)
        for w in shift:
            if w[0] == 0:
                t = 0
                while t < w[1]:
                    partial_sol.append(partial_sol.popleft())
                    t += 1
            else:
                t = 0
                while t < w[1]:
                    partial_sol.appendleft(partial_sol.pop())
                    t += 1
        return ''.join(partial_sol)
