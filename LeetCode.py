import collections
import copy
import heapq
import itertools
import math
import time
from typing import List

# Definition for singly-linked list.
from pyparsing import unicode


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class DfsTreeNode():
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class PathNode:
    def __init__(self, x):
        self.cost = x
        self.path = [[0, 0]]

    def __lt__(self, other):
        return len(self.path) > len(other.path)


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """


class LeetCode:

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
        if middle % 2 != 0:
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
        """1 Two Sum

        Given an array of integers nums and an integer target, return indices of the two numbers
        such that they add up to target.

        You may assume that each input would have exactly one solution. You may not use the same
        element twice.

        You can return the answer in any order

        Args:
            nums: (List[int]) Array of integers
            target: (int) Sum target

        Returns: (List[int]) List of numbers that sum to target

        """
        dict_map = {}
        for i, num in enumerate(nums):
            if num in dict_map:
                return [dict_map[num], i]
            dict_map[target - num] = i

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
        """39 Combination Sum

        Time Complexity: O(N^((T/M)+1))
            - Total number of steps = number of nodes in the tree
            - Linear to the number of nodes of the execution tree

        Space Complexity: O(T/M)

        Given an array of distinct integers, candidates, and a target integer, target, return
        a list of all unique combinations of 'candidates' where the chosen numbers sum to target.
        You may return the combinations in any order.

        The same number may be chosen from candidates an unlimited number of times. Two combinations
        are unique if the frequency of at least one of the chosen numbers is different.

        It is guaranteed that the number of unique combinations that sum up to target is less than 150
        combinations for the given input

        Args:
            candidates: (List[int]) Distinct Integer List
            target: (int) target number

        Returns: (List[int]) list of all unique combinations of candidates

        """
        # Set so that the elements are unique
        results = []

        def backtrack(remain, comb, start):
            # Base Case: If we've reached the target sum
            if remain == 0:
                # Add this combination to the list
                results.append(list(comb))
            # Base Case: If we've exceeded our target sum
            if remain < 0:
                # Stop exploring this branch
                return

            # Go through each combination branch
            for i in range(start, len(candidates)):
                # Add the number into the combination here
                comb.append(candidates[i])
                # Give the current number another chance
                backtrack(remain - candidates[i], comb, i)
                # Backtrack, remove the number from the combination
                comb.pop()

        backtrack(target, [], 0)
        return results

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
        ## Brute Force -> Fails Time Requirements  ##
        # # Keep track of the max so far
        # max_so_far = 0
        # # Iterate through every combination of pillars
        # for i, n in enumerate(height):
        #     for j in range(1, len(height)):
        #         # Compare the new area with the max so far and replace if bigger
        #         max_so_far = max(max_so_far, min(n, height[j]) * (j - i))
        # # Return the biggest area we found
        # return max_so_far

        ## Two-Pointer ##
        # Keep track of the best solution so far
        max_area_so_far = 0
        # Start with the farthest left pillar
        left = 0
        # Start with the farthest right pillar
        right = len(height) - 1
        # Run until the pillars are side by side
        while left < right:
            # Check the area between the left and right pillar and store the higher
            max_area_so_far = max(max_area_so_far, min(height[left], height[right]) * (right - left))
            # Move over left or right depending on which pillar is currently taller
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        # Return the max area
        return max_area_so_far

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

    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(grid, row, col):
            if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]) or grid[row][col] != '1':
                return
            grid[row][col] = '#'
            dfs(grid, row + 1, col)
            dfs(grid, row - 1, col)
            dfs(grid, row, col + 1)
            dfs(grid, row, col - 1)

        c = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == '1':
                    dfs(grid, row, col)
                    c += 1
        return c

    def minPathSum(self, grid: List[List[int]]) -> int:
        start = time.time()

        def get_bssf():
            ans = 0
            for row in grid:
                if row == grid[0]:
                    for n in row:
                        ans += n
                else:
                    ans += row[len(row) - 1]
            return ans

        bssf = get_bssf()
        sol_queue = []
        heapq.heapify(sol_queue)
        start_node = LeetCode.PathNode(grid[0][0])
        # sol_queue.append(start_node)
        heapq.heappush(sol_queue, start_node)

        while len(sol_queue) > 0 and time.time() - start < 30:
            node = heapq.heappop(sol_queue)
            # node = sol_queue.pop()
            if node.cost < bssf:
                cur_row = node.path[-1][0]
                cur_col = node.path[-1][1]
                if cur_row is not len(grid) - 1 or cur_col is not len(grid[0]) - 1:
                    next_horiz_idx = cur_col + 1 if cur_col + 1 < len(grid[0]) else None
                    next_vert_idx = cur_row + 1 if cur_row + 1 < len(grid) else None
                    if next_horiz_idx is not None:
                        if node.cost + grid[cur_row][next_horiz_idx] < bssf:
                            ps = LeetCode.PathNode(node.cost + grid[cur_row][next_horiz_idx])
                            ps.path = node.path.copy()
                            ps.path.append([cur_row, next_horiz_idx])
                            heapq.heappush(sol_queue, ps)
                            # sol_queue.append(ps)
                    if next_vert_idx is not None:
                        if node.cost + grid[next_vert_idx][cur_col] < bssf:
                            ps = LeetCode.PathNode(node.cost + grid[next_vert_idx][cur_col])
                            ps.path = node.path.copy()
                            ps.path.append([next_vert_idx, cur_col])
                            heapq.heappush(sol_queue, ps)
                            # sol_queue.append(ps)
                else:
                    if node.cost < bssf:
                        bssf = node.cost

        return bssf

    def numJewelsInStones(self, J: str, S: str) -> int:
        # Keep a counter for our jewels
        jewel_count = 0
        stones = list(S)
        for jewel in list(J):
            jewel_count += stones.count(jewel)

        return jewel_count

    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        ransom_char_list = list(ransomNote)
        char_list = list(magazine)
        while len(ransom_char_list) > 0:
            c = ransom_char_list.pop()
            if char_list.__contains__(c):
                char_list.remove(c)
            else:
                return False
        return True

    def bitwiseComplement(self, N: int) -> int:
        binary = "{0:b}".format(N)
        binaryList = list(binary)
        binaryComplement = ""
        for i, d in enumerate(binaryList):
            binaryList[i] = '0' if d == '1' else '1'

        return int(binaryComplement.join(binaryList), 2)

    def firstUniqChar(self, s: str) -> int:
        counter = collections.Counter(s)
        try:
            uniqList = list(counter.keys())[list(counter.values()).index(1)]
            return s.index(uniqList[0])
        except:
            return -1

    def majorityElement(self, nums: List[int]) -> int:
        num = collections.Counter(nums).most_common(1)
        return num[0][0] if num[0][1] > len(nums) // 2 else -1

    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        slope = None

        def getSlope(p1: List[int], p2: List[int]) -> float:
            try:
                return (p2[0] - p1[0]) / (p2[1] - p1[1])
            except:
                return 0

        for pIdx, p1 in enumerate(coordinates):
            # Calculate slope with next
            if pIdx < len(coordinates) - 1:
                s = getSlope(p1, coordinates[pIdx + 1])
                if slope is None:
                    slope = s
                elif slope != s:
                    return False

        return True

    def isPerfectSquare(self, num: int) -> bool:
        # The identity is 1+3+5...(2n-1)
        i = 1  # We first subtract 1 from the number
        # Then 3, 5, 7, 8...etc until num is 0 if perfect square or not if negative
        while num > 0:
            num -= i
            i += 2
        return num == 0

    def findJudge(self, N: int, trust: List[List[int]]) -> int:
        relations = {}
        people = list(range(1, N + 1))
        for r in trust:
            if r[0] not in relations:
                people.remove(r[0])
                relations[r[0]] = [r[1]]
            else:
                relations.get(r[0]).append(r[1])

        if len(people) == 0:
            return -1
        else:
            for p in people:
                for _, val in relations.items():
                    if val.__contains__(p):
                        pass
                    else:
                        return -1
        return people[0]

    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        def paintPixel(image: List[List[int]], startingPixel: int, curRow: int, curCol: int, newColor: int):
            fourDirections = [[0, 1], [0, -1], [-1, 0], [1, 0]]
            # Set the pixel to the newColor
            image[curRow][curCol] = newColor
            for nextDir in fourDirections:
                nextRow = curRow + nextDir[0]
                nextCol = curCol + nextDir[1]
                if nextRow >= 0 and nextRow < len(image) and nextCol >= 0 and nextCol < len(image[0]) and \
                        image[nextRow][nextCol] == startingPixel:
                    paintPixel(image, startingPixel, nextRow, nextCol, newColor)

        if image[sr][sc] != newColor:
            paintPixel(image, image[sr][sc], sr, sc, newColor)
        return image

    def singleNonDuplicate(self, nums: List[int]) -> int:
        counter = collections.Counter(nums)
        # Sort the counter by frequency ** USEFUL **
        freqSort = sorted(counter.items(), key=lambda i: (i[1], -i[0]))
        return freqSort[0][0]

    def largestTimeFromDigits(self, A: List[int]) -> str:
        # Bounds for 24 hour format
        hourBound = 24
        minuteBound = 60
        # So we can get to the tens digit
        formatOffset = 10
        # Start with some problem P0
        P0 = A.copy()
        # Sort and reverse so we have the larger permutations at the front
        # Use this to replace our queue (S) for backtracking
        # First solution is the actual solution
        P0.sort()
        P0.reverse()
        # Break P0 into subproblems Pi (All the permutations)
        sortedPerm = list(itertools.permutations(P0))
        # Test each subprobelm Pi
        for p in sortedPerm:
            # Unpack each of the values
            h0, h1, m0, m1 = p
            # Calculate the hour value
            hour = h0 * formatOffset + h1
            # Calculate the minute value
            minute = m0 * formatOffset + m1
            # If our bounds are satisfied, we have a solution
            if hour < hourBound and minute < minuteBound:
                return "{}{}:{}{}".format(h0, h1, m0, m1)
        # If no solution, return empty string
        return ""

    def mergeTwoLists(self, l1: ListNode, l2: ListNode):
        while l2.next is not None:
            if l2.val > l1.val:
                l1 = l1.next
            else:
                prevFirstNode = l1
                l1 = ListNode(l2.val, prevFirstNode.next)
                l2 = l2.next

    def rotateString(self, A: str, B: str) -> bool:
        """796 Rotate String

        We are given two strings, A and B

        A shift on A consists of taking string A and moving the leftmost
        character to the rightmost position.

        Args:
            A: (str)
            B: (str)

        Returns: (bool) true or false

        """
        # Concatenate A with itself and then search for B
        return len(A) == len(B) and B in (A + A)

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        pass

    def repeatedSubstringPattern(self, s: str) -> bool:
        # Duplicate and concat the given string
        dupStr = s + s
        # Remove the first and last characters (first: first of substr, last: last of substr)
        dubStrTrim = dupStr[1: -1]
        # Search for the given string in our concatenated/trimmed string
        return s in dubStrTrim

    def removeDuplicatesInSortedArray(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        idx = 0
        for n in nums:
            if n != nums[idx]:
                idx += 1
                nums[idx] = n

        return idx + 1

    def maxProfit(self, prices: List[int]) -> int:
        pass

    def lengthOfLastWord(self, s: str) -> int:
        pass

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """56 Merge Intervals

        Time Complexity: O(nlogn)
            O(n) -> Linear scan of list after sort
            O(nlogn) -> Sort

        Args:
            intervals: array of intervals

        Returns: array of non-overlapping intervals

        """
        # Sort the intervals
        intervals.sort(key=lambda interval: interval[0])
        # Store the merged intervals
        sol = []
        for interval in intervals:
            # If there are no intervals already added
            # Or if the end of the last interval is less than the beginning of the current
            # Append it to the list because there is no overlap
            if not sol or sol[-1][1] < interval[0]:
                sol.append(interval)
            else:
                # Replace the last number in the interval with the max between existing and new
                sol[-1][1] = max(sol[-1][1], interval[1])
        return sol

    def findMaximumXOR(self, nums: List[int]) -> int:
        pass

    def numberOfLines(self, widths: List[int], S: str) -> List[int]:
        print(widths)
        print(S)
        lines = 1
        chars_in_line = 0
        for letter in S:
            # Get the width index for that letter
            letter_idx = abs(ord(letter) - 97)
            # Make sure adding that width to the line doesn't go over boundary
            if chars_in_line + widths[letter_idx] > 100:
                # Store the leftover after adding to line
                # left_over = chars_in_line + widths[letter_idx] - 100
                # Increment lines
                lines += 1
                # Reset the line char count
                chars_in_line = 0
                # Add the leftover to the new line
                chars_in_line += widths[letter_idx]
            else:
                # Add the width to the new line
                chars_in_line += widths[letter_idx]
        return [lines, chars_in_line]

    def placeNQueens(self, n, board):
        def is_safe(i, j, b):
            for c in range(len(b)):
                for r in range(len(b)):
                    # check if i,j share row with any queen
                    if b[c][r] == 'q' and i == c and j != r:
                        return False
                    # check if i,j share column with any queen
                    elif b[c][r] == 'q' and j == r and i != c:
                        return False
                    # check if i,j share diagonal with any queen
                    elif (i + j == c + r or i - j == c - r) and b[c][r] == 'q':
                        return False
            return True

        def n_queens(r, num, b):
            # base case, when queens have been placed in all rows return
            if r == num:
                return True, b
            # else in r-th row, check for every box whether it is suitable to place queen
            for i in range(num):
                if is_safe(r, i, b):
                    # if i-th columns is safe to place queen, place the queen there and check recursively for other rows
                    b[r][i] = 'q'
                    okay, newboard = n_queens(r + 1, num, b)
                    # if all next queens were placed correctly, recursive call should return true, and we should return true here too
                    if okay:
                        return True, newboard
                    # else this is not a suitable box to place queen, and we should check for next box
                    b[r][i] = '-'
            return False, b

        # USE THIS IN MAIN WHEN RUNNING TO SEE RESULTS
        # n = 4
        # board = [["-" for _ in range(n)] for _ in range(n)]
        # qBoard = lc.placeNQueens(n, board)
        # qBoard = "\n".join(["".join(x) for x in qBoard])
        # print(qBoard)
        return n_queens(0, n, board)[1]

    # Global dictionary to allow memoization (storing results)
    fib_mem = {}

    # ^^^^^^^^^^^
    def fib_with_memoization(self, n):
        """Calculate Fibonacci using memoization

        Complexity:
            T: O(n) -> To evaluate fib(n) we need the results of fib(n-1) and fib(n-2)
                but fib(n-2) would have already been evaluated from the call to fib(n-1)'s
                recursive call. Thus it's value would already be available and would not have
                to be recomputed. Getting the value from the hashtable is O(1)

        Args:
            n: The Fibonacci number we want to get

        Returns: The nth Fibonacci number

        """

        if n == 0:  # Base case 1
            return 0
        if n == 1:  # Base case 2
            return 1
        elif n in self.fib_mem:  # If we've already computed a value, don't recompute.
            return self.fib_mem[n]  # Get the value stored previously
        else:  # If we haven't computed this value yet, compute and store for later uses
            self.fib_mem[n] = self.fib_with_memoization(n - 1) + self.fib_with_memoization(n - 2)
            return self.fib_mem[n]

    def nth_stair(self, n, m, memo):
        if n == 0:  # base case of when there is no stair
            return 1
        if n in memo:  # before recursive step check if result is memoized
            return memo[n]
        ways = 0
        for i in range(1, m + 1):  # iterate over number of steps, we can take
            # if steps remaining is smaller than the jump step, skip
            if i <= n:
                # recursive call with n i units lesser where i is the number of steps taken here
                ways += self.nth_stair(n - i, m, memo)
                # memoize result before returning
        memo[n] = ways
        return ways

    def staircase(self, n, m):
        memo = {}
        # helper function to add memo dictionary to function
        return self.nth_stair(n, m, memo)

    def shuffle(self, nums: List[int], n: int) -> List[int]:
        ''' 1470 Shuffle The Array
        Given the array 'nums' consisting of '2n' elements in the form [x1, x2, ..., xn, y1, y2, ..., yn]
        return the array in the form [x1, y1, x2, y2, ..., xn, yn]

        Args:
            nums: array of 2n elements
            n: Number of x and y elements in array

        Example:
            Input: nums = [2,5,1,3,4,7], n = 3
            Output: [2,3,5,4,1,7]

        Returns: the array in the form [x1, y1, x2, y2, ..., xn, yn]
        '''
        ## BRUTE FORCE ##
        solution = []
        # x_elem = nums[:n]
        # y_elem = nums[n:]
        # for i in range(n):
        #     solution.append(x_elem[i])
        #     solution.append(y_elem[i])

        ## FASTEST RUNTIME ##
        for i in range(n):
            solution.append(nums[i])
            solution.append(nums[i + n])

        return solution

    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        '''
        Given the array candies and the integer extraCandies, where candies[i]
        represents the number of candies that the ith kid has.

        For each kid check if there is a way to distribute extraCandies among the
        kids such that he or she can have the greatest number of candies among them.
        Notice that multiple kids can have the greatest number of candies.

        Args:
            candies: array, holds number of candies ith kid has
            extraCandies: int, number of candies that can be distributed

        Returns: boolean array, true: kid has greatest; falst: kid does not have greatest
        '''

        ## BRUTE FORCE ##
        # arr_cpy = candies.copy()
        # arr_cpy.sort()
        # max_e = arr_cpy[len(arr_cpy) - 1]
        # for i, k in enumerate(candies):
        #     if max_e - candies[i] <= extraCandies:
        #         candies[i] = True
        #     else:
        #         candies[i] = False
        ## FASTEST SOLUTION ##
        m = max(candies)
        return [i + extraCandies >= m for i in candies]

    def defangIPaddr(self, address: str) -> str:
        ''' 1108 - Defanging an IP Address
        Given a valid (IPv4) IP address, return a defanged version of that IP address.

        A defanged IP address replaces every period "." with "[.]"

        Args:
            address: str; ip address to defang

        Returns: str; defanged IP address
        '''
        ## MY SOLUTION
        # return address.replace(".", "[.]")

        ## FASTEST SOLUTION ##
        x = ""
        for i in address:
            if i == ".":
                x = x + "[.]"
            else:
                x = x + i
        return x

    def numIdenticalPairs(self, nums: List[int]) -> int:
        ''' 1512 - Number of Good Pairs

        Given an array of integers 'nums'

        A pair (i, j) is called good if nums[i] == nums[j] and i < j

        return the number of good pairs

        Args:
            nums: List[int]; array of numbers

        Returns: (int) number of pairs
        '''
        ## MY SOLUTION ##
        pairs = 0
        nums_set = set(nums)
        for n in nums_set:
            pairs += nums.count(n) // 2
        return pairs

    def maskPII(self, S: str) -> str:
        # Check if we have an email or number
        if S.__contains__("@"):
            # Convert to lower case
            S = S.lower()
            # Get first and last letters of first name
            n1, leftover = S.split('@')
            # 5 starts in between
            mask = "{}*****{}@{}".format(n1[0], n1[-1], leftover)
            return mask
        else:
            # Filter out everything but digits
            nums = ''.join(n for n in S if n.isdigit())
            # Create the general mask
            mask = "***-***-{}".format(nums[-4:])
            # Add to the mask if we have a country code
            if len(nums) > 10:
                return "+{}-".format('*' * (len(nums) - 10)) + mask
            # Otherwise, return general mask
            return mask

    def lengthOfLongestSubstring(self, s: str) -> int:
        ''' 3. Longest Substring Without Repeating Characters

        Given a string s, find the length of the longest substring without repeating characters.

        Args:
            s: (str) full string

        Returns: (int) length of largest substring
        '''
        pass

    def dietPlanPerformance(self, calories: List[int], k: int, lower: int, upper: int) -> int:
        ''' 1176 Diet Plan Performance

        Dieter consumes calories[i] calories on the i-th day.

        Given an integer k, for every consecutive sequence of k days, they look at T, the total calories
        consumed during that seqence of k days.
            If T < lower, they performed poorly on their diet and lose 1 point
            if T > upper, they performed well on their diet and gain 1 point
            Otherwise, they performed normally and there is no change in points

        Initially the dieter has 0 points. Return the total number of points the diter has after dieting
        for calories.length days.

        Args:
            calories: (int arr) calories consumed on the i-th day
            k: (int) sequence of days
            lower: (int) bound for poor diet
            upper: (int) bound for good diet

        Returns: (int) total number of points the dieter has after dieting for calories.length days
        '''
        diet_score, T = 0, sum(calories[:k - 1])
        for i in range(k - 1, len(calories)):
            T += calories[i] - (calories[i - k] if i - k >= 0 else 0)
            diet_score += T > upper
            diet_score -= T < lower

        return diet_score

    def numKLenSubstrNoRepeats(self, S: str, K: int) -> int:
        ''' 1100 Find K-Length Substrings With No Repreated Characters

        Given a string, S, retur the number of substrings of length K with no repreated characters

        Args:
            S: (str) string
            K: (int) length of substring

        Returns:(int) number of substrings with no repeated characters
        '''

    def matrixBlockSum(self, mat: List[List[int]], K: int) -> List[List[int]]:
        ''' 1314 Matrix Block Sum

        Given a m * n matrix, mat, and an integer, k, return a matrix, answer,
        where each answer[i][j] is the sum of all elements mat[r][c] for
        i - k <= r <= i + K, j-k <= x <= j+K, and (r,c) is a valid position in
        the matrix.

        Args:
            mat: (int[][]) m * n matrix
            K: (int) radius to sum

        Returns: (int[][]) m * n matrix with correct sums in each position

        '''
        m, n = len(mat), len(mat[0])
        rangeSum = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                rangeSum[i + 1][j + 1] = rangeSum[i + 1][j] + rangeSum[i][j + 1] - rangeSum[i][j] + mat[i][j]
        ans = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                r1, c1, r2, c2 = max(0, i - K), max(0, j - K), min(m, i + K + 1), min(n, j + K + 1)
                ans[i][j] = rangeSum[r2][c2] - rangeSum[r1][c2] - rangeSum[r2][c1] + rangeSum[r1][c1]
        return ans

    def minFallingPathSum(self, A: List[List[int]]) -> int:
        ''' 931 Minimum Falling Path Sum

        Given a square array of integers, A, we want the minimum sum of a falling path
        through A. A Falling path starts at any element in the first row, and chooses
        one element from each row. The next row's choice must be in a column that is
        different from the previous rows column by at most one.

        Args:
            A: (List(int[])) array of integers

        Returns: (int) Minimum sum of a falling path

        '''
        while len(A) >= 2:
            r = A.pop()
            for i in range(len(r)):
                A[-1][i] += min(r[max(0, i - 1): min(len(r), i + 2)])
        return min(A[0])

    def knightProbability(self, N: int, K: int, r: int, c: int) -> float:
        ''' 688 Knight Probability In Chessboard

        On an N*N chessboard, a knight starts at the r-th row and c-th column
        and attempts to make exactly K moves. The rows and columns are 0 indexed
        so the top-left square is (0,0) and the bottom right is (N-1, N-1)

        A chess knight has 8 possible moves it can make. Each move is two squares
        in a cardinal direction, then one square in an orthogonal direction

        Each time the knight is to move, it chooses one of 8 possible moves uniformly
        (even if that move would take it off the board) and moves there.

        The knight continues moving until it has made exactly K moves or has
        moved off of the board

        Return the probability that the knight remains on the board after it has
        stopped moving

        Args:
            N: (int) Dimension for square N*N board
            K: (int) Moves the knight makes
            r: (int) Starting row
            c: (int) Starting column

        Returns: (float) Probability the knight remains on board after it stops moving

        '''
        pass

    def maxDistance(self, grid: List[List[int]]) -> int:
        ''' 1162 As Far From Land As Possible

        Given an n*n grid containing only values 0 and 1, where 0 represents water
        and 1 represents land, find a water cell such that its distance to the nearest
        land cell is maximized, and return the distance. If no land or water exists in
        the grid, return -1

        Use Manhattan distance:
            Distance between two cells (x0, y0) and (x1, y1) is |x0-x1| + |y0-y1|

        Args:
            grid: (List(int[])) n*n grid

        Returns: (int) Maximum distance between a water cell and land cell

        '''
        pass

    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        '''1631 Path With Minimum Effort

        You are a hiker preparing for an upcoming hike. You are given 'heights', a 2D array
        of size rows * columns where heights[row][col] represents the height of cell (row, col).
        You are situated in the top-left cell (0,0) and you hope to travel to the bottom-right cell, (rows-1, columns-1)
        You can move up, down, left, or right and you wish to find a route that requires the
        minimum effort.



        Args:
            heights:

        Returns:

        '''
        pass

    def containsCycle(self, grid: List[List[str]]) -> bool:
        ''' 1559 Detect Cycles in 2D Grid

        Args:
            grid:

        Returns:

        '''
        pass

    def shortestDistance(self, words: List[str], word1: str, word2: str) -> int:
        pass

    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left) + 1, self.maxDepth(root.right) + 1)

    def shortestDistance(self, words: List[str], word1: str, word2: str) -> int:
        '''

        Given a list of words and two words word1 and word2, return the shortest
        distance between these two words in the list.
        Args:
            words:
            word1:
            word2:

        Returns:

        '''
        pass

    def rotate(self, matrix: List[List[int]]) -> None:
        """48 Rotate Image

        Time Complexity: O(N^2)
            - Nested loop

        Space Complexity: O(1)
            - In place rotation

        You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise)

        You have to rotate the image in-place, which means you have to modify the input 2D matrix directly.
        DO NOT allocate another 2D matrix and do the rotation.

        Args:
            matrix: n*n 2D matrix

        Returns: nothing, just modify the matrix in-place instead

        """
        # Get the dimension of the n*n matrix
        n = len(matrix[0])
        for i in range(n // 2 + n % 2):
            for j in range(n // 2):
                # Rotate four rectangles working from outside inward
                tmp = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 - i]
                matrix[j][n - 1 - i] = matrix[i][j]
                matrix[i][j] = tmp
        print(matrix)

    def findMin(self, nums: List[int]) -> int:
        """153 Find Minimum in Rotated Sorted Array

        Time Complexity: O(logN)
            - Binary Search: O(logN)

        Space Complexity: O(1)

        Suppose an array of length n sorted in ascending order is rotated between 1 and n times.

        Args:
            nums: (List[int]) Sorted rotated array

        Returns: (int) Minimum element of nums

        """
        # IF only one element in nums, return that element
        if len(nums) == 1:
            return nums[0]

        # Point to left and right side
        left, right = 0, len(nums) - 1

        # If last > first, no rotations took place
        if nums[right] > nums[left]:
            # The first element is the smalles
            return nums[0]

        # Otherwise, use binary search to find the smallest
        while right >= left:
            # Locate the middle element
            mid = left + (right - left) // 2
            # If this middle element > mid+1, then mid+1 is smallest
            if nums[mid] > nums[mid + 1]:
                return nums[mid + 1]

            # If this middle element < mid-1, then mid-1 is the smallest
            if nums[mid - 1] > nums[mid]:
                return nums[mid]

            # If the mid element > than first element, smallest value is on right side
            if nums[mid] > nums[0]:
                left = mid + 1
            # If the mid element < the first element
            else:
                right = mid - 1

    def findPeakElement(self, nums: List[int]) -> int:
        """162 Find Peak Element

        Time Complexity: O(logN)
            - Binary Search: O(logN)

        Space Complexity: O(1)

        A peak element is an element that is striclty greater than its neightbors.

        given an integer array, nums, find a peak element and return its index. If the array
        contains multiple peaks, return the index to any of the peaks.

        You may imagine that nums[-1] = nums[n] = -INF

        Args:
            nums: (List[int]) Array of integers

        Returns: (int) The index of the peak

        """
        # Left and right pointers
        left, right = 0, len(nums) - 1

        while left < right:
            # Get the middle of the range being considered
            mid = (left + right) // 2
            # If the middle value is greater than the next value
            if nums[mid] > nums[mid + 1]:
                # Search the left side
                right = mid
            else:
                # Search the right side
                left = mid + 1
        return left

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """54 Spiral Matrix

        Given an m*n matrix, return all elements of the matrix in spiral order

        Args:
            matrix: (List[List[int]]) M*N Matrix

        Returns: (List[int]) Elements of matrix in spiral order

        """
        pass

    def maxProduct(self, nums: List[int]) -> int:
        """ 152. Maximum Product Subarray

        Time Complexity: O(N)
            - One loop through N

        Space Complexity: O(1)

        Given an integer array nums, find the contiguous subarray within an array (containing at least on number)
        which has the largest product

        Args:
            nums: (List[int]) Array of integers

        Returns: (int) Largest product from contiguous subarray

        """
        if len(nums) == 0:
            return 0

        cur_max, cur_min = nums[0], nums[0]
        res = cur_max

        for i in range(1, len(nums)):
            cur = nums[i]
            tmp_max = max(cur, cur_max * cur, cur_min * cur)
            cur_min = min(cur, cur_max * cur, cur_min * cur)

            cur_max = tmp_max

            res = max(cur_max, res)
        return res

    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        """1424 Diagonal Traverse 2

        Given a list of lists of integers, nums, return all elements of nums in diagonal order

        Args:
            nums:

        Returns:

        """
        d = len(nums)
        print(f'Depth: {d}')
        for i in range(d):
            pass

    def minDifference(self, nums: List[int]) -> int:
        """ 1509 Minimum Difference Between Largest and Smallest Value in Three Moves

        Args:
            nums:

        Returns:

        """
        moves = 3
        nums_size = len(nums)
        # If we can change all of the values to the smallest value, the difference is 0
        if nums_size - 1 <= moves:
            return 0

        # If we sort our numbers, we can quickly hold onto our largest and smallest values
        nums.sort()
        print(nums)

        # Get our initial difference with no changes
        min_dif = nums[-1] - nums[0]

        # We have 4 possible changes to consider within a window of 3
        # Make changes to the beginning and end
        # Make changes to the first or last two
        for i in range(4):
            # As our window shifts, we have a new largest and smallest to calculate difference
            min_dif = min(min_dif, nums[nums_size - 4 + i] - nums[i])
        return min_dif

    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        pass

    def smallestSubsequence(self, s: str) -> str:
        # STRING PROBLEM
        # Keep count of all the letters
        letter_count = collections.Counter(s)
        # Keep track of letters seen
        seen_set = set()
        # Keep track of substring chars
        char_list = []
        # For each letter in s
        for letter in s:
            # Decrement that letters count
            letter_count[letter] -= 1
            # Check to see if we've seen this letter before
            if letter in seen_set:
                # Skip if we have
                continue
            # While our list isn't empty
            while char_list:
                # If correct lexicographic order and count of last letter > 0
                if char_list[-1] > letter and letter_count[char_list[-1]] > 0:
                    # Pop off the last letter
                    last_letter = char_list.pop()
                    # Remove from seen set
                    seen_set.remove(last_letter)
                # Otherwise, break from the while loop
                else:
                    break
            # Add the letter our char list
            char_list.append(letter)
            # Mark that letter as seen
            seen_set.add(letter)
        return ''.join(char_list)

    def maxOperations(self, nums: List[int], k: int) -> int:
        # Keep a running count of each number
        num_count = collections.Counter(nums)
        # Keep track of operations
        op_count = 0
        # Go through each number
        for n in num_count:
            # Calculate our target addend
            target = k - n
            # If our target is the same as the current number
            if target == n:
                # Add to op_count how many operations we completed with this number
                op_count += num_count[n] // 2
            # If our target is not the same as our current number
            else:
                # Add to op_count if an operation can be performed, otherwise 0
                op_count += min(num_count[n], num_count[target])
            # Set the count of the current number to 0 so we don't use it again
            num_count[n] = 0
        return op_count

    pathSumCount = 0

    def pathSum(self, root: TreeNode, sum: int) -> int:
        def dfs(node, target_sum):
            # DFS traversal through the tree
            # From each node, we need to check all the sums of its paths

            # Base case
            if node is None:
                return

            # What is the sum start from this node
            cur_sum = 0

            # Check if any path from this node equals sum and add to solution
            dfsPathSumCheck(node, target_sum, cur_sum)
            # Go to the next node and try again
            dfs(node.left, target_sum)
            dfs(node.right, target_sum)

        def dfsPathSumCheck(node, target_sum, cur_sum):
            # Base case
            if node is None:
                return
            # What is the current sum
            cur_sum = node.val + cur_sum
            # If the current sum is our target, add this to our path count
            if cur_sum == target_sum:
                self.pathSumCount += 1

            # Traverse further down path
            dfsPathSumCheck(node.left, target_sum, cur_sum)
            dfsPathSumCheck(node.right, target_sum, cur_sum)

        dfs(root, sum)
        return self.pathSumCount

    def addTwoDigits(self, n):
        dig = [int(d) for d in str(n)]
        return dig[0] + dig[1]

    def minDeletions(self, arr):
        return self.lis(arr, len(arr))

    def lis(self, arr, n):
        n = len(arr)

        # Declare the list (array) for LIS and
        # initialize LIS values for all indexes
        lis = [1] * n

        # Compute optimized LIS values in bottom up manner
        for i in range(1, n):
            for j in range(0, i):
                if arr[i] > arr[j] and lis[i] < lis[j] + 1:
                    lis[i] = lis[j] + 1


        return max(lis)

    def canPermutePalindrome(self, s: str) -> bool:
        pass