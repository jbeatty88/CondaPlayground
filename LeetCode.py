import collections
import copy
import heapq
import itertools
import math
import time
from typing import List


# Definition for singly-linked list.
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
        pass

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
            solution.append(nums[i+n])

        return solution

    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        '''
        Given the array candies and the integer extraCandies, where candies[i]
        represents the number of candies that the ith kid has.

        For each kid check if there is a way to distribute extraCandies among the
        kids such that he or she can have the greatest number of candies among them.
        Notice that multiple kids can have the greatest number of candies.

        Args:
            candies:
            extraCandies:

        Returns:

        '''
