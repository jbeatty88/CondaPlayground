import math
import sys
import random
from typing import List


class BC:
    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
        rec1_left = rec1[0]
        rec1_right = rec1[0] + rec1[2]
        rec1_top = rec1[1]
        rec1_bottom = rec1[1] + rec1[3]

        rec2_left = rec2[0]
        rec2_right = rec2[0] + rec2[2]
        rec2_top = rec2[1]
        rec2_bottom = rec2[1] + rec2[3]

        x_olap = max(0, min(rec1_right, rec2_right) - max(rec1_left, rec2_left))
        y_olap = max(0, min(rec1_bottom, rec2_bottom) - max(rec1_top, rec2_top))

        return True if x_olap * y_olap > 0 else False

    def modulo(self):
        numList = []
        # Store the input
        for i in range(10):
            numList.append(int(input()))
        print(len(set([n % 42 for n in numList])))

    def shiftingLetters(self, S: str, shifts: List[int]) -> str:
        letter = list(range(97, 123))
        for i in range(len(S)):
            shiftBy = shifts.pop(0)
            for c in range(i + 1):
                # Get idx of number
                numIdx = letter.index(ord(S[c]))
                newIdx = numIdx + shiftBy
                if newIdx > 25:
                    newIdx = newIdx - 25

                newChar = chr(letter[newIdx])
                S = S.replace(S[c], str(newChar))

        return S

    def prsteni(self):
        count = int(input())  # Get ring count
        ring_list = list(map(int, input().split()))  # Get list of rings and convert elements to int
        first_ring = int(ring_list[0])  # Save first ring radius
        for n in range(1, count):  # Compute each ring turns
            gcf = math.gcd(first_ring, ring_list[n]) if first_ring > ring_list[n] \
                else math.gcd(ring_list[n], first_ring)  # Find gcf, first number must be greater than second
            print('{}/{}'.format(first_ring // gcf, ring_list[n] // gcf))  # Divide first and next ring by gcf

    def das_blinkenlights(self):
        def lcm(a, b):  # Simple lcm method
            return abs(a * b) // math.gcd(a, b)

        l1Interval, l2Interval, maxSec = map(int, input().split())  # Read input
        print('yes' if lcm(l1Interval, l2Interval) <= maxSec else 'no')

    def pseudoprime_numbers(self):
        bound = 46340
        sieve, primes = [True] * (bound + 1), []
        sieve[0] = sieve[1] = False

        for i in range(2, bound + 1):
            if not sieve[i]:
                continue
            primes.append(i)
            for j in range(i * i, bound + 1, i):
                sieve[j] = False

        def isPrime(num):
            if num <= bound:
                return sieve[num]
            for p in primes:
                if num % p == 0:
                    return False
            return True

        # read input by line
        for i in sys.stdin:
            p, a = map(int, i.split())  # Split the line into p and a
            if not p and not a:  # Check if end of test case
                break
            if a > p or a < 1:
                print('no')
            if p < 2 or p > 1000000000:
                print('no')
            if isPrime(p):
                print('no')
            else:
                print('yes' if (pow(a, p, p)) is a else 'no')

    def flower_garden(self):
        testCaseCount = int(input())
        flowerCoord = []
        for i in sys.stdin:
            flowerCount, maxDist = map(int, i.split())

    def divisors2(self):
        ## Sieve of Eratosthenes ##
        bound = 46341  # Max bound for primes + 1 for index offset
        sieve, primes = [True] * (bound), []  # Initialize sieve to True array
        sieve[0] = sieve[1] = False  # Set 0, 1 to false
        for i in range(2, bound):  # Use sieve to generate list of primes
            if not sieve[i]:  # If not a prime, skip it
                continue
            primes.append(i)  # If it is a prime, add it to primes
            for j in range(i * i, bound, i):  # Set every ith not prime, from i * i
                sieve[j] = False

        ## Binomial function ## Python 3.8 you can use math.comb(n, k)
        def nK(n, k):  # Function to calculate number for prime factorization
            f = math.factorial
            return f(n) // f(k) // f(n - k)

        ## Number of Divisors
        def numDiv(n):
            PF_idx = 0
            PF = primes[PF_idx]
            ans = 1  # Start from ans = 1
            while PF * PF <= n:
                power = 0  # Count the power
                while n % PF == 0:
                    n /= PF
                    power += 1
                ans *= (power + 1)  # According to the formula
                PF_idx += 1
                PF = primes[PF_idx]
            if n != 1:  # Last factor has pow = 1, we add 1 to it
                ans *= 2
            return ans

        for i in sys.stdin.readlines():  # Until you reach EOF
            n, k = map(int, i.split())  # Get inputs
            print(numDiv(nK(n, k)))

    def divisors(self):
        def nK(n, k):  # Function to calculate number for prime factorization
            return (math.factorial(n)) / (math.factorial(k) * math.factorial(n - k))

        def prod(myList):  # Python 3.8 we can use math.prod()
            # Multiply elements one by one
            result = 1
            for x in myList:
                result = result * x
            return result

        # def primeFactors(n):
        #     primeFactors = []
        #     while n % 2 == 0:
        #         primeFactors.append(2)
        #         n = n / 2
        #     for i in range(3, int(math.sqrt(n)) + 1, 2):
        #         while n % i == 0:
        #             primeFactors.append(i)
        #             n = n / i
        #     if n > 2:
        #         primeFactors.append(n)
        #     return primeFactors

        bound = 46341  # Max bound for primes + 1 for index offset
        sieve, primes = [True] * (bound), []  # Initialize sieve to True array
        sieve[0] = sieve[1] = False  # Set 0, 1 to false

        for i in range(2, bound):  # Use sieve to generate list of primes
            if not sieve[i]:
                continue
            primes.append(i)
            for j in range(i * i, bound, i):
                sieve[j] = False

        for i in sys.stdin.readlines():  # Until you reach EOF
            n, k = map(int, i.split())  # Get inputs
            c = nK(n, k)
            exp = []
            div = c
            for prime in primes:
                expoCount = 0
                # While this prime number goes into c, divide it
                while (div / prime).is_integer():
                    div = div / prime
                    expoCount += 1
                expoCount += 1
                exp.append(expoCount)
                if prime ** 2 > c:
                    if div == 1:
                        break
                    elif div != 1:
                        exp.append(2)

            # primeFacList = primeFactors(nK(n, k))
            # expList = [0] * int(primeFacList[-1])
            # for x in primeFacList:
            #     expList[int(x) - 1] += 1
            # expList = [x+1 for x in expList]
            print(prod(exp))

    def sieveOfEratosthenes(self, bound: int) -> List[int]:
        sieve, primes = [True] * (bound), []
        sieve[0] = sieve[1] = False
        for i in range(2, bound + 1):
            if not sieve[i]:
                continue
            primes.append(i)
            for j in range(i * i, bound + 1, i):
                sieve[j] = False
        return primes

    def divisorsnonethree(self):
        ## Sieve of Eratosthenes ##
        def sieveOfErato(bound):
            sieve, primes, primeSquare = [1] * (bound + 1), [], [0] * ((bound) + 1)  # Initialize sieve to True array
            sieve[0] = sieve[1] = False  # Set 0, 1 to false
            for i in range(2, bound + 1):  # Use sieve to generate list of primes
                if not sieve[i]:  # If not a prime, skip it
                    continue
                primes.append(i)  # If it is a prime, add it to primes
                for j in range(i ** 2, bound + 1, i):  # Set every ith not prime, from i * i
                    sieve[j] = False

            for p in primes:
                if p ** 2 < (bound ** 2) + 2:
                    primeSquare[p ** 2] = True
            return sieve, primes, primeSquare

        ## Binomial function ## Python 3.8 you can use math.comb(n, k)
        def nK(n, k):  # Function to calculate number for prime factorization
            f = math.factorial
            return f(n) // f(k) // f(n - k)

        ## Number of Divisors
        def numDiv(n):
            sieve, primes, primeSquare = sieveOfErato(int(math.sqrt((2 ** 63) - 1)))
            ans = 1
            for p in primes:
                if p ** 3 > n:
                    break
                count = 1
                while (n / p).is_integer():
                    n = n // p
                    count = count + 1
                ans = ans * count
            if sieve[int(n)] == True:
                ans = ans * 2
            elif primeSquare[int(n)] == True:
                ans = ans * 3
            elif n != 1:
                ans = ans * 4
            return ans

        for i in sys.stdin.readlines():  # Until you reach EOF
            n, k = map(int, i.split())  # Get input
            print(numDiv(nK(n, k)))

    def divisorsQuick(self):
        ## Sieve of Eratosthenes ##
        bound = 46341  # Max bound for primes + 1 for index offset
        sieve, primes = [True] * (bound), []  # Initialize sieve to True array
        sieve[0] = sieve[1] = False  # Set 0, 1 to false
        for i in range(2, bound):  # Use sieve to generate list of primes
            if not sieve[i]:  # If not a prime, skip it
                continue
            primes.append(i)  # If it is a prime, add it to primes
            for j in range(i * i, bound, i):  # Set every ith not prime, from i * i
                sieve[j] = False

        # Recursive prime factorization for factorials
        def primeFactorFactorial(n):
            factors = []
            if n == 1:
                return []
            factors += primeFactorFactorial(n - 1)
            PF_idx = 0
            PF = primes[PF_idx]
            while PF ** 2 <= n:
                while n % PF == 0:
                    n /= PF
                    factors.append(PF)
                PF_idx += 1
                PF = primes[PF_idx]
            if n != 1:
                factors.append(n)
            return factors

        for i in sys.stdin.readlines():  # Until you reach EOF
            n, k = map(int, i.split())  # Get input
            num_pf = primeFactorFactorial(n)  # Get prime factors for n!
            denom_pf = primeFactorFactorial(k) + primeFactorFactorial(n - k)  # Get prime factors for k! and (k-n)!
            for d in denom_pf:  # Cancel out exponents
                num_pf.remove(d)
            nSet = list(set(num_pf))
            res = 1
            for num in nSet:
                res *= num_pf.count(num) + 1
            print(res)

    def inverseFactorial(self):
        n = int(input())

        divisor = 2
        while n // divisor != 1:
            n = n // divisor
            divisor += 1
        print(divisor)

    def iBoard(self):
        for line in sys.stdin.readlines():
            leftClick = True
            rightClick = True
            line = line.rstrip()
            for ch in line:
                for b in range(7):
                    if (ord(ch) >> b) & 1:
                        leftClick = not leftClick
                    else:
                        rightClick = not rightClick

            if leftClick and rightClick:
                print('free')
            else:
                print('trapped')

    def bits(self):
        numTestCases = int(input())
        for testCase in range(numTestCases):
            n = input()
            bits = 0
            for b in range(1, len(n) + 1):
                c = int(n[:b])
                bits = max(bits, bin(c).count("1"))
            print(bits)

    def deathstar(self):
        dimension = int(input())
        res = [0] * dimension
        matrix = []
        for n in range(dimension):
            # Get the line
            row = list(map(int, input().split()))
            # Remove the diagonal element
            # row.pop(n)
            # Add it to the final array
            matrix.append(row)

        # At this point, we have the matrix
        # Now compute A st. each Mij is equal to Ai & Aj
        for i in range(dimension):
            for j in range(dimension):
                # Skip the diagonal
                if i == j:
                    continue
                res[i] |= matrix[i][j]
                res[j] |= matrix[i][j]

        print(' '.join(str(elem) for elem in res))

    def bitByBit(self):
        while True:
            register = ["?"] * 32
            commands = int(input())
            if commands == 0:
                break
            for line in range(commands):
                command = list(input().split())
                if len(command) == 2:
                    if command[0] == "SET":
                        register[int(command[1])] = 1
                    else:
                        register[int(command[1])] = 0
                else:
                    i = int(command[1])
                    j = int(command[2])
                    if command[0] == "OR":
                        # One OR anything is 1
                        if register[i] == 1 or register[j] == 1:
                            register[i] = 1
                        # Handle the 0 with ? cases
                        elif register[i] == 0 and register[j] == "?":
                            register[i] = "?"
                        elif register[i] == "?" and register[j] == 0:
                            register[i] = "?"
                        # Two unknowns
                        elif register[i] == "?" and register[j] == "?":
                            register[i] = "?"
                        # Two knowns (in this case 0 and 1)
                        else:
                            register[i] = int(register[i]) | int(register[j])
                    else:
                        if register[i] == "?" and register[j] == 0:
                            register[i] = 0
                        elif register[i] == 0 and register[j] == "?":
                            register[i] = 0
                        elif register[i] == "?" or register[j] == "?":
                            register[i] = "?"
                        else:
                            register[i] = int(register[i]) & int(register[j])
            print("".join(reversed(''.join(str(e) for e in register))))

    def virtual_friends(self):
        social_network = {}
        num_test_cases = int(input())
        for test_case in range(num_test_cases):
            print(num_test_cases)
            num_friendships_formed = int(input())
            print(num_friendships_formed)
            for f in range(num_friendships_formed):
                n1, n2 = input().split()

    def even_up_solitare(self, fileName = None):
        if fileName:
            input_source = open("/home/joshbeatty/PycharmProjects/CondaPlayground/inputfiles/"+fileName)
        else:
            input_source = sys.stdin

        card_deck = []
        card_count = int(input_source.readline())
        for line in input_source.readlines():
            card_deck = list(map(int, line.split(' ')))

        n1 = 0
        n2 = 1
        while n2 < card_count:
            if (card_deck[n1] + card_deck[n2]) % 2 == 0:
                card_deck.pop(n1)
                card_deck.pop(n2)
                card_count = len(card_deck)
            else:
                n1 += 1
                n2 += 1

        input_source.close()

        return len(card_deck)

