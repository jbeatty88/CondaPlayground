from sys import stdin


class AdventOfCode2020:
    def p1_1twoSumTo2020(self):
        num_list = []
        for line in stdin:
            if line == '\n':
                break
            num_list.append(int(line))

        for num in num_list:
            target = 2020 - num
            if num_list.__contains__(target):
                print(f'{target} + {num} = {target + num}\n{target} * {num} = {target * num}')
                break

    def p1_2threeSumTo2020(self):
        # num_list = [1721, 979, 366, 299, 675, 1456]
        num_list = []
        for line in stdin:
            if line == '\n':
                break
            num_list.append(int(line))
        print(num_list)

        for num in num_list:
            target1 = 2020 - num
            for n in num_list:
                target2 = target1 - n
                if num_list.__contains__(target2):
                    print(
                        f'{n} + {num} + {target2} = {n + num + target2}\n{n} * {num} * {target2} = {n * num * target2}')
                    break

    def p2_1PasswordPhilosophy(self):
        num_valid_pwds = 0

        for line in stdin:
            if line == '\n':
                break
            # Strip trailing spaces or escape chars
            line = line.rstrip()
            # Split at space
            bound_frag, letter_frag, pwd = line.split(" ")
            letter = letter_frag.strip(':')
            l_bound, r_bound = map(int, bound_frag.split('-'))
            num_valid_pwds += l_bound <= pwd.count(letter) <= r_bound
        return num_valid_pwds

    def p2_2PasswordPhilosophy(self):
        num_valid_pwds = 0

        for line in stdin:
            if line == '\n':
                break
            # Strip trailing spaces or escape chars
            line = line.rstrip()
            # Split at space
            index_frag, letter_frag, pwd = line.split(" ")
            letter = letter_frag.strip(':')
            i1, i2 = map(int, index_frag.split('-'))
            num_valid_pwds += i1 <= pwd.count(letter) <= i2
        return num_valid_pwds
