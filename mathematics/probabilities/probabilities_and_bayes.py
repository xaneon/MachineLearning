from numpy.random import random, randn, choice, randint

P1 = P2 = P3 = P4 = P5 = P6 = 1/6
P_up = P_down = 1/2     # P(A) and P(B) are independent
                        # => P(B | A) = P(B)

def roll_dice():
    return int(random(1) * 6) + 1


def toss_coin():
    return choice(["heads", "tails"], p=[1/2, 1/2])
    # return choice([0, 1], p=[1/2, 1/2])


def bowl1():
    return choice(["vanilla", "chocolate"], p=[3/4, 1/4])


def bowl2():
    return choice(["vanilla", "chocolate"], p=[1/2, 1/2])


if __name__ == "__main__":
    # conditional probability:  P(A | B)
    #                           probability of A, given B is true
    # conjoint probability:     P(A and B) = p(A) * p(B)
    #                           probability that both A and B true
    print(roll_dice())
    print(toss_coin())
    print(f"Probability of 1 and 2: {P1 * P2}")
    print(f"Throwing the two dices: {(roll_dice(), roll_dice())}")
    print(f"Probability of two dices facing up: {P_up * P_down}")
    print(f"Throwing two coins: {toss_coin(), toss_coin()}")
    # if A: rain today, B: rain tomorrow
    # and A and B are dependent:
    # => P(A and B) = P(A) * P(B|A) # chance of rain higher tomorrow if it rains today
    print(f"Get cookie from bowl1: {bowl1()}")
    print(f"Get cookie from bowl2: {bowl2()}")
    # if you randomly choose a cookie from either bowl1
    # or bowl2 and it is vanilla, how likely is it that it
    # came from bowl1 ? P(A|B) is not equal to P(B|A)
    # Bayes:
    # P(B) * P(A|B) = P(A) * P(B|A)
    # => P(A|B) = (P(A) * P(B|A)) / P(B)
    # => P(bowl 1 | vanilla) = (P(bowl 1) * P(vanilla |bowl 1))
    #                                  / P(vanilla)
    # P(vanilla) = 5/8
    # P(bowl 1) = 1/2, P(bowl 1 | vanilla) = ?
    # P(vanilla | bowl 1) = 3/4
    # P(bowl 1 | vanilla) = (1/2 * 3/4) / 5/8 = (3/8 * 8/5) = 24/40 = 3/5 => 60 %


