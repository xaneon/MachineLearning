from numpy.random import random, randn, choice, randint

P1 = P2 = P3 = P4 = P5 = P6 = 1/6

def roll_dice():
    return int(random(1) * 6) + 1

def toss_coin():
    # return choice(["head", "tail"], p=[1/2, 1/2])
    return choice([0, 1], p=[1/2, 1/2])


if __name__ == "__main__":
    # conditional probability:  P(A | B)
    #                           probability of A, given B is true
    # conjoint probability:     P(A and B) = p(A) * p(B)
    #                           probability that both A and B true
    print(roll_dice())
    print(toss_coin())
    print(f"Probability of 1 and 2: {P1 * P2}")
    print(f"Throwing the two dices: {(roll_dice(), roll_dice())}")

