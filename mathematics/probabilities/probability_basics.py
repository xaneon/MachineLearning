from numpy.random import randint, random, randn

P_1 = 1/6  # probability that 1 occurs
P_2 = P_3 = P_4 = P_5 = P_6 = 1/6

P_12 = P_1 * P_2  # probability that 1 and 2 occurs

def roll_dice():
    return int(random(1) * 6) + 1

if __name__ == "__main__":
    print(f"Wahrscheintlichkeit für 1: {P_1}")
    print(f"Wahrscheintlichkeit für 2: {P_2}")
    print(f"Wahrscheintlichkeit für 1+2: {P_12}")
    print(f"Würfeln ergab: {roll_dice()}")
    # TODO
