import numpy as np


class Random():

    def __init__(self):
        self.bins = [0.5, 0.5]  # bins needs to be overwriten in the children

    def __gen_bins__(self, prob):
        # generate an array monotonically increasing with the cumulative of the probabilities. Each of the bins
        # represent a margin. Those margins will be used to place a number, say x like bins[i-1] <= x < bins[i]
        bins = np.empty((0, len(prob)))
        cum = 0
        for i in range(len(prob)):
            cum += prob[i]
            bins = np.append(bins, cum)

        return bins

    def get_random(self):
        n = np.random.random_sample()  # generate a number between 0 and 1
        # numpy.digitize function returns the position of the bin in wich a number x falls.
        # The bins are already generated based in the probabilities of the numbers in two dice.
        # Each index ``i`` returned by digitize is such that ``bins[i-1] <= x < bins[i]`` if
        # `bins` is monotonically increasing, or ``bins[i-1] > x >= bins[i]`` if
        # `bins` is monotonically decreasing
        return np.digitize([n], self.bins)[0]


class Dices(Random):

    def __init__(self):
        # probabilities of each number from 1 to 12 of a two dice roll
        self.prob = [0, (1 / 36), (2 / 36), (3 / 36), (4 / 36), (5 / 36), (6 / 36), (5 / 36), (4 / 36), (3 / 36),
                     (2 / 36), (1 / 36)]
        # generate an array of bins. With these bins we can generate a number between 2 and 12
        # based in their probabilities
        self.bins = self.__gen_bins__(self.prob)
        self.doub_prob = [0, 1, 0, (1 / 3), 0, (2 / 10), 0, (2 / 10), 0, (1 / 3), 0, 1]

    def roll_dice(self):
        # roll contains the position of n in the bins array. It coincides with the two dice number generated
        # it is cero based which means the final number will need to add 1
        roll = self.get_random()
        double = False

        if self.doub_prob[roll] > 0:
            # if the number generated has a probability greater than zero of being double
            # another number bewtween 0 and 1 is generated which depending of the double probability
            # will decide if the roll was a double or not
            d = np.random.random_sample()
            if d < self.doub_prob[roll]:
                # we only have two bins, it is not necesary to call digitize here.
                # if 'd' is less than the prob of double we consider it was a double, it was not
                # in the rest of the cases
                double = True
        return double, roll + 1  # we add 1 bc the roll number is 0 based


class Chance(Random):
    # card names
    STAY = 'Stay in Chance'
    JAIL = 'Go to Jail'
    TO_GO = 'Advance to GO'
    ILL = 'Go to Illinois Ave.'
    ST_CHR = 'Advance to St. Charles Place'
    NXT_RAIL = 'Advance to the nearest Railroad'
    BCK = 'Go Back 3 Spaces'
    READ_RAIL = 'Take a trip to Reading Railroad'
    BOARDWALK = 'Take a walk on the Boardwalk'
    UTIL = 'Advance token to nearest Utility'

    def __init__(self):
        # probabilities of each card. Since the only cards that are interesting for us are the movement cards,
        # the rest are merged in Stay in Chance option.
        self.card_names = [self.STAY, self.JAIL, self.TO_GO, self.ILL, self.ST_CHR, self.NXT_RAIL, self.BCK,
                           self.READ_RAIL, self.BOARDWALK, self.UTIL]
        self.prob = [(7 / 16), (1 / 16), (1 / 16), (1 / 16), (1 / 16), (1 / 16), (1 / 16), (1 / 16), (1 / 16), (1 / 16)]
        # generate an array of bins. With these bins we can generate a card corresponding to the bin selected
        # based in their probabilities.
        self.bins = self.__gen_bins__(self.prob)

    def get_card(self):
        return self.card_names[self.get_random()]


class Community(Random):
    # card names
    OUT_JAIL = 'Get Out Of Jail Free'
    COLL_MONEY = 'Collect Money'
    ADVANCE = 'Advance to GO'
    PAY = 'Pay Money'
    GO_JAIL = 'Go to Jail'

    def __init__(self):
        # probabilities of each card from 1 to 12 of a two dice roll
        self.card_names = [self.OUT_JAIL, self.COLL_MONEY, self.ADVANCE, self.PAY, self.GO_JAIL]
        self.prob = [(1 / 16), (3 / 16), (8 / 16), (3 / 16), (1 / 16)]
        # generate an array of bins. With these bins we can generate a number between 2 and 12
        # based in their probabilities
        self.bins = self.__gen_bins__(self.prob)

    def get_card(self):
        return self.card_names[self.get_random()]


class Board():
    # squares
    JAIL = 'Jail'
    GO_JAIL = 'Go to Jail'
    GO = 'Go'
    RAIL = 'Railroad'
    READ_RAIL = 'Reading Railroad'
    CHANCE = 'Chance'
    COMM = 'Community chest'
    ILL = 'Illinois Avenue'
    BOARDWALK = 'Boardwalk'
    ST_CHR = 'St. Charles Place'
    UTILITY = 'Utility'  # Not necesary the name of the specific Utility

    # rules
    MAX_JAIL = 3

    def __init__(self):
        self.squares = [self.GO, 'Mediterranean Avenue', self.COMM, None, None, self.READ_RAIL, None, self.CHANCE, None, None, self.JAIL,
                        self.ST_CHR, self.UTILITY, None, None, self.RAIL, None, self.COMM, None, None, None, None,
                        self.CHANCE, None, self.ILL, self.RAIL, None, None, self.UTILITY, None, self.GO_JAIL, None,
                        None, self.COMM, None, self.RAIL, self.CHANCE, None, None, self.BOARDWALK]
        self.board = {self.UTILITY: np.array([12, 28]), self.RAIL: np.array([5, 15, 25, 35])}
        self.chance = Chance()
        self.community = Community()

    def is_communitychest(self, pos):
        return self.COMM == self.squares[pos % len(self.squares)]

    def is_chance(self, pos):
        return self.CHANCE == self.squares[pos % len(self.squares)]

    def is_jail(self, pos):
        return self.JAIL == self.squares[pos % len(self.squares)]

    def move(self, pos, num):

        next = (pos + num) % len(self.squares)  # make sure it is not out of bounds.
        path = [next]
        free_jail = False
        if self.squares[next] == self.GO_JAIL:
            path.append(self.squares.index(self.JAIL))
        elif self.is_chance(next):
            card = self.chance.get_card()
            f, p = self.execute_chance(next, card)
            free_jail = free_jail or f
            path.extend(p)
        elif self.is_communitychest(next):
            card = self.community.get_card()
            f, p = self.execute_community(next, card)
            free_jail = free_jail or f
            path.extend(p)
        return free_jail, path

    def execute_chance(self, pos, card):

        if card == Chance.STAY:
            return False, []
        elif card == Chance.JAIL:
            return False, [self.squares.index(self.JAIL)]
        elif card == Chance.TO_GO:
            return False, [self.squares.index(self.GO)]
        elif card == Chance.ILL:
            return False, [self.squares.index(self.ILL)]
        elif card == Chance.ST_CHR:
            return False, [self.squares.index(self.ST_CHR)]
        elif card == Chance.READ_RAIL:
            return False, [self.squares.index(self.READ_RAIL)]
        elif card == Chance.BOARDWALK:
            return False, [self.squares.index(self.BOARDWALK)]
        elif card == Chance.UTIL:
            utils = self.board[self.UTILITY]
            next = utils[utils > pos]
            return False, [utils[0]] if len(next) == 0 else [next[0]]
        elif card == Chance.NXT_RAIL:
            rails = self.board[self.RAIL]
            next = rails[rails > pos]
            return False, [rails[0]] if len(next) == 0 else [next[0]]
        elif card == Chance.BCK:
            return self.move(pos, -3)
        else:
            return False, []

    def execute_community(self, pos, card):

        if card == Community.GO_JAIL:
            return False, [self.squares.index(self.JAIL)]
        elif card == Community.ADVANCE:
            return False, [self.squares.index(self.GO)]
        elif card == Community.OUT_JAIL:
            return True, []
        else:
            return False, []


class MonopolyGame():

    def __init__(self):
        self.dices = Dices()
        self.board = Board()

    def game_move(self, times):
        result = np.zeros((1, len(self.board.squares)), dtype=int)
        pos = 0
        n = 0
        jail = False
        free_jail = False
        max_jail = 0

        while n < times:

            double, next = self.dices.roll_dice()
            jail = (jail and not free_jail) or (
                    jail and not double)  # if you are in jail and the roll is not double you keep in jail

            if jail and (
                    max_jail < self.board.MAX_JAIL):  # assume that in three rounds we pay the fine and we get out of jail
                result[:, pos] += 1
                n += 1
                max_jail += 1
                continue

            f, path = self.board.move(pos, next)
            free_jail = f or free_jail
            print(path)
            result[:, path] += 1
            pos = path[-1]

            jail = self.board.is_jail(pos)

            if not double or jail:
                # if it is double, you throw again without counting the move. If it is double but you are in jail,
                # the move counts anyways
                n += 1

        return result


if __name__ == '__main__':
    d = Dices()
    double, num = d.roll_dice()
    print("double: " + str(double))
    print(num)
    while double:
        double, num = d.roll_dice()
        print("double: " + str(double))
        print(num)

    monopoly = MonopolyGame()
    res = np.zeros((1, len(monopoly.board.squares)), dtype=int)

    for i in range(100):
        res = res + monopoly.game_move(100)

    print(res[0])
    sum=res.sum(axis=1)[0]
    print (sum)
    print('--------AVERAGES-----------')
    for i in range(len(res[0])):
        square=monopoly.board.squares[i]
        if square != None:
            av= res[0][i]/sum
            print(f'Square {square}: {av}')
