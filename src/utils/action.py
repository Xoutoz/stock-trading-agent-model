from enum import Enum

class Action(Enum):
    """
    An enumeration representing possible actions in a stock trading environment.
    Attributes:
        HOLD (int): Represents the action of holding the current position (value: 0).
        BUY (int): Represents the action of buying stocks (value: 1).
        SELL (int): Represents the action of selling stocks (value: 2).
    Methods:
        __repr__(): Returns the name of the action as its string representation.
    """

    BUY = 0
    SELL = 1
    HOLD = 2

    def __repr__(self):
        return self.name


if __name__ == "__main__":
    action = Action.BUY
    print(f"Action: {action.name}, Value: {action.value}")

    action = Action.SELL
    print(f"Action: {action.name}, Value: {action.value}")

    action = Action.HOLD
    print(f"Action: {action.name}, Value: {action.value}")