from enum import Enum

class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


if __name__ == "__main__":
    action = Action.BUY
    print(f"Action: {action.name}, Value: {action.value}")

    action = Action.SELL
    print(f"Action: {action.name}, Value: {action.value}")

    action = Action.HOLD
    print(f"Action: {action.name}, Value: {action.value}")