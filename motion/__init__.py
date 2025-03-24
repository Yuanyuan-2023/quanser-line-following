import pandas as pd


def get_motion_queque(motion):
    outer_path = "motion/outerClockwise.csv"
    inner_path = "motion/innerClockwise.csv"
    
    if motion == "outerTurnRight":
        df = pd.read_csv(outer_path)
    elif motion == "outerTurnLeft":
        df = pd.read_csv(outer_path)
        df["Turn Speed"] = -df["Turn Speed"]
    elif motion == "innerTurnLeft":
        df = pd.read_csv(inner_path)
    elif motion == "innerTurnRight":
        df = pd.read_csv(inner_path)
        df["Turn Speed"] = -df["Turn Speed"]

    fd_list = list(df["Forward Speed"])
    turn_list = list(df["Turn Speed"])

    return fd_list, turn_list

def main():
    fd_list, turn_list = get_motion_queque("outerTurnRight")
    print(fd_list[:3], turn_list[:3])

    fd_list, turn_list = get_motion_queque("outerTurnLeft")
    print(fd_list[:3], turn_list[:3])

    fd_list, turn_list = get_motion_queque("innerTurnRight")
    print(fd_list[:3], turn_list[:3])

    fd_list, turn_list = get_motion_queque("innerTurnLeft")
    print(fd_list[:3], turn_list[:3])


if __name__ == "__main__":
    main()