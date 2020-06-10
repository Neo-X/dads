import seaborn as sns; sns.set()
import pandas as pd
import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_files", type=str, nargs="+")
    parser.add_argument("--names", type=str, nargs="+")
    args = parser.parse_args()

    data_frames = []
    for f, name in zip(args.csv_files, args.names):
        df = pd.read_csv(f, names=["goal_x",
                                   "goal_y",
                                   "reward",
                                   "steps_to_goaL",
                                   "distance_0",
                                   "distance_1",
                                   "distance_2"])
        df["name"] = name
        df["goal"] = df["goal_x"].astype(str) + '_' + df["goal_y"].astype(str)
        data_frames.append(df)
        print(f, name)

    data_frames = pd.concat(data_frames)

    plt.clf()
    plt.title("Downstream Goal Reward")
    ax = sns.barplot(x="goal", y="reward", hue="name", data=data_frames)
    plt.savefig("bars.png")
