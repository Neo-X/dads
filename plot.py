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
        df = pd.read_csv(f)
        df["name"] = name
        data_frames.append(df)
        print(f, name)

    data_frames = pd.concat(data_frames)

    plt.clf()
    ax = sns.lineplot(x="Step", y="Value", hue="name", data=data_frames)
    plt.savefig("plot.png")
