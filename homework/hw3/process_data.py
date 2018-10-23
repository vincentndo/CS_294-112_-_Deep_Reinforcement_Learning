import os
import argparse
import pickle
import math
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    args = parser.parse_args()

    plt.figure()
    for data_dir in args.logdir:

        with open( os.path.join(data_dir, "log.pkl"), 'rb' ) as file:
            dirname = data_dir.split('/')
            model_name = dirname[-1].split('-')[0] if dirname[-1] else dirname[-2].split('-')[0]
            dct = pickle.load(file)
            data_tuple = sorted( [(key, dct[key][0], dct[key][1]) for key in dct] )
            x1 = [e[0] for e in data_tuple if not math.isnan(e[1])]
            mean_episode_reward = [e[1] for e in data_tuple if not math.isnan(e[1])]
            x2 = [e[0] for e in data_tuple if not math.isinf(e[2])]
            best_mean_episode_reward = [e[2] for e in data_tuple if not math.isinf(e[2])]
            plt.plot(x1, mean_episode_reward, label=model_name + ' ' + "mean_episode_reward")
            plt.plot(x2, best_mean_episode_reward, label=model_name + ' ' + "best_mean_episode_reward")
            plt.xlabel("timestep t")
            plt.ylabel("(best) mean episode reward")
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.legend(loc="best")
    plt.grid()
    plt.show()

if __name__ == "__main__":

    main()