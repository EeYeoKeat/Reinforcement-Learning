
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_from_csv(data):
    
    for no, col in enumerate(list(data.columns[1:])):
        plt.plot(data[col], label=col, linewidth=0.5)
    
    plt.xlabel('Episodes',fontsize=12)
    plt.ylabel('Scores',fontsize=12)
    plt.title('Scores recieved by agents',fontsize=12)
    plt.legend()
    plt.show()


def read_csv(filename):
    results = pd.read_csv(filename)
    old_name = results.columns[0]
    results.rename(columns={old_name: 'episodes'}, inplace=True)
    
    return results


if __name__ == '__main__':
    filename = 'agent_scores.csv'
    result_data = read_csv(filename)
    plot_from_csv(result_data)

    

