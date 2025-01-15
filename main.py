from HadamardALS import *
from HadamardHALS_multiple import BCD_multiple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from skimage import data
from skimage.transform import resize
import networkx as nx

def load_dataset(dataset_name, m=100, n=100, h=35):
    """
    Load the demanded dataset
    
    Parameters:
    dataset_name (str): Name of the dataset. 
                        Options : 'olivetti', 'camera', 'spectrometer', 'football', 'synthetic', 'low_rank_synthetic', 'miserables'
    m (int): first matrix dimension if synthetic data
    n (int): second matrix dimension if synthetic data
    h (int): true matrix rank if low rank synthetic data
    
    Returns:
    np.ndarray: Loaded dataset as NumPy array.
    """

    if dataset_name == "olivetti":
        from sklearn.datasets import fetch_olivetti_faces
        D = np.array(fetch_olivetti_faces()['data'])
    
    elif dataset_name == "camera":
        image = data.camera()
        image = resize(image, (256, 256), anti_aliasing=True)
        D = np.array(image)
    
    elif dataset_name == "spectrometer":
        with open("../datasets/lrs.data") as inp:
            spectro = inp.read().split('(')
        D = []
        spectro.pop(0)
        for line in spectro:
            line = line.split()[2:]
            line = [float(i.rstrip(')')) for i in line]
            D.append(line)
        D = np.array(D)
    
    elif dataset_name == "football":
        G = nx.read_gml('../datasets/football.gml')
        D = nx.to_numpy_array(G)

    elif dataset_name == "miserables":
        G = nx.read_gml('../datasets/lesmis.gml')
        D = nx.to_numpy_array(G)

    elif dataset_name == "synthetic":
        D = np.random.rand(m, n)

    elif dataset_name == "low_rank_synthetic":
        A1, B1, A2, B2 = rnd_initialization(2, m, n, h)
        D = (A1 @ B1) * (A2 @ B2)
    
    else:
        raise ValueError(f"Dataset '{dataset_name}' not available "
                         "Options : 'olivetti', 'camera', 'spectrometer', 'football', 'synthetic', 'low_rank_synthetic', 'miserables'")
    
    return D

if __name__ == "__main__":
    D = load_dataset("football")
    print(D.shape)

    fig, ax = plt.subplots()

    W1, H1, W2, H2, error, times = Hadamard_BCD(D, r=6)
    D_approx = (W1@H1)*(W2@H2)
    print(f"Final error for 2 factors : {np.linalg.norm(D_approx-D, 'fro')/np.linalg.norm(D, 'fro')}")
    ax.plot(times, error, alpha=0.6, marker="^", linewidth=2, label="2 factors", c='r')

    factors, error, times = BCD_multiple(D, r=4, n_factors=3)
    D_approx = np.prod([W@H for (W, H) in factors.values()], axis=0)
    print(f"Final error for 3 factors : {np.linalg.norm(D_approx-D, 'fro')/np.linalg.norm(D, 'fro')}")
    ax.plot(times, error, alpha=0.6, marker="o", linewidth=2, label="3 factors", c='limegreen')

    factors, error, times = BCD_multiple(D, r=3, n_factors=4)
    D_approx = np.prod([W@H for (W, H) in factors.values()], axis=0)
    print(f"Final error for 4 factors : {np.linalg.norm(D_approx-D, 'fro')/np.linalg.norm(D, 'fro')}")
    ax.plot(times, error, alpha=0.6, marker="*", linewidth=2, label="4 factors", c='b')

    ax.set_title(f"Multi factors comparison", fontsize=35)
    ax.set_xlabel("# vectors", fontsize=28)
    ax.set_ylabel("Relative error", fontsize=28)

    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(fontsize=24)
    plt.grid()
    plt.show()