from simulation import Simulation
from functions import *

if __name__ == "__main__":
    # Stworzenie obiektu Simulation dla danych z Monte Carlo (w domysle true_data=False)
    # Przekazywana jest ścieżka do danych i ścieżka do katalogu w którym zostaną umieszczone wyniki
    sim = Simulation('/home/jakub/Desktop/data/', "/home/jakub/PycharmProjects/Root/results")
    # Wywołanie funkcji wypełniającej histogramy
    sim.fill_all_histograms()
    # Wywołanie funkcji zapisującej histogramy do plików
    sim.save_all_histograms()

    # Stworzenie obiektu Simulation dla danych doświadczalnych
    # Przekazywana jest ścieżka do danych i ścieżka do katalogu w którym zostaną umieszczone wyniki
    sim_true_data = Simulation('/home/jakub/Desktop/data_true/', f"/home/jakub/PycharmProjects/Root/results",
                               true_data=True)
    # Wywołanie funkcji wypełniającej histogramy
    sim_true_data.fill_all_histograms()
    # Wywołanie funkcji zapisującej histogramy do plików
    sim_true_data.save_all_histograms()

    # Wywołanie funkcji do stworzenia łączonych histogramów
    draw_combined_histograms(sim, sim_true_data)
