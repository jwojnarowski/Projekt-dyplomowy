import ROOT
import uproot
import numpy as np
import pandas as pd
import math
from hist_types import *
from mass_histograms import *
from line_profiler_pycharm import profile
from pathlib import Path
from particle_masses import *


class Simulation:

    def __init__(self, data_path, results_path, true_data=False):
        """
        Konstruktor obiektu Simulation
        """

        # ścieżka do folderu 'results' jest ustawiana jako parametr obiektu
        self.results_path = results_path
        # true_data jest ustawiane jako parametr obiektu
        self.true_data = true_data
        # inicjalizacja obiektu do przechowywania danych
        self.data = pd.DataFrame([])
        # Otwieranie plików z danymi
        if true_data:
            self._create_dataframe_true_data(data_path)
        else:
            self._create_dataframe(data_path)
        # Sortowanie danych wraz z rosnącym eventNumber
        self.data = self.data.sort_values('eventNumber')
        # Wywołanie preselekcji
        self._preselection()

        # Inicjalizacja histogramów mas (w przypadku true_data=False są to histogramy z rekonstrukcji)
        self.mass_pipi = ROOT.TH1F('mass_pipi', '#pi#pi mass;m_{#pi#pi} [MeV];events', 100, 250, 1000)
        self.mass_ppi = ROOT.TH1F('mass_ppi', 'p#pi mass;m_{p#pi} [MeV];events', 100, 1000, 2500)
        self.mass_KK = ROOT.TH1F('mass_KK', 'KK mass;m_{KK} [MeV];events', 100, 950, 1500)

        # Inicjalizacja histogramów zliczeń (w przypadku true_data=False są to histogramy z rekonstrukcji)
        self.count_pi = ROOT.TH1I('count_pi', '#pi multiplicity;#pi multiplicity;events', 50, 0, 50)
        self.count_p = ROOT.TH1I('count_p', 'p multiplicity;p multiplicity;events', 20, 0, 20)
        self.count_K = ROOT.TH1I('count_K', 'K multiplicity;K multiplicity;events', 20, 0, 20)

        # Kroki wykonywane tylko dla danych z symulacji Monte Carlo
        if not self.true_data:
            # Wyznaczanie statystyk dla kryteriów na PID
            self.statistics_PID = self._get_statistics(PID, ['piplus_PIDK', 'piplus_PIDp'])
            # Wyznaczanie statystyk dla kryteriów dla ProbNN
            self.statistics_ProbNN = self._get_statistics(ProbNN, ['piplus_ProbNNk', 'piplus_ProbNNp'])
            # Zapisanie statystyk do pliku .csv
            self._save_statistics()
            # Obliczenie wydajności kryteriów na ProbNN oraz ich czystości identyfikacji w zależności od
            # pędu poprzecznego i pseudopośpieszności i zapisanie ich do plików
            self.calculate_efficiency_pt_1()
            self.calculate_efficiency_eta_1()
            self.calculate_efficiency_pt_2()
            self.calculate_efficiency_eta_2()

            # Tworzenie histogramów PID z kryteriami PID > 0.1
            self.pid_K = self._create_histogram_1D(PID, 'pid_K', 'PIDK;PIDK;events')
            self.pid_p = self._create_histogram_1D(PID, 'pid_p', 'PIDp;PIDp;events')
            self.pid_K_true = self._create_histogram_1D(PID, 'pid_K_true', 'PIDK if particle is K;PIDK;events')
            self.pid_p_true = self._create_histogram_1D(PID, 'pid_p_true', 'PIDp if particle is p;PIDp;events')
            self.pid_K_pi = self._create_histogram_1D(PID, 'pid_K_pi', 'PIDK if particle is pi;PIDK;events')
            self.pid_p_pi = self._create_histogram_1D(PID, 'pid_p_pi', 'PIDp if particle is pi;PIDp;events')

            # Tworzenie histogramów z ProbNN z kryteriami ProbNN > 0.9
            self.probnn_K = self._create_histogram_1D(ProbNN, 'probnn_K', 'ProbNNK;ProbNNK;events')
            self.probnn_p = self._create_histogram_1D(ProbNN, 'probnn_p', 'ProbNNp;ProbNNp;events')
            self.probnn_K_true = self._create_histogram_1D(ProbNN, 'probnn_K_true',
                                                           'ProbNNK if particle is K;ProbNNK;events')
            self.probnn_p_true = self._create_histogram_1D(ProbNN, 'probnn_p_true',
                                                           'ProbNNp if particle is p;ProbNNp;events')
            self.probnn_K_pi = self._create_histogram_1D(ProbNN, 'probnn_K_pi',
                                                         'ProbNNK if particle is pi;ProbNNK;events')
            self.probnn_p_pi = self._create_histogram_1D(ProbNN, 'probnn_p_pi',
                                                         'ProbNNp if particle is pi;ProbNNp;events')
            self.probnn_pi = self._create_histogram_1D(ProbNN, 'probnn_pi', 'ProbNNpi;ProbNNpi;events')
            self.probnn_pi_true = self._create_histogram_1D(ProbNN, 'probnn_pi_true',
                                                            'ProbNNpi if particle is pi;ProbNNpi;events')
            self.probnn_pi_not = self._create_histogram_1D(ProbNN, 'probnn_pi_not',
                                                           'ProbNNpi if particle is not pi;ProbNNpi;events')

            # Tworzenie 2-wymiarowych histogramów z PID/ProbNN z kryteriami: PID > 0.1, ProbNN > 0.9
            self.hist_K = self._create_histogram_2D(PID_ProbNN, 'hist_K', 'PIDK/ProbNNK;PIDK;ProbNNK')
            self.hist_p = self._create_histogram_2D(PID_ProbNN, 'hist_p', 'PIDp/ProbNNp;PIDp;ProbNNp')
            self.hist_K_true = self._create_histogram_2D(PID_ProbNN, 'hist_K_true',
                                                         'PIDK/ProbNNK if particle is K;PIDK;ProbNNK')
            self.hist_p_true = self._create_histogram_2D(PID_ProbNN, 'hist_p_true',
                                                         'PIDp/ProbNNp if particle is p;PIDp;ProbNNp')
            self.hist_K_pi = self._create_histogram_2D(PID_ProbNN, 'hist_K_pi',
                                                       'PIDK/ProbNNK if particle is pi;PIDK;ProbNNK')
            self.hist_p_pi = self._create_histogram_2D(PID_ProbNN, 'hist_p_pi',
                                                       'PIDp/ProbNNp if particle is pi;PIDp;ProbNNp')

            # Tworzenie 2-wymiarowych histogramów z PID/ProbNNpi z kryteriami: PID > 0.1
            self.hist_Kpi = self._create_histogram_2D(PID_ProbNNpi, 'hist_Kpi', 'PIDK/ProbNNpi;PIDK;ProbNNpi')
            self.hist_ppi = self._create_histogram_2D(PID_ProbNNpi, 'hist_ppi', 'PIDp/ProbNNpi;PIDp;ProbNNpi')
            self.hist_Kpi_true = self._create_histogram_2D(PID_ProbNNpi, 'hist_Kpi_true',
                                                           'PIDK/ProbNNpi if particle is K;PIDK;ProbNNpi')
            self.hist_ppi_true = self._create_histogram_2D(PID_ProbNNpi, 'hist_ppi_true',
                                                           'PIDp/ProbNNpi if particle is p;PIDp;ProbNNpi')
            self.hist_Kpi_pi = self._create_histogram_2D(PID_ProbNNpi, 'hist_Kpi_pi',
                                                         'PIDK/ProbNNpi if particle is pi;PIDK;ProbNNpi')
            self.hist_ppi_pi = self._create_histogram_2D(PID_ProbNNpi, 'hist_ppi_pi',
                                                         'PIDp/ProbNNpi if particle is pi;PIDp;ProbNNpi')

            # Tworzenie 2-wymiarowych histogramów ProbNN/p_T z kryteriami: ProbNN > 0.9
            self.probnnm_K = self._create_histogram_2D(ProbNN_m, 'ProbNNm_K', 'ProbNNK/P_{t};P_{t} [MeV];ProbNNK')
            self.probnnm_p = self._create_histogram_2D(ProbNN_m, 'ProbNNm_p', 'ProbNNp/P_{t};P_{t} [MeV];ProbNNp')
            self.probnnm_K_true = self._create_histogram_2D(ProbNN_m, 'ProbNNm_K_true',
                                                            'ProbNNK/P_{t} if particle is K;P_{t} [MeV];ProbNNK')
            self.probnnm_p_true = self._create_histogram_2D(ProbNN_m, 'ProbNNm_p_true',
                                                            'ProbNNp/P_{t} if particle is p;P_{t} [MeV];ProbNNp')
            self.probnnm_K_pi = self._create_histogram_2D(ProbNN_m, 'ProbNNm_K_pi',
                                                          'ProbNNK/P_{t} if particle is pi;P_{t} [MeV];ProbNNK')
            self.probnnm_p_pi = self._create_histogram_2D(ProbNN_m, 'ProbNNm_p_pi',
                                                          'ProbNNp/P_{t} if particle is pi;P_{t} [MeV];ProbNNp')
            self.probnnm_pi = self._create_histogram_2D(ProbNN_m, 'probnnm_pi', 'ProbNNpi/P_{t};P_{t} [MeV];ProbNNpi')
            self.probnnm_pi_true = self._create_histogram_2D(ProbNN_m, 'probnnm_pi_true',
                                                             'ProbNNpi/P_{t} if particle is pi;P_{t} [MeV];ProbNNpi')
            self.probnnm_pi_not = self._create_histogram_2D(ProbNN_m, 'probnnm_pi_not',
                                                            'ProbNNpi/P_{t} if particle is not pi;P_{t} [MeV];ProbNNpi')

            # Tworzenie 2-wymiarowych histogramów ProbNN/eta z kryteriami: ProbNN > 0.9
            self.probnneta_K = self._create_histogram_2D(ProbNN_eta, 'ProbNNeta_K', 'ProbNNK/#eta;#eta;ProbNNK')
            self.probnneta_p = self._create_histogram_2D(ProbNN_eta, 'ProbNNeta_p', 'ProbNNp/#eta;#eta;ProbNNp')
            self.probnneta_K_true = self._create_histogram_2D(ProbNN_eta, 'ProbNNeta_K_true',
                                                              'ProbNNK/#eta if particle is K;#eta;ProbNNK')
            self.probnneta_p_true = self._create_histogram_2D(ProbNN_eta, 'ProbNNeta_p_true',
                                                              'ProbNNp/#eta if particle is p;#eta;ProbNNp')
            self.probnneta_K_pi = self._create_histogram_2D(ProbNN_eta, 'ProbNNeta_K_pi',
                                                            'ProbNNK/#eta if particle is #eta;ProbNNK')
            self.probnneta_p_pi = self._create_histogram_2D(ProbNN_eta, 'ProbNNeta_p_pi',
                                                            'ProbNNp/#eta if particle is pi;#eta;ProbNNp')
            self.probnneta_pi = self._create_histogram_2D(ProbNN_eta, 'probnneta_pi', 'ProbNNpi/#eta;#eta;ProbNNpi')
            self.probnneta_pi_true = self._create_histogram_2D(ProbNN_eta, 'probnneta_pi_true',
                                                               'ProbNNpi/#eta if particle is pi;#eta;ProbNNpi')
            self.probnneta_pi_not = self._create_histogram_2D(ProbNN_eta, 'probnneta_pi_not',
                                                              'ProbNNpi/#eta if particle is not pi;#eta;ProbNNpi')

            # Tworzenie histogramów mas wyznaczonych ze zmiennej TRUEID
            self.mass_pipi_true = ROOT.TH1F('mass_pipi_true', '#pi#pi mass;m_{#pi#pi} [MeV];events', 100, 250, 1000)
            self.mass_ppi_true = ROOT.TH1F('mass_ppi_true', 'p#pi mass;m_{p#pi} [MeV];events', 100, 1000, 2500)
            self.mass_KK_true = ROOT.TH1F('mass_KK_true', 'KK mass;m_{KK} [MeV];events', 100, 950, 1500)

            # Tworzenie histogramów zliczeń wyznaczonych ze zmiennej TRUEID
            self.count_pi_true = ROOT.TH1I('count_pi_true', '#pi multiplicity;#pi multiplicity;events', 50, 0, 50)
            self.count_p_true = ROOT.TH1I('count_p_true', 'p multiplicity;p multiplicity;events', 20, 0, 20)
            self.count_K_true = ROOT.TH1I('count_K_true', 'K multiplicity;K multiplicity;events', 20, 0, 20)

    def __call__(self):
        """
        Gdy obiekt Simulation zostanie wywołany to:
        1. Gdy true_data=True - zwróci histogramy mas
        2. Gdy true_data=False - zwróci histogramy z rekonstrukcji mas
        """
        return self.mass_pipi, self.mass_ppi, self.mass_KK

    def _create_dataframe(self, directory: str):
        """
        Funkcja wczytująca dane z plików dla danych z symulacji Monte Carlo
        """
        # Stworzenie obiektu Path z przekazanej ścieżki do danych
        path = Path(directory)
        # Stworzenie listy plików .root w katalogu do którego prowadzi ścieżka
        files = list(path.glob('*.root'))
        files_to_dataframe = []
        # Iteracja po plikach
        for idx, file in enumerate(files):
            # Żeby dostać się do danych do ścieżki do każdego pliku trzeba dopisać ':minbias;1/DecayTree;1',
            # ponieważ wewnątrz plików są takie katalogi
            file_temp = str(file) + ':minbias;1/DecayTree;1'
            files_to_dataframe.append(file_temp)

        # Wczytujemy wartości wybranych zmiennych ze wszystkich plików do obiektu pd.DataFrame
        self.data = uproot.concatenate(files_to_dataframe, filter_name=['piplus_TRUEID', 'piplus_ID', 'piplus_TRUEP_E',
                                                                        'piplus_TRUEP_X', 'piplus_TRUEP_Y',
                                                                        'piplus_TRUEP_Z',
                                                                        'piplus_TRUEPT', 'piplus_P', 'piplus_PX',
                                                                        'piplus_PY',
                                                                        'piplus_PZ', 'piplus_PT', 'piplus_ETA',
                                                                        'piplus_PIDK', 'piplus_PIDp', 'piplus_ProbNNk',
                                                                        'piplus_ProbNNp', 'piplus_ProbNNpi',
                                                                        'eventNumber',
                                                                        'piplus_TRACK_GhostProb',
                                                                        'piplus_TRACK_CHI2NDOF',
                                                                        'piplus_IPCHI2_OWNPV'],
                                       library='pd')

    def _create_dataframe_true_data(self, directory: str):
        """
        Funkcja wczytująca dane z plików dla danych doświadczalnych
        """
        # Stworzenie obiektu Path z przekazanej ścieżki do danych
        path = Path(directory)
        # Stworzenie listy plików .root w katalogu do którego prowadzi ścieżka
        files = list(path.glob('*.root'))
        event_files = []
        # Iteracja po plikach
        for file in files:
            # Żeby dostać się do danych do ścieżki do każdego pliku trzeba dopisać ':minbias;1/DecayTree;1',
            # ponieważ wewnątrz plików są takie katalogi
            event_files.append(str(file) + ':minbias;1/DecayTree;1')

        # Wczytujemy wartości wybranych zmiennych ze wszystkich plików do obiektu pd.DataFrame
        self.data = uproot.concatenate(event_files, filter_name=['piplus_ID', 'piplus_P', 'piplus_PX', 'piplus_PY',
                                                                 'piplus_PZ', 'piplus_PT', 'piplus_ProbNNk',
                                                                 'piplus_ProbNNp', 'piplus_ProbNNpi', 'eventNumber',
                                                                 'piplus_TRACK_GhostProb', 'piplus_TRACK_CHI2NDOF',
                                                                 'piplus_IPCHI2_OWNPV'],
                                       library='pd')

    def _preselection(self):
        """
        Funkcja stosująca na danych kryteria preselekcyjne
        """
        # Kryterium na TRUE_ID jest stosowane tylko gdy true_data=False
        if not self.true_data:
            self.data = self.data.drop(self.data[self.data['piplus_TRUEID'] == 0].index)
        self.data = self.data.drop(self.data[self.data['piplus_P'] > 100000].index)
        self.data = self.data.drop(self.data[self.data['piplus_PT'] < 100].index)
        self.data = self.data.drop(self.data[self.data['piplus_TRACK_GhostProb'] > 0.3].index)
        self.data = self.data.drop(self.data[self.data['piplus_TRACK_CHI2NDOF'] > 3].index)
        self.data = self.data.drop(self.data[self.data['piplus_IPCHI2_OWNPV'] > 3].index)

    @staticmethod
    def _create_histogram_1D(hist_type: dict, name: str, title: str):
        """
        Funkcja tworząca 1-wymiarowe histogramy. Przekazywany jest typ histogramu zdefiniowany w pliku hist_types.py
        """
        return ROOT.TH1F(name, title, hist_type["nBins"], hist_type["xmin"], hist_type["xmax"])

    @staticmethod
    def _create_histogram_2D(hist_type: dict, name: str, title: str):
        """
        Funkcja tworząca 2-wymiarowe histogramy. Przekazywany jest typ histogramu zdefiniowany w pliku hist_types.py
        """
        return ROOT.TH2F(name, title, hist_type["nBins_1"], hist_type["xmin_1"], hist_type["xmax_1"],
                         hist_type["nBins_2"], hist_type["xmin_2"], hist_type["xmax_2"])

    def fill_all_histograms(self):
        """
        Funkcja wypełniająca histogramy
        """
        # Wypełnianie histogramów mas
        if self.true_data:
            # mass/count
            self.create_mass_histogram(data_reco, [self.mass_pipi, self.mass_ppi, self.mass_KK],
                                       [self.count_pi, self.count_p, self.count_K])
        else:
            # mass/count
            self.create_mass_histogram(data_true, [self.mass_pipi_true, self.mass_ppi_true, self.mass_KK_true],
                                       [self.count_pi_true, self.count_p_true, self.count_K_true])
            self.create_mass_histogram(data_reco, [self.mass_pipi, self.mass_ppi, self.mass_KK],
                                       [self.count_pi, self.count_p, self.count_K])

            # Wypełnianie histogramów PID
            self._fill_histogram_1D(self.pid_K_true, PID, 'piplus_PIDK', 321)
            self._fill_histogram_1D(self.pid_K_pi, PID, 'piplus_PIDK', 211)
            self._fill_histogram_1D(self.pid_K, PID, 'piplus_PIDK')
            self._fill_histogram_1D(self.pid_p_true, PID, 'piplus_PIDp', 2212)
            self._fill_histogram_1D(self.pid_p_pi, PID, 'piplus_PIDp', 211)
            self._fill_histogram_1D(self.pid_p, PID, 'piplus_PIDK')

            # Wypełnianie histogramów ProbNN
            self._fill_histogram_1D(self.probnn_K_true, ProbNN, 'piplus_ProbNNk', 321)
            self._fill_histogram_1D(self.probnn_K_pi, ProbNN, 'piplus_ProbNNk', 211)
            self._fill_histogram_1D(self.probnn_K, ProbNN, 'piplus_ProbNNk')
            self._fill_histogram_1D(self.probnn_p_true, ProbNN, 'piplus_ProbNNp', 2212)
            self._fill_histogram_1D(self.probnn_p_pi, ProbNN, 'piplus_ProbNNp', 211)
            self._fill_histogram_1D(self.probnn_p, ProbNN, 'piplus_ProbNNp')
            self._fill_histogram_1D(self.probnn_pi, ProbNN, 'piplus_ProbNNpi')
            self._fill_histogram_1D(self.probnn_pi_true, ProbNN, 'piplus_ProbNNpi', 211)
            self._fill_histogram_1D(self.probnn_pi_not, ProbNN, 'piplus_ProbNNpi', -211)

            # Wypełnianie histogramów PID/ProbNN
            self._fill_histogram_2D(self.hist_K_true, PID_ProbNN, ['piplus_PIDK', 'piplus_ProbNNk'], 321)
            self._fill_histogram_2D(self.hist_K_pi, PID_ProbNN, ['piplus_PIDK', 'piplus_ProbNNk'], 211)
            self._fill_histogram_2D(self.hist_K, PID_ProbNN, ['piplus_PIDK', 'piplus_ProbNNk'])
            self._fill_histogram_2D(self.hist_p_true, PID_ProbNN, ['piplus_PIDp', 'piplus_ProbNNp'], 2212)
            self._fill_histogram_2D(self.hist_p_pi, PID_ProbNN, ['piplus_PIDp', 'piplus_ProbNNp'], 211)
            self._fill_histogram_2D(self.hist_p, PID_ProbNN, ['piplus_PIDp', 'piplus_ProbNNp'])

            # Wypełnianie histogramów PID/ProbNNpi
            self._fill_histogram_2D(self.hist_Kpi_true, PID_ProbNNpi, ['piplus_PIDK', 'piplus_ProbNNk'], 321,
                                    ['piplus_PIDK', 'piplus_ProbNNpi'])
            self._fill_histogram_2D(self.hist_Kpi_pi, PID_ProbNNpi, ['piplus_PIDK', 'piplus_ProbNNk'], 211,
                                    ['piplus_PIDK', 'piplus_ProbNNpi'])
            self._fill_histogram_2D(self.hist_Kpi, PID_ProbNNpi, ['piplus_PIDK', 'piplus_ProbNNk'],
                                    to_save=['piplus_PIDK', 'piplus_ProbNNpi'])
            self._fill_histogram_2D(self.hist_ppi_true, PID_ProbNNpi, ['piplus_PIDp', 'piplus_ProbNNp'], 2212,
                                    ['piplus_PIDp', 'piplus_ProbNNpi'])
            self._fill_histogram_2D(self.hist_ppi_pi, PID_ProbNNpi, ['piplus_PIDp', 'piplus_ProbNNp'], 211,
                                    to_save=['piplus_PIDp', 'piplus_ProbNNpi'])
            self._fill_histogram_2D(self.hist_ppi, PID_ProbNNpi, ['piplus_PIDp', 'piplus_ProbNNp'],
                                    to_save=['piplus_PIDp', 'piplus_ProbNNpi'])

            # Wypełnianie histogramów ProbNN/pęd poprzeczny
            self._fill_histogram_2D(self.probnnm_K_true, ProbNN_m, ['piplus_TRUEPT', 'piplus_ProbNNk'], 321)
            self._fill_histogram_2D(self.probnnm_K_pi, ProbNN_m, ['piplus_TRUEPT', 'piplus_ProbNNk'], 211)
            self._fill_histogram_2D(self.probnnm_K, ProbNN_m, ['piplus_TRUEPT', 'piplus_ProbNNk'])
            self._fill_histogram_2D(self.probnnm_p_true, ProbNN_m, ['piplus_TRUEPT', 'piplus_ProbNNp'], 2212)
            self._fill_histogram_2D(self.probnnm_p_pi, ProbNN_m, ['piplus_TRUEPT', 'piplus_ProbNNp'], 211)
            self._fill_histogram_2D(self.probnnm_p, ProbNN_m, ['piplus_TRUEPT', 'piplus_ProbNNp'])
            self._fill_histogram_2D(self.probnnm_pi_true, ProbNN_m, ['piplus_TRUEPT', 'piplus_ProbNNpi'], 211)
            self._fill_histogram_2D(self.probnnm_pi_not, ProbNN_m, ['piplus_TRUEPT', 'piplus_ProbNNpi'], -211)
            self._fill_histogram_2D(self.probnnm_pi, ProbNN_m, ['piplus_TRUEPT', 'piplus_ProbNNpi'])

            # Wypełnianie histogramów ProbNN/pseudopośpieszność
            self._fill_histogram_2D(self.probnneta_K_true, ProbNN_eta, ['piplus_ETA', 'piplus_ProbNNk'], 321)
            self._fill_histogram_2D(self.probnneta_K_pi, ProbNN_eta, ['piplus_ETA', 'piplus_ProbNNk'], 211)
            self._fill_histogram_2D(self.probnneta_K, ProbNN_eta, ['piplus_ETA', 'piplus_ProbNNk'])
            self._fill_histogram_2D(self.probnneta_p_true, ProbNN_eta, ['piplus_ETA', 'piplus_ProbNNp'], 2212)
            self._fill_histogram_2D(self.probnneta_p_pi, ProbNN_eta, ['piplus_ETA', 'piplus_ProbNNp'], 211)
            self._fill_histogram_2D(self.probnneta_p, ProbNN_eta, ['piplus_ETA', 'piplus_ProbNNp'])
            self._fill_histogram_2D(self.probnneta_pi_true, ProbNN_eta, ['piplus_ETA', 'piplus_ProbNNpi'], 211)
            self._fill_histogram_2D(self.probnneta_pi_not, ProbNN_eta, ['piplus_ETA', 'piplus_ProbNNpi'], -211)
            self._fill_histogram_2D(self.probnneta_pi, ProbNN_eta, ['piplus_ETA', 'piplus_ProbNNpi'])

    def _fill_histogram_1D(self, hist: ROOT.TH1F, hist_type: dict, key: str, true_id: int = 0,
                           to_save: str = None):
        """
        Funkcja wypełniająca histogramy 1D

        :param hist: Histogram do wypełnienia
        :param hist_type: Typ histogramu z pliku hist_types.py
        :param key: Nazwa zmiennej na którą nałożone jest kryterium selekcji
        :param true_id: Warunek na TRUEID:
            1. Jeśli większe od 0, to cząstki na histogramie muszą być cząstką określoną przez podany numer
            2. Jeśli równe 0, to brak warunku na TRUEID
            3. Jeśli mniejsze od 0, to cząstki na histogramie mogą być dowolną cząstką, oprócz tej określonej przez
                moduł podanego numeru
        :param to_save: Nazwa zmiennej, którą zostanie wypełniony histogram (domyślnie zmienna przekazana przez key)
        """

        # Jeśli nie została przekazana zmienna którą ma być wypełniony histogram to jest nią ta przekazana jako key
        if to_save is None:
            to_save = key

        # Definiowanie warunków na wartości które mają wypełnić histogram
        if true_id == 0:
            condition = self.data[key] > hist_type['cutoff']
        elif true_id > 0:
            condition = (self.data[key] > hist_type['cutoff']) & (abs(self.data['piplus_TRUEID']) == true_id)
        else:
            condition = (self.data[key] > hist_type['cutoff']) & (abs(self.data['piplus_TRUEID']) != -true_id)

        # Wyciągnięcie z danych szukanych wartości (w miejsce odrzuconych cząstek pojawia się NaN)
        values = np.where(condition, self.data[to_save], np.NaN)
        # Odrzucenie wartości NaN z szukanych wartości
        values = values[~np.isnan(values)]
        # Wypełnienie histogramu szukanymi wartościami
        hist.FillN(values.size, values, np.ones(shape=values.size))

    def _fill_histogram_2D(self, hist: ROOT.TH2F, hist_type: dict, keys: list[str],
                           true_id: int = 0, to_save: list[str] = None):
        """
        Funkcja wypełniająca histogramy 2D

        :param hist: Histogram do wypełnienia
        :param hist_type: Typ histogramu z pliku hist_types.py
        :param keys: Nazwy zmiennych na którą nałożone są kryteria selekcji
        :param true_id: Warunek na TRUEID:
            1. Jeśli większe od 0, to cząstki na histogramie muszą być cząstką określoną przez podany numer
            2. Jeśli równe 0, to brak warunku na TRUEID
            3. Jeśli mniejsze od 0, to cząstki na histogramie mogą być dowolną cząstką, oprócz tej określonej przez
                moduł podanego numeru
        :param to_save: Nazwy zmiennych, którymi zostanie wypełniony histogram
            (domyślnie zmienne przekazane przez keys)
        """
        # Jeśli nie zostały przekazane zmienne którymi ma być wypełniony histogram to są nimi te przekazane jako keys
        if to_save is None:
            to_save = keys

        # Definiowanie warunków na wartości które mają wypełnić histogram
        if true_id == 0:
            condition = (self.data[keys[0]] > hist_type['cutoff_1']) & (self.data[keys[1]] > hist_type['cutoff_2'])
        elif true_id > 0:
            condition = (self.data[keys[0]] > hist_type['cutoff_1']) & (self.data[keys[1]] > hist_type['cutoff_2']) & (
                    abs(self.data['piplus_TRUEID']) == true_id)
        else:
            condition = (self.data[keys[0]] > hist_type['cutoff_1']) & (self.data[keys[1]] > hist_type['cutoff_2']) & (
                    abs(self.data['piplus_TRUEID']) != -true_id)
        # Wyciągnięcie z danych szukanych wartości 1 zmiennej (w miejsce odrzuconych cząstek pojawia się NaN)
        values_1 = np.where(condition, self.data[to_save[0]], np.NaN)
        # Odrzucenie wartości NaN z szukanych wartości
        values_1 = values_1[~np.isnan(values_1)]
        # Wyciągnięcie z danych szukanych wartości 2 zmiennej (w miejsce odrzuconych cząstek pojawia się NaN)
        values_2 = np.where(condition, self.data[to_save[1]], np.NaN)
        # Odrzucenie wartości NaN z szukanych wartości
        values_2 = values_2[~np.isnan(values_2)]
        # Wypełnienie histogramu szukanymi wartościami
        hist.FillN(values_1.size, values_1, values_2, np.ones(shape=values_1.size))

    def save_all_histograms(self):
        """
        Funkcja zapisująca wszytskie histogramy do plików
        """

        # Zapisywanie histogramów dla danych doświadczalnych
        if self.true_data:
            # Zapisywanie histogramów mas i krotności
            self._save_histogram(self.mass_pipi, f'{self.results_path}/mass_histograms_true_data/pipi_mass.png')
            self._save_histogram(self.mass_ppi, f'{self.results_path}/mass_histograms_true_data/ppi_mass.png')
            self._save_histogram(self.mass_KK, f'{self.results_path}/mass_histograms_true_data/KK_mass.png')
            self._save_histogram(self.count_pi, f'{self.results_path}/count_histograms_true_data/pi_count.png')
            self._save_histogram(self.count_p, f'{self.results_path}/count_histograms_true_data/p_count.png')
            self._save_histogram(self.count_K, f'{self.results_path}/count_histograms_true_data/K_count.png')

        # Zapisywanie histogramów dla danych symulacyjnych
        else:
            # Zapisywanie histogramów mas i krotności ze zmiennej TRUEID
            self._save_histogram(self.mass_pipi_true, f'{self.results_path}/mass_histograms/pipi_mass_true.png')
            self._save_histogram(self.mass_ppi_true, f'{self.results_path}/mass_histograms/ppi_mass_true.png')
            self._save_histogram(self.mass_KK_true, f'{self.results_path}/mass_histograms/KK_mass_true.png')
            self._save_histogram(self.count_pi_true, f'{self.results_path}/count_histograms/pi_count_true.png')
            self._save_histogram(self.count_p_true, f'{self.results_path}/count_histograms/p_count_true.png')
            self._save_histogram(self.count_K_true, f'{self.results_path}/count_histograms/K_count_true.png')

            # Zapisywanie histogramów mas i krotności z rekonstrukcji
            self._save_histogram(self.mass_pipi, f'{self.results_path}/mass_histograms/pipi_mass_reco.png')
            self._save_histogram(self.mass_ppi, f'{self.results_path}/mass_histograms/ppi_mass_reco.png')
            self._save_histogram(self.mass_KK, f'{self.results_path}/mass_histograms/KK_mass_reco.png')
            self._save_histogram(self.count_pi, f'{self.results_path}/count_histograms/pi_count_reco.png')
            self._save_histogram(self.count_p, f'{self.results_path}/count_histograms/p_count_reco.png')
            self._save_histogram(self.count_K, f'{self.results_path}/count_histograms/K_count_reco.png')

            # Zapisywanie histogramów PID
            self._save_histogram(self.pid_K_true, f"{self.results_path}/PID/pid_K_true.png")
            self._save_histogram(self.pid_p_true, f"{self.results_path}/PID/pid_p_true.png")
            self._save_histogram(self.pid_K_pi, f"{self.results_path}/PID/pid_K_pi.png")
            self._save_histogram(self.pid_p_pi, f"{self.results_path}/PID/pid_p_pi.png")
            self._save_histogram(self.pid_K, f"{self.results_path}/PID/pid_K.png")
            self._save_histogram(self.pid_p, f"{self.results_path}/PID/pid_p.png")

            # Zapisywanie histogramów ProbNN
            self._save_histogram(self.probnn_K_true, f"{self.results_path}/ProbNN/probnn_K_true.png")
            self._save_histogram(self.probnn_p_true, f"{self.results_path}/ProbNN/probnn_p_true.png")
            self._save_histogram(self.probnn_K_pi, f"{self.results_path}/ProbNN/probnn_K_pi.png")
            self._save_histogram(self.probnn_p_pi, f"{self.results_path}/ProbNN/probnn_p_pi.png")
            self._save_histogram(self.probnn_K, f"{self.results_path}/ProbNN/probnn_K.png")
            self._save_histogram(self.probnn_p, f"{self.results_path}/ProbNN/probnn_p.png")
            self._save_histogram(self.probnn_pi, f"{self.results_path}/ProbNN/probnn_pi.png")
            self._save_histogram(self.probnn_pi_true, f"{self.results_path}/ProbNN/probnn_pi_true.png")
            self._save_histogram(self.probnn_pi_not, f"{self.results_path}/ProbNN/probnn_pi_not.png")

            # Zapisywanie histogramów PID/ProbNN
            self._save_histogram(self.hist_K_true, f"{self.results_path}/PID_ProbNN/hist_K_true.png", True)
            self._save_histogram(self.hist_p_true, f"{self.results_path}/PID_ProbNN/hist_p_true.png", True)
            self._save_histogram(self.hist_K_pi, f"{self.results_path}/PID_ProbNN/hist_K_pi.png", True)
            self._save_histogram(self.hist_p_pi, f"{self.results_path}/PID_ProbNN/hist_p_pi.png", True)
            self._save_histogram(self.hist_K, f"{self.results_path}/PID_ProbNN/hist_K.png", True)
            self._save_histogram(self.hist_p, f"{self.results_path}/PID_ProbNN/hist_p.png", True)

            # Zapisywanie histogramów PID/ProbNNpi
            self._save_histogram(self.hist_Kpi_true, f"{self.results_path}/PID_ProbNNpi/hist_Kpi_true.png", True)
            self._save_histogram(self.hist_ppi_true, f"{self.results_path}/PID_ProbNNpi/hist_ppi_true.png", True)
            self._save_histogram(self.hist_Kpi_pi, f"{self.results_path}/PID_ProbNNpi/hist_Kpi_pi.png", True)
            self._save_histogram(self.hist_ppi_pi, f"{self.results_path}/PID_ProbNNpi/hist_ppi_pi.png", True)
            self._save_histogram(self.hist_Kpi, f"{self.results_path}/PID_ProbNNpi/hist_Kpi.png", True)
            self._save_histogram(self.hist_ppi, f"{self.results_path}/PID_ProbNNpi/hist_ppi.png", True)

            # Zapisywanie histogramów ProbNN/pęd poprzeczny
            self._save_histogram(self.probnnm_K_true, f"{self.results_path}/ProbNN_m/probnnm_K_true.png", True)
            self._save_histogram(self.probnnm_p_true, f"{self.results_path}/ProbNN_m/probnnm_p_true.png", True)
            self._save_histogram(self.probnnm_K_pi, f"{self.results_path}/ProbNN_m/probnnm_K_pi.png", True)
            self._save_histogram(self.probnnm_p_pi, f"{self.results_path}/ProbNN_m/probnnm_p_pi.png", True)
            self._save_histogram(self.probnnm_K, f"{self.results_path}/ProbNN_m/probnnm_K.png", True)
            self._save_histogram(self.probnnm_p, f"{self.results_path}/ProbNN_m/probnnm_p.png", True)
            self._save_histogram(self.probnnm_pi, f"{self.results_path}/ProbNN_m/probnnm_pi.png", True)
            self._save_histogram(self.probnnm_pi_true, f"{self.results_path}/ProbNN_m/probnnm_pi_true.png", True)
            self._save_histogram(self.probnnm_pi_not, f"{self.results_path}/ProbNN_m/probnnm_pi_not.png", True)

            # # Zapisywanie histogramów ProbNN/pseudopośpieszność
            self._save_histogram(self.probnneta_K_true, f"{self.results_path}/ProbNN_eta/probnneta_K_true.png", True)
            self._save_histogram(self.probnneta_p_true, f"{self.results_path}/ProbNN_eta/probnneta_p_true.png", True)
            self._save_histogram(self.probnneta_K_pi, f"{self.results_path}/ProbNN_eta/probnneta_K_pi.png", True)
            self._save_histogram(self.probnneta_p_pi, f"{self.results_path}/ProbNN_eta/probnneta_p_pi.png", True)
            self._save_histogram(self.probnneta_K, f"{self.results_path}/ProbNN_eta/probnneta_K.png", True)
            self._save_histogram(self.probnneta_p, f"{self.results_path}/ProbNN_eta/probnneta_p.png", True)
            self._save_histogram(self.probnneta_pi, f"{self.results_path}/ProbNN_eta/probnneta_pi.png", True)
            self._save_histogram(self.probnneta_pi_true, f"{self.results_path}/ProbNN_eta/probnneta_pi_true.png", True)
            self._save_histogram(self.probnneta_pi_not, f"{self.results_path}/ProbNN_eta/probnneta_pi_not.png", True)

    @staticmethod
    def _save_histogram(hist: ROOT.TObject, path: str, colz: bool = False):
        """
        Funkcja zapisująca histogram
        """

        # Ustawienia wizualne histogramów i stworzenie obiektu TCanvas
        hist.SetStats(0)
        hist.GetXaxis().SetTitleSize(0.05)
        hist.GetYaxis().SetTitleSize(0.05)
        hist.GetXaxis().SetLabelSize(0.04)
        hist.GetYaxis().SetLabelSize(0.04)
        c = ROOT.TCanvas('c', 'c', 375, 350)
        c.Draw()
        c.SetLeftMargin(0.15)
        c.SetRightMargin(0.15)
        c.SetBottomMargin(0.125)
        c.SetTopMargin(0.125)

        # Jeśli colz = True, histogram będzie kolorowy (dostępne tylko dla histogramów 2D)
        if colz:
            hist.Draw('COLZ')
        else:
            hist.Draw()
        # Zapisanie histogramu
        c.Print(path)

    def _get_statistics(self, hist_type: dict, keys: list[str]) -> dict:
        """
        Funkcja obliczająca statystyki

        :param hist_type: typ histogramu z pliku hist_types.py (dostępne tylko PID i ProbNN).
            Obliczne są statystyki dla kryteriów przekazanych przez typ histogramu dla kaonów i protonów
        :param keys: Zmienne na których stosowane są kryteria
        """

        # Oblicznie liczby cząstek K spełniających kryterium
        total_K = np.sum(np.where(self.data[keys[0]] > hist_type['cutoff'], 1, 0))
        # Obliczanie wielkości true_positive, false_positive, true_negative, false_negative dla cząśtki K
        # po spełnieniu kryteriów
        true_positive_K = np.sum(
            np.where((self.data[keys[0]] > hist_type['cutoff']) & (abs(self.data['piplus_TRUEID']) == 321) & (
                    abs(self.data['piplus_ID']) == 321),
                     1, 0))
        false_positive_K = np.sum(
            np.where((self.data[keys[0]] > hist_type['cutoff']) & (abs(self.data['piplus_TRUEID']) != 321) & (
                    abs(self.data['piplus_ID']) == 321),
                     1, 0))
        true_negative_K = np.sum(
            np.where((self.data[keys[0]] > hist_type['cutoff']) & (abs(self.data['piplus_TRUEID']) != 321) & (
                    abs(self.data['piplus_ID']) != 321),
                     1, 0))
        false_negative_K = np.sum(
            np.where((self.data[keys[0]] > hist_type['cutoff']) & (abs(self.data['piplus_TRUEID']) == 321) & (
                    abs(self.data['piplus_ID']) != 321),
                     1, 0))

        # Oblicznie liczby cząstek K spełniających kryterium
        total_p = np.sum(np.where(self.data[keys[1]] > hist_type['cutoff'], 1, 0))
        # Obliczanie wielkości true_positive, false_positive, true_negative, false_negative dla cząśtki K
        # po spełnieniu kryteriów
        true_positive_p = np.sum(
            np.where((self.data[keys[1]] > hist_type['cutoff']) & (abs(self.data['piplus_TRUEID']) == 2212) & (
                    abs(self.data['piplus_ID']) == 2212),
                     1, 0))
        false_positive_p = np.sum(
            np.where((self.data[keys[1]] > hist_type['cutoff']) & (abs(self.data['piplus_TRUEID']) != 2212) & (
                    abs(self.data['piplus_ID']) == 2212),
                     1, 0))
        true_negative_p = np.sum(
            np.where((self.data[keys[1]] > hist_type['cutoff']) & (abs(self.data['piplus_TRUEID']) != 2212) & (
                    abs(self.data['piplus_ID']) != 2212),
                     1, 0))
        false_negative_p = np.sum(
            np.where((self.data[keys[1]] > hist_type['cutoff']) & (abs(self.data['piplus_TRUEID']) == 2212) & (
                    abs(self.data['piplus_ID']) != 2212),
                     1, 0))

        # Tworzenie słownika z obliczonych wartości
        statistics = {
            'total_K': total_K,
            'true_positive_K': true_positive_K,
            'false_positive_K': false_positive_K,
            'true_negative_K': true_negative_K,
            'false_negative_K': false_negative_K,
            'total_p': total_p,
            'true_positive_p': true_positive_p,
            'false_positive_p': false_positive_p,
            'true_negative_p': true_negative_p,
            'false_negative_p': false_negative_p,
        }

        return statistics

    def _save_statistics(self):
        """
        Funkcja zapisująca statystyki do pliku .csv
        """
        statistics_DF = pd.DataFrame([self.statistics_PID, self.statistics_ProbNN], index=['PID', 'ProbNN'])
        statistics_DF.to_csv(f'{self.results_path}/statistics/statistics.csv')

    @profile
    def create_mass_histogram(self, hist_type: dict, mass_hists: list[ROOT.TH1], count_hists: list[ROOT.TH1]):
        """
        Funkcja obliczająca i tworząca histogramy mas i krotności

        :param hist_type: Typ histogramu z pliku mass_hisstograms.py:
            1. data_reco dla rekonstrukcji masy z danych symulacyjnych i dla masy z danych doświadczalnych
            2. data_true dla masy ze zmiennej TRUEID z danych symulacyjnych
        :param mass_hists: Histogramy mas do wypełnienia (po kolei: pipi, ppi, KK)
        :param count_hists: Histogramy krotności do wypełnienia (po kolei: pi, p, K)
        :return:
        """

        # Wyciągamy z danych interesujące nas zmienne
        if hist_type["ID"] == "piplus_TRUEID":
            data = self.data[
                [hist_type["ID"], hist_type["E"], hist_type["Px"], hist_type["Py"], hist_type["Pz"], "eventNumber"]]
        elif hist_type["ID"] == "piplus_ID":
            data = self.data[
                [hist_type["ID"], hist_type["E"], hist_type["Px"], hist_type["Py"], hist_type["Pz"],
                 hist_type["ProbNNK"], hist_type["ProbNNp"], hist_type["ProbNNpi"], "eventNumber"]]
        else:
            data = self.data["eventNumber"]

        # Tablica numerów zdarzeń dla wszystkich cząstek
        event_numbers_all = data["eventNumber"].values
        # Usunięcie wszystkich duplikujących się numerów zdarzeń
        event_numbers = np.unique(event_numbers_all)
        # Stworzenie tablicy przechowującej informację w którym miejscu w danych zaczyna się następne zdarzenie.
        # Ważna jest kolejność w jakiej dane są ustawione
        new_event = np.concatenate(np.array(np.where(np.diff(event_numbers_all)))) + 1
        new_event = np.append(new_event, event_numbers_all.size)
        new_event = np.insert(new_event, 0, 0)

        # Warunki na cząstki
        if hist_type["ID"] == "piplus_TRUEID":
            # Warunki z TRUEID
            condition_pi_plus = data[hist_type["ID"]] == 211
            condition_pi_minus = data[hist_type["ID"]] == -211
            condition_K_plus = data[hist_type["ID"]] == 321
            condition_K_minus = data[hist_type["ID"]] == -321
            condition_p_plus = data[hist_type["ID"]] == 2212
            condition_p_minus = data[hist_type["ID"]] == -2212
        elif hist_type["ID"] == "piplus_ID":
            # Warunki z kryteriami selekcji. Do zdeterminowania ładunku cząstku użyta została zmienna ID.
            # Jeśli ID >= 0: ładunek dodatni (0 jest uwzględnione, żeby program nie przestał działać jeśli wystąpi
            # taka wartość. Nie powinno się to nigdy zdarzyć, jest to tylko środek zapobiegawczy)
            # Jeśli ID < 0: ładunek ujemny
            condition_pi_plus = (data[hist_type["ProbNNpi"]] > hist_type["cutoff_pi"]) & (data[hist_type["ID"]] >= 0)
            condition_pi_minus = (data[hist_type["ProbNNpi"]] > hist_type["cutoff_pi"]) & (data[hist_type["ID"]] < 0)
            condition_K_plus = (data[hist_type["ProbNNK"]] > hist_type["cutoff_K"]) & (data[hist_type["ID"]] >= 0)
            condition_K_minus = (data[hist_type["ProbNNK"]] > hist_type["cutoff_K"]) & (data[hist_type["ID"]] < 0)
            condition_p_plus = (data[hist_type["ProbNNp"]] > hist_type["cutoff_p"]) & (data[hist_type["ID"]] >= 0)
            condition_p_minus = (data[hist_type["ProbNNp"]] > hist_type["cutoff_p"]) & (data[hist_type["ID"]] < 0)
        else:
            # Uwzglęniony else, żeby kompilator nie zgłaszał warningów. Ten przypadek nigdy nie występuje.
            condition_pi_plus = 0
            condition_pi_minus = 0
            condition_K_plus = 0
            condition_K_minus = 0
            condition_p_plus = 0
            condition_p_minus = 0

        if hist_type["ID"] == "piplus_ID":
            # Stworzenie tablic zawierających energie konkretnych rodzajów cząstek w celu późniejszych szybszych
            # obliczeń. Przypadek dla typu histogramu: data_reco. Pod zmienną hist_type["E"] przypisany jest pęd cząstki
            pi_plus_E_all = np.sqrt(data.loc[condition_pi_plus][hist_type["E"]].values ** 2 + pi_mass ** 2)
            pi_minus_E_all = np.sqrt(data.loc[condition_pi_minus][hist_type["E"]].values ** 2 + pi_mass ** 2)
            p_plus_E_all = np.sqrt(data.loc[condition_p_plus][hist_type["E"]].values ** 2 + p_mass ** 2)
            p_minus_E_all = np.sqrt(data.loc[condition_p_minus][hist_type["E"]].values ** 2 + p_mass ** 2)
            K_plus_E_all = np.sqrt(data.loc[condition_K_plus][hist_type["E"]].values ** 2 + K_mass ** 2)
            K_minus_E_all = np.sqrt(data.loc[condition_K_minus][hist_type["E"]].values ** 2 + K_mass ** 2)
        else:
            # Stworzenie tablic zawierających energie konkretnych rodzajów cząstek w celu późniejszych szybszych
            # obliczeń. Przypadek dla typu histogramu: data_true.
            pi_plus_E_all = data.loc[condition_pi_plus][hist_type["E"]].values
            pi_minus_E_all = data.loc[condition_pi_minus][hist_type["E"]].values
            p_plus_E_all = data.loc[condition_p_plus][hist_type["E"]].values
            p_minus_E_all = data.loc[condition_p_minus][hist_type["E"]].values
            K_plus_E_all = data.loc[condition_K_plus][hist_type["E"]].values
            K_minus_E_all = data.loc[condition_K_minus][hist_type["E"]].values

        # Stworzenie tablic zawierających wszystkie współrzędne pędów konkretnych rodzajów cząstek w celu późniejszych
        # szybszych obliczeń
        pi_plus_PX_all = data.loc[condition_pi_plus][hist_type["Px"]].values
        pi_plus_PY_all = data.loc[condition_pi_plus][hist_type["Py"]].values
        pi_plus_PZ_all = data.loc[condition_pi_plus][hist_type["Pz"]].values
        pi_minus_PX_all = data.loc[condition_pi_minus][hist_type["Px"]].values
        pi_minus_PY_all = data.loc[condition_pi_minus][hist_type["Py"]].values
        pi_minus_PZ_all = data.loc[condition_pi_minus][hist_type["Pz"]].values
        p_plus_PX_all = data.loc[condition_p_plus][hist_type["Px"]].values
        p_plus_PY_all = data.loc[condition_p_plus][hist_type["Py"]].values
        p_plus_PZ_all = data.loc[condition_p_plus][hist_type["Pz"]].values
        p_minus_PX_all = data.loc[condition_p_minus][hist_type["Px"]].values
        p_minus_PY_all = data.loc[condition_p_minus][hist_type["Py"]].values
        p_minus_PZ_all = data.loc[condition_p_minus][hist_type["Pz"]].values
        K_plus_PX_all = data.loc[condition_K_plus][hist_type["Px"]].values
        K_plus_PY_all = data.loc[condition_K_plus][hist_type["Py"]].values
        K_plus_PZ_all = data.loc[condition_K_plus][hist_type["Pz"]].values
        K_minus_PX_all = data.loc[condition_K_minus][hist_type["Px"]].values
        K_minus_PY_all = data.loc[condition_K_minus][hist_type["Py"]].values
        K_minus_PZ_all = data.loc[condition_K_minus][hist_type["Pz"]].values

        # Inicjalizacja kumulatywnych zmiennych służących do liczenia cząstek wewnątrz pętli
        pi_plus_cum = 0
        pi_minus_cum = 0
        p_plus_cum = 0
        p_minus_cum = 0
        K_plus_cum = 0
        K_minus_cum = 0

        # Iteracja po wszystkich numerach zdarzeń
        for idx, event in enumerate(event_numbers):
            if hist_type["ID"] == "piplus_TRUEID":
                # Przypadek dla histogramu typu: data_true. Definiowanie warunków na konkretne cząstki znajdujące się
                # w danym zdarzeniu. Tablica new_event zawiera indeksy z głównych danych na których zaczynają się
                # kolejne zdarzenia.
                condition_pi_plus = data[hist_type["ID"]].values[new_event[idx]:new_event[idx + 1]] == 211
                condition_pi_minus = data[hist_type["ID"]].values[new_event[idx]:new_event[idx + 1]] == -211
                condition_K_plus = data[hist_type["ID"]].values[new_event[idx]:new_event[idx + 1]] == 321
                condition_K_minus = data[hist_type["ID"]].values[new_event[idx]:new_event[idx + 1]] == -321
                condition_p_plus = data[hist_type["ID"]].values[new_event[idx]:new_event[idx + 1]] == 2212
                condition_p_minus = data[hist_type["ID"]].values[new_event[idx]:new_event[idx + 1]] == -2212
            elif hist_type["ID"] == "piplus_ID":
                # Przypadek dla histogramu typu: data_reco. Definiowanie warunków na konkretne cząstki znajdujące się
                # w danym zdarzeniu. Ponownie do zdeterminowania ładunku wykorzystana jest zmienna ID
                condition_pi_plus = (data[hist_type["ProbNNpi"]].values[new_event[idx]:new_event[idx + 1]] > hist_type[
                    "cutoff_pi"]) & (data[hist_type["ID"]].values[new_event[idx]:new_event[idx + 1]] >= 0)
                condition_pi_minus = (data[hist_type["ProbNNpi"]].values[new_event[idx]:new_event[idx + 1]] > hist_type[
                    "cutoff_pi"]) & (data[hist_type["ID"]].values[new_event[idx]:new_event[idx + 1]] < 0)
                condition_K_plus = (data[hist_type["ProbNNK"]].values[new_event[idx]:new_event[idx + 1]] > hist_type[
                    "cutoff_K"]) & (data[hist_type["ID"]].values[new_event[idx]:new_event[idx + 1]] >= 0)
                condition_K_minus = (data[hist_type["ProbNNK"]].values[new_event[idx]:new_event[idx + 1]] > hist_type[
                    "cutoff_K"]) & (data[hist_type["ID"]].values[new_event[idx]:new_event[idx + 1]] < 0)
                condition_p_plus = (data[hist_type["ProbNNp"]].values[new_event[idx]:new_event[idx + 1]] > hist_type[
                    "cutoff_p"]) & (data[hist_type["ID"]].values[new_event[idx]:new_event[idx + 1]] >= 0)
                condition_p_minus = (data[hist_type["ProbNNp"]].values[new_event[idx]:new_event[idx + 1]] > hist_type[
                    "cutoff_p"]) & (data[hist_type["ID"]].values[new_event[idx]:new_event[idx + 1]] < 0)
            else:
                # Uwzglęniony else, żeby kompilator nie zgłaszał warningów. Ten przypadek nigdy nie występuje.
                condition_pi_plus = 0
                condition_pi_minus = 0
                condition_K_plus = 0
                condition_K_minus = 0
                condition_p_plus = 0
                condition_p_minus = 0

            # Zliczane są liczby konkretnych cząstek w danym zdarzeniu
            pi_plus_count = np.count_nonzero(np.where(condition_pi_plus, 1, 0))
            pi_minus_count = np.count_nonzero(np.where(condition_pi_minus, 1, 0))
            p_plus_count = np.count_nonzero(np.where(condition_p_plus, 1, 0))
            p_minus_count = np.count_nonzero(np.where(condition_p_minus, 1, 0))
            K_plus_count = np.count_nonzero(np.where(condition_K_plus, 1, 0))
            K_minus_count = np.count_nonzero(np.where(condition_K_minus, 1, 0))

            # Zliczone wartości dodawane są do zmiennych kumulatywnych
            pi_plus_cum += pi_plus_count
            pi_minus_cum += pi_minus_count
            p_plus_cum += p_plus_count
            p_minus_cum += p_minus_count
            K_plus_cum += K_plus_count
            K_minus_cum += K_minus_count

            # Wypełnianie histogramów krotności
            count_hists[0].Fill(pi_plus_count + pi_minus_count)
            count_hists[1].Fill(p_plus_count + p_minus_count)
            count_hists[2].Fill(K_plus_count + K_minus_count)

            # Jeśli w danym zdarzeniu zarejestrowano przynajmniej 1 pi+ i przynajmniej 1 pi-
            if pi_plus_count != 0 and pi_minus_count != 0:
                # Tworzone są tablice 2x2 wwszystkich kombinacji energii i współrzędnych pędu czątek pi+ i pi-
                # zarejestrowanych w tym zdarzeniu
                pi_plus_E, pi_minus_E = np.meshgrid(pi_plus_E_all[pi_plus_cum - pi_plus_count:pi_plus_cum],
                                                    pi_minus_E_all[pi_minus_cum - pi_minus_count:pi_minus_cum])
                pi_plus_PX, pi_minus_PX = np.meshgrid(pi_plus_PX_all[pi_plus_cum - pi_plus_count:pi_plus_cum],
                                                      pi_minus_PX_all[pi_minus_cum - pi_minus_count:pi_minus_cum])
                pi_plus_PY, pi_minus_PY = np.meshgrid(pi_plus_PY_all[pi_plus_cum - pi_plus_count:pi_plus_cum],
                                                      pi_minus_PY_all[pi_minus_cum - pi_minus_count:pi_minus_cum])
                pi_plus_PZ, pi_minus_PZ = np.meshgrid(pi_plus_PZ_all[pi_plus_cum - pi_plus_count:pi_plus_cum],
                                                      pi_minus_PZ_all[pi_minus_cum - pi_minus_count:pi_minus_cum])
                # Obliczanie mas niezmienniczych dla wszystkich kombinacji
                values_to_fill = np.sqrt((pi_plus_E + pi_minus_E) ** 2 - (pi_plus_PX + pi_minus_PX) ** 2 - (
                        pi_plus_PY + pi_minus_PY) ** 2 - (pi_plus_PZ + pi_minus_PZ) ** 2)
                # Redukowanie wymiaru macierzy jeśli tablice kombinacji nie są wektorami
                if pi_plus_count > 1 and pi_minus_count > 1:
                    values_to_fill = np.concatenate(values_to_fill, axis=0)
                # Wypełniane histogramu masy dla pary cząstek pipi
                mass_hists[0].FillN(values_to_fill.size, values_to_fill, np.ones(values_to_fill.size))

            # Jeśli w danym zdarzeniu zarejestrowano przynajmniej 1 K+ i przynajmniej 1 K-
            if K_plus_count != 0 and K_minus_count != 0:
                # Tworzone są tablice 2x2 wwszystkich kombinacji energii i współrzędnych pędu czątek K+ i K-
                # zarejestrowanych w tym zdarzeniu
                K_plus_E, K_minus_E = np.meshgrid(K_plus_E_all[K_plus_cum - K_plus_count:K_plus_cum],
                                                  K_minus_E_all[K_minus_cum - K_minus_count:K_minus_cum])
                K_plus_PX, K_minus_PX = np.meshgrid(K_plus_PX_all[K_plus_cum - K_plus_count:K_plus_cum],
                                                    K_minus_PX_all[K_minus_cum - K_minus_count:K_minus_cum])
                K_plus_PY, K_minus_PY = np.meshgrid(K_plus_PY_all[K_plus_cum - K_plus_count:K_plus_cum],
                                                    K_minus_PY_all[K_minus_cum - K_minus_count:K_minus_cum])
                K_plus_PZ, K_minus_PZ = np.meshgrid(K_plus_PZ_all[K_plus_cum - K_plus_count:K_plus_cum],
                                                    K_minus_PZ_all[K_minus_cum - K_minus_count:K_minus_cum])
                # Obliczanie mas niezmienniczych dla wszystkich kombinacji
                values_to_fill = np.sqrt(
                    (K_plus_E + K_minus_E) ** 2 - (K_plus_PX + K_minus_PX) ** 2 - (K_plus_PY + K_minus_PY) ** 2 - (
                            K_plus_PZ + K_minus_PZ) ** 2)
                # Redukowanie wymiaru macierzy jeśli tablice kombinacji nie są wektorami
                if K_plus_count > 1 and K_minus_count > 1:
                    values_to_fill = np.concatenate(values_to_fill, axis=0)
                # Wypełniane histogramu masy dla pary cząstek KK
                mass_hists[2].FillN(values_to_fill.size, values_to_fill, np.ones(values_to_fill.size))

            # Jeśli w danym zdarzeniu zarejestrowano przynajmniej 1 p i przynajmniej 1 pi-
            if p_plus_count != 0 and pi_minus_count != 0:
                # Tworzone są tablice 2x2 wwszystkich kombinacji energii i współrzędnych pędu czątek p i pi-
                # zarejestrowanych w tym zdarzeniu
                p_plus_E, pi_minus_E = np.meshgrid(p_plus_E_all[p_plus_cum - p_plus_count:p_plus_cum],
                                                   pi_minus_E_all[pi_minus_cum - pi_minus_count:pi_minus_cum])
                p_plus_PX, pi_minus_PX = np.meshgrid(p_plus_PX_all[p_plus_cum - p_plus_count:p_plus_cum],
                                                     pi_minus_PX_all[pi_minus_cum - pi_minus_count:pi_minus_cum])
                p_plus_PY, pi_minus_PY = np.meshgrid(p_plus_PY_all[p_plus_cum - p_plus_count:p_plus_cum],
                                                     pi_minus_PY_all[pi_minus_cum - pi_minus_count:pi_minus_cum])
                p_plus_PZ, pi_minus_PZ = np.meshgrid(p_plus_PZ_all[p_plus_cum - p_plus_count:p_plus_cum],
                                                     pi_minus_PZ_all[pi_minus_cum - pi_minus_count:pi_minus_cum])
                # Obliczanie mas niezmienniczych dla wszystkich kombinacji
                values_to_fill = np.sqrt(
                    (p_plus_E + pi_minus_E) ** 2 - (p_plus_PX + pi_minus_PX) ** 2 - (p_plus_PY + pi_minus_PY) ** 2 - (
                            p_plus_PZ + pi_minus_PZ) ** 2)
                # Redukowanie wymiaru macierzy jeśli tablice kombinacji nie są wektorami
                if p_plus_count > 1 and pi_minus_count > 1:
                    values_to_fill = np.concatenate(values_to_fill, axis=0)
                # Wypełniane histogramu masy dla pary cząstek ppi
                mass_hists[1].FillN(values_to_fill.size, values_to_fill, np.ones(values_to_fill.size))

            # Jeśli w danym zdarzeniu zarejestrowano przynajmniej 1 pi+ i przynajmniej 1 anty-p
            if pi_plus_count != 0 and p_minus_count != 0:
                # Tworzone są tablice 2x2 wwszystkich kombinacji energii i współrzędnych pędu czątek pi+ i anty-p
                # zarejestrowanych w tym zdarzeniu
                pi_plus_E, p_minus_E = np.meshgrid(pi_plus_E_all[pi_plus_cum - pi_plus_count:pi_plus_cum],
                                                   p_minus_E_all[p_minus_cum - p_minus_count:p_minus_cum])
                pi_plus_PX, p_minus_PX = np.meshgrid(pi_plus_PX_all[pi_plus_cum - pi_plus_count:pi_plus_cum],
                                                     p_minus_PX_all[p_minus_cum - p_minus_count:p_minus_cum])
                pi_plus_PY, p_minus_PY = np.meshgrid(pi_plus_PY_all[pi_plus_cum - pi_plus_count:pi_plus_cum],
                                                     p_minus_PY_all[p_minus_cum - p_minus_count:p_minus_cum])
                pi_plus_PZ, p_minus_PZ = np.meshgrid(pi_plus_PZ_all[pi_plus_cum - pi_plus_count:pi_plus_cum],
                                                     p_minus_PZ_all[p_minus_cum - p_minus_count:p_minus_cum])
                # Obliczanie mas niezmienniczych dla wszystkich kombinacji
                values_to_fill = np.sqrt(
                    (pi_plus_E + p_minus_E) ** 2 - (pi_plus_PX + p_minus_PX) ** 2 - (pi_plus_PY + p_minus_PY) ** 2 - (
                            pi_plus_PZ + p_minus_PZ) ** 2)
                # Redukowanie wymiaru macierzy jeśli tablice kombinacji nie są wektorami
                if pi_plus_count > 1 and p_minus_count > 1:
                    values_to_fill = np.concatenate(values_to_fill, axis=0)
                # Wypełniane histogramu masy dla pary cząstek ppi
                mass_hists[1].FillN(values_to_fill.size, values_to_fill, np.ones(values_to_fill.size))

        xehs = 5

    @profile
    def calculate_efficiency_pt_1(self):
        """
        Funkcja obliczająca czystość identyfikacji w funkcji pędu poprzecznego
        """

        # Tworzenie osi pędów poprzecznych
        pt = np.linspace(0, 2000, 51)

        # Wyciągnięcie z danych interesujących nas zmiennych
        data = self.data[
            ['piplus_TRUEID', 'piplus_PT', 'piplus_ProbNNk', 'piplus_ProbNNp', 'piplus_ProbNNpi']]

        # Sortowanie danych wraz z rosnącym pędem poprzecznym. Kolejność jest ważna.
        data_pt = data.sort_values('piplus_PT')

        # Stworzenie tablic interesujących nas zmiennych w celu późniejszych szybszych obliczeń
        ProbNNpi = data_pt['piplus_ProbNNpi'].values
        ProbNNp = data_pt['piplus_ProbNNp'].values
        ProbNNK = data_pt['piplus_ProbNNk'].values
        TRUEID = data_pt['piplus_TRUEID'].values
        PT = data_pt['piplus_PT'].values

        # Inicjalizujemy 50 binów do wykresów z zerami jako wszystkimi wartościami
        pi_values = np.zeros(pt.size - 1)
        p_values = np.zeros(pt.size - 1)
        K_values = np.zeros(pt.size - 1)

        # Inicjalizujemy zmienne liczące cząstki
        pi_count = 0
        pi_true_count = 0
        p_count = 0
        p_true_count = 0
        K_count = 0
        K_true_count = 0

        # Inicjalizujemy zmienną liczącą biny
        idx = 0

        # Iteracja po wszystkich cząstkach
        for number in range(TRUEID.size):
            # Jeśli pęd poprzeczny cząstki jest większy od górnej granicy aktualnego binu
            if PT[number] > pt[idx + 1]:
                # Zapisujemy zliczone w aktualnym binie dane upewniając się, że nie dzielimy przez 0. Jeśli w danym
                # binie nie było szukanego rodzaju cząstki nic nie jest wpisywane, ponieważ biny miały
                # domyślnie wartości 0
                if pi_true_count > 0:
                    pi_values[idx] = pi_count / pi_true_count
                if p_true_count > 0:
                    p_values[idx] = p_count / p_true_count
                if K_true_count > 0:
                    K_values[idx] = K_count / K_true_count
                # Przesunięcie do następnego binu
                idx += 1
                # Wyzerowanie zmiennych liczących
                pi_count = 0
                pi_true_count = 0
                p_count = 0
                p_true_count = 0
                K_count = 0
                K_true_count = 0
            # Przerwanie pętli po przejściu przez wszystkie biny
            if idx == pt.size - 1:
                break
            # Zliczanie cząstek
            if abs(TRUEID[number]) == 211:
                pi_true_count += 1
                if ProbNNpi[number] > 0.9:
                    pi_count += 1
            elif abs(TRUEID[number]) == 321:
                K_true_count += 1
                if ProbNNK[number] > 0.9:
                    K_count += 1
            elif abs(TRUEID[number]) == 2212:
                p_true_count += 1
                if ProbNNp[number] > 0.9:
                    p_count += 1

        # Tworzenie wykresów
        graph_pt_pi = ROOT.TGraph(pt.size - 1, pt / 1000, pi_values)
        graph_pt_pi.SetTitle("#pi identification purity;P_{t} [GeV];Efficiency")
        graph_pt_p = ROOT.TGraph(pt.size - 1, pt / 1000, p_values)
        graph_pt_p.SetTitle("p identification purity;P_{t} [GeV];Efficiency")
        graph_pt_K = ROOT.TGraph(pt.size - 1, pt / 1000, K_values)
        graph_pt_K.SetTitle("K identification purity;P_{t} [GeV];Efficiency")

        # Zapisywanie wykresów
        self._save_histogram(graph_pt_pi, f'{self.results_path}/efficiency/efficiency_pt_pi_1.png')
        self._save_histogram(graph_pt_p, f'{self.results_path}/efficiency/efficiency_pt_p_1.png')
        self._save_histogram(graph_pt_K, f'{self.results_path}/efficiency/efficiency_pt_K_1.png')

    @profile
    def calculate_efficiency_eta_1(self):
        """
        Funkcja obliczająca czystość identyfikacji w funkcji pseudopośpieszności
        """

        # Tworzenie osi pseudopośpieszności
        eta = np.linspace(2, 5, 26)

        # Wyciągnięcie z danych interesujących nas zmiennych
        data = self.data[
            ['piplus_TRUEID', 'piplus_ETA', 'piplus_ProbNNk', 'piplus_ProbNNp', 'piplus_ProbNNpi']]

        # Sortowanie danych wraz z rosnącą pseudopośpiesznością. Kolejność jest ważna.
        data_eta = data.sort_values('piplus_ETA')

        # Stworzenie tablic interesujących nas zmiennych w celu późniejszych szybszych obliczeń
        ProbNNpi = data_eta['piplus_ProbNNpi'].values
        ProbNNp = data_eta['piplus_ProbNNp'].values
        ProbNNK = data_eta['piplus_ProbNNk'].values
        TRUEID = data_eta['piplus_TRUEID'].values
        ETA = data_eta['piplus_ETA'].values

        # Inicjalizujemy 50 binów do wykresów z zerami jako wszystkimi wartościami
        pi_values = np.zeros(eta.size - 1)
        p_values = np.zeros(eta.size - 1)
        K_values = np.zeros(eta.size - 1)

        # Inicjalizujemy zmienne liczące cząstki
        pi_count = 0
        pi_true_count = 0
        p_count = 0
        p_true_count = 0
        K_count = 0
        K_true_count = 0

        # Inicjalizujemy zmienną liczącą biny
        idx = 0

        # Iteracja po wszystkich cząstkach
        for number in range(TRUEID.size):
            # Jeśli pseudopośpieszność cząstki jest większa od górnej granicy aktualnego binu
            if ETA[number] > eta[idx + 1]:
                # Zapisujemy zliczone w aktualnym binie dane upewniając się, że nie dzielimy przez 0. Jeśli w danym
                # binie nie było szukanego rodzaju cząstki nic nie jest wpisywane, ponieważ biny miały
                # domyślnie wartości 0
                if pi_true_count > 0:
                    pi_values[idx] = pi_count / pi_true_count
                if p_true_count > 0:
                    p_values[idx] = p_count / p_true_count
                if K_true_count > 0:
                    K_values[idx] = K_count / K_true_count
                # Przesunięcie do następnego binu
                idx += 1
                # Wyzerowanie zmiennych liczących
                pi_count = 0
                pi_true_count = 0
                p_count = 0
                p_true_count = 0
                K_count = 0
                K_true_count = 0
            # Przerwanie pętli po przejściu przez wszystkie biny
            if idx == eta.size - 1:
                break
            # Zliczanie cząstek
            if abs(TRUEID[number]) == 211:
                pi_true_count += 1
                if ProbNNpi[number] > 0.9:
                    pi_count += 1
            elif abs(TRUEID[number]) == 321:
                K_true_count += 1
                if ProbNNK[number] > 0.9:
                    K_count += 1
            elif abs(TRUEID[number]) == 2212:
                p_true_count += 1
                if ProbNNp[number] > 0.9:
                    p_count += 1

        # Tworzenie wykresów
        graph_eta_pi = ROOT.TGraph(eta.size - 1, eta, pi_values)
        graph_eta_pi.SetTitle("#pi identification purity;#eta;Efficiency")
        graph_eta_p = ROOT.TGraph(eta.size - 1, eta, p_values)
        graph_eta_p.SetTitle("p identification purity;#eta;Efficiency")
        graph_eta_K = ROOT.TGraph(eta.size - 1, eta, K_values)
        graph_eta_K.SetTitle("K identification purity;#eta;Efficiency")

        # Zapisywanie wykresów
        self._save_histogram(graph_eta_pi, f'{self.results_path}/efficiency/efficiency_eta_pi_1.png')
        self._save_histogram(graph_eta_p, f'{self.results_path}/efficiency/efficiency_eta_p_1.png')
        self._save_histogram(graph_eta_K, f'{self.results_path}/efficiency/efficiency_eta_K_1.png')

    @profile
    def calculate_efficiency_pt_2(self):
        """
        Funkcja obliczająca wydajności kryteriów selekcji w funkcji pędu poprzecznego
        """

        # Tworzenie osi pędów poprzecznych
        pt = np.linspace(0, 2000, 51)

        # Wyciągnięcie z danych interesujących nas zmiennych
        data = self.data[
            ['piplus_TRUEID', 'piplus_PT', 'piplus_ProbNNk', 'piplus_ProbNNp', 'piplus_ProbNNpi']]

        # Sortowanie danych wraz z rosnącym pędem poprzecznym. Kolejność jest ważna.
        data_pt = data.sort_values('piplus_PT')

        # Stworzenie tablic interesujących nas zmiennych w celu późniejszych szybszych obliczeń
        ProbNNpi = data_pt['piplus_ProbNNpi'].values
        ProbNNp = data_pt['piplus_ProbNNp'].values
        ProbNNK = data_pt['piplus_ProbNNk'].values
        TRUEID = data_pt['piplus_TRUEID'].values
        PT = data_pt['piplus_PT'].values

        # Inicjalizujemy 50 binów do wykresów z zerami jako wszystkimi wartościami
        pi_values = np.zeros(pt.size - 1)
        p_values = np.zeros(pt.size - 1)
        K_values = np.zeros(pt.size - 1)

        # Inicjalizujemy zmienne liczące cząstki
        pi_count = 0
        pi_true_count = 0
        p_count = 0
        p_true_count = 0
        K_count = 0
        K_true_count = 0

        # Inicjalizujemy zmienną liczącą biny
        idx = 0

        # Iteracja po wszystkich cząstkach
        for number in range(TRUEID.size):
            # Jeśli pęd poprzeczny cząstki jest większy od górnej granicy aktualnego binu
            if PT[number] > pt[idx + 1]:
                # Zapisujemy zliczone w aktualnym binie dane upewniając się, że nie dzielimy przez 0. Jeśli w danym
                # binie nie było szukanego rodzaju cząstki nic nie jest wpisywane, ponieważ biny miały
                # domyślnie wartości 0
                if pi_true_count > 0:
                    pi_values[idx] = pi_true_count / pi_count
                if p_true_count > 0:
                    p_values[idx] = p_true_count / p_count
                if K_true_count > 0:
                    K_values[idx] = K_true_count / K_count
                # Przesunięcie do następnego binu
                idx += 1
                # Wyzerowanie zmiennych liczących
                pi_count = 0
                pi_true_count = 0
                p_count = 0
                p_true_count = 0
                K_count = 0
                K_true_count = 0
            # Przerwanie pętli po przejściu przez wszystkie biny
            if idx == pt.size - 1:
                break
            # Zliczanie cząstek
            if ProbNNpi[number] > 0.9:
                pi_count += 1
                if abs(TRUEID[number]) == 211:
                    pi_true_count += 1
            elif ProbNNK[number] > 0.9:
                K_count += 1
                if abs(TRUEID[number]) == 321:
                    K_true_count += 1
            elif ProbNNp[number] > 0.9:
                p_count += 1
                if abs(TRUEID[number]) == 2212:
                    p_true_count += 1

        # Tworzenie wykresów
        graph_pt_pi = ROOT.TGraph(pt.size - 1, pt / 1000, pi_values)
        graph_pt_pi.SetTitle("#pi cut efficiency;P_{t} [GeV];Efficiency")
        graph_pt_p = ROOT.TGraph(pt.size - 1, pt / 1000, p_values)
        graph_pt_p.SetTitle("p cut efficiency;P_{t} [GeV];Efficiency")
        graph_pt_K = ROOT.TGraph(pt.size - 1, pt / 1000, K_values)
        graph_pt_K.SetTitle("K cut efficiency;P_{t} [GeV];Efficiency")

        # Zapisywanie wykresów
        self._save_histogram(graph_pt_pi, f'{self.results_path}/efficiency/efficiency_pt_pi_2.png')
        self._save_histogram(graph_pt_p, f'{self.results_path}/efficiency/efficiency_pt_p_2.png')
        self._save_histogram(graph_pt_K, f'{self.results_path}/efficiency/efficiency_pt_K_2.png')

    @profile
    def calculate_efficiency_eta_2(self):
        """
        Funkcja obliczająca wydajności kryteriów selekcji w funkcji pseudopośpieszności
        """

        # Tworzenie osi pseudopośpieszności
        eta = np.linspace(2, 5, 26)

        # Wyciągnięcie z danych interesujących nas zmiennych
        data = self.data[
            ['piplus_TRUEID', 'piplus_ETA', 'piplus_ProbNNk', 'piplus_ProbNNp', 'piplus_ProbNNpi']]

        # Sortowanie danych wraz z rosnącą pseudopośpiesznością. Kolejność jest ważna.
        data_eta = data.sort_values('piplus_ETA')

        # Stworzenie tablic interesujących nas zmiennych w celu późniejszych szybszych obliczeń
        ProbNNpi = data_eta['piplus_ProbNNpi'].values
        ProbNNp = data_eta['piplus_ProbNNp'].values
        ProbNNK = data_eta['piplus_ProbNNk'].values
        TRUEID = data_eta['piplus_TRUEID'].values
        ETA = data_eta['piplus_ETA'].values

        # Inicjalizujemy 50 binów do wykresów z zerami jako wszystkimi wartościami
        pi_values = np.zeros(eta.size - 1)
        p_values = np.zeros(eta.size - 1)
        K_values = np.zeros(eta.size - 1)

        # Inicjalizujemy zmienne liczące cząstki
        pi_count = 0
        pi_true_count = 0
        p_count = 0
        p_true_count = 0
        K_count = 0
        K_true_count = 0

        # Inicjalizujemy zmienną liczącą biny
        idx = 0

        # Iteracja po wszystkich cząstkach
        for number in range(TRUEID.size):
            # Jeśli pseudopośpieszność cząstki jest większa od górnej granicy aktualnego binu
            if ETA[number] > eta[idx + 1]:
                # Zapisujemy zliczone w aktualnym binie dane upewniając się, że nie dzielimy przez 0. Jeśli w danym
                # binie nie było szukanego rodzaju cząstki nic nie jest wpisywane, ponieważ biny miały
                # domyślnie wartości 0
                if pi_true_count > 0:
                    pi_values[idx] = pi_true_count / pi_count
                if p_true_count > 0:
                    p_values[idx] = p_true_count / p_count
                if K_true_count > 0:
                    K_values[idx] = K_true_count / K_count
                # Przesunięcie do następnego binu
                idx += 1
                # Wyzerowanie zmiennych liczących
                pi_count = 0
                pi_true_count = 0
                p_count = 0
                p_true_count = 0
                K_count = 0
                K_true_count = 0
            # Przerwanie pętli po przejściu przez wszystkie biny
            if idx == eta.size - 1:
                break
            # Zliczanie cząstek
            if ProbNNpi[number] > 0.9:
                pi_count += 1
                if abs(TRUEID[number]) == 211:
                    pi_true_count += 1
            elif ProbNNK[number] > 0.9:
                K_count += 1
                if abs(TRUEID[number]) == 321:
                    K_true_count += 1
            elif ProbNNp[number] > 0.9:
                p_count += 1
                if abs(TRUEID[number]) == 2212:
                    p_true_count += 1

        # Tworzenie wykresów
        graph_eta_pi = ROOT.TGraph(eta.size - 1, eta, pi_values)
        graph_eta_pi.SetTitle("#pi cut efficiency;#eta;Efficiency")
        graph_eta_p = ROOT.TGraph(eta.size - 1, eta, p_values)
        graph_eta_p.SetTitle("p cut efficiency;#eta;Efficiency")
        graph_eta_K = ROOT.TGraph(eta.size - 1, eta, K_values)
        graph_eta_K.SetTitle("K cut efficiency;#eta;Efficiency")

        # Zapisywanie wykresów
        self._save_histogram(graph_eta_pi, f'{self.results_path}/efficiency/efficiency_eta_pi_2.png')
        self._save_histogram(graph_eta_p, f'{self.results_path}/efficiency/efficiency_eta_p_2.png')
        self._save_histogram(graph_eta_K, f'{self.results_path}/efficiency/efficiency_eta_K_2.png')
