# 1-wymiarowy histogram zmiennej PID
PID = {
    # Kryterium selekcji
    "cutoff": 0.1,
    # Liczba binów
    "nBins": 50,
    # Wartość minimalna
    "xmin": 0,
    # Wartość maksymalna
    "xmax": 100
}

# 1-wymiarowy histogram zmiennej ProbNN
ProbNN = {
    "cutoff": 0.9,
    "nBins": 50,
    "xmin": 0.9,
    "xmax": 1
}

# 2-wymiarowy histogram zmiennych PID i ProbNN
PID_ProbNN = {
    # Kryterium selekcji 1 zmiennej
    "cutoff_1": 0.1,
    # Kryterium selekcji 2 zmiennej
    "cutoff_2": 0.9,
    # Liczba binów 1 zmiennej
    "nBins_1": 50,
    # Liczba binów 2 zmiennej
    "nBins_2": 50,
    # Wartość minimalna 1 zmiennej
    "xmin_1": 0,
    # Wartość minimalna 2 zmiennej
    "xmax_1": 100,
    # Wartość maksymalna 1 zmiennej
    "xmin_2": 0.9,
    # Wartość maksymalna 2 zmiennej
    "xmax_2": 1
}

# 2-wymiarowy histogram zmiennych PID i ProbNNpi
PID_ProbNNpi = {
    "cutoff_1": 0.1,
    "cutoff_2": 0.9,
    "nBins_1": 50,
    "nBins_2": 50,
    "xmin_1": 0,
    "xmax_1": 100,
    "xmin_2": 0,
    "xmax_2": 1
}

# 2-wymiarowy histogram pędu poprzecznego i ProbNN
ProbNN_m = {
    "cutoff_1": 0,
    "cutoff_2": 0.9,
    "nBins_1": 50,
    "nBins_2": 50,
    "xmin_1": 0,
    "xmax_1": 2000,
    "xmin_2": 0.9,
    "xmax_2": 1
}

# 2-wymiarowy histogram pseudopośpieszności i ProbNN
ProbNN_eta = {
    "cutoff_1": 0,
    "cutoff_2": 0.9,
    "nBins_1": 50,
    "nBins_2": 50,
    "xmin_1": 2,
    "xmax_1": 5,
    "xmin_2": 0.9,
    "xmax_2": 1
}

# ID czstkek pi
pipi = {
    "id": [211]
}

# ID cząstek pi i p
ppi = {
    "id": [211, 2212]
}

# ID cząstek K
KK = {
    "id": [321]
}
