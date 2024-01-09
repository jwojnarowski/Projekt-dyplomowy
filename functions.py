import ROOT
from simulation import Simulation


def draw_combined_histograms(sim: Simulation, sim_true: Simulation):
    """
    Funkcja łączaca histogramy rekonstrukcji mas z danych symulacyjnych i histogramów mas z danych doświadczalnych
    """

    # Obiekt Simulation zwraca histogramy rekonstrukcji mas z danych symulacyjnych
    mass_pipi, mass_ppi, mass_KK = sim()
    # Obiekt Simulation zwraca histogramy mas z danych doświadczalnych
    mass_pipi_true, mass_ppi_true, mass_KK_true = sim_true()
    # Tworzenie obiekty TCanvas
    c = ROOT.TCanvas('c', 'c', 375, 350)
    # Modyfikacje wizualne
    c.SetLeftMargin(0.15)
    c.SetRightMargin(0.15)
    c.SetBottomMargin(0.125)
    c.SetTopMargin(0.125)
    # Czynnik do jakiego normalizowane będą histogramy
    norm = 1

    # Histogramy dla pary cząstek pipi
    # Normalizacja histogramów
    scale = norm / mass_pipi.Integral()
    mass_pipi.Scale(scale)
    scale = norm / mass_pipi_true.Integral()
    mass_pipi_true.Scale(scale)
    # Modyfikacje wizualne
    mass_pipi.SetStats(0)
    mass_pipi.GetYaxis().SetTitle("normalized events")
    mass_pipi.GetXaxis().SetTitleSize(0.05)
    mass_pipi.GetYaxis().SetTitleSize(0.05)
    mass_pipi.GetXaxis().SetLabelSize(0.041)
    mass_pipi.GetYaxis().SetLabelSize(0.041)
    mass_pipi_true.SetStats(0)
    mass_pipi_true.GetYaxis().SetTitle("normalized events")
    mass_pipi_true.GetXaxis().SetTitleSize(0.05)
    mass_pipi_true.GetYaxis().SetTitleSize(0.05)
    mass_pipi_true.GetXaxis().SetLabelSize(0.041)
    mass_pipi_true.GetYaxis().SetLabelSize(0.041)
    mass_pipi_true.SetLineColor(632)
    c.Draw()
    mass_pipi.Draw('h')
    mass_pipi_true.Draw("SAME, h")
    legend = ROOT.TLegend(0.6, 0.85, 1, 1)
    legend.AddEntry(mass_pipi, "m_{#pi #pi} MC reconstruction", "l")
    legend.AddEntry(mass_pipi_true, "m_{#pi #pi} true data", "l")
    legend.SetTextSize(0.035)
    legend.Draw()
    # Zapisanie histogramu
    c.Print(f"{sim.results_path}/combined_histograms/mass_pipi_comb.png")

    # Histogramy dla pary cząstek ppi
    # Normalizacja histogramów
    scale = norm / mass_ppi.Integral()
    mass_ppi.Scale(scale)
    scale = norm / mass_ppi_true.Integral()
    mass_ppi_true.Scale(scale)
    # Modyfikacje wizualne
    mass_ppi.SetStats(0)
    mass_ppi.GetYaxis().SetTitle("normalized events")
    mass_ppi.GetXaxis().SetTitleSize(0.05)
    mass_ppi.GetYaxis().SetTitleSize(0.05)
    mass_ppi.GetXaxis().SetLabelSize(0.041)
    mass_ppi.GetYaxis().SetLabelSize(0.041)
    mass_ppi_true.SetStats(0)
    mass_ppi_true.GetYaxis().SetTitle("normalized events")
    mass_ppi_true.GetXaxis().SetTitleSize(0.05)
    mass_ppi_true.GetYaxis().SetTitleSize(0.05)
    mass_ppi_true.GetXaxis().SetLabelSize(0.041)
    mass_ppi_true.GetYaxis().SetLabelSize(0.041)
    mass_ppi_true.SetLineColor(632)
    c.Draw()
    mass_ppi.Draw('h')
    mass_ppi_true.Draw("SAME, h")
    legend = ROOT.TLegend(0.60, 0.85, 1, 1)
    legend.AddEntry(mass_ppi, "m_{p #pi} MC reconstruction", "l")
    legend.AddEntry(mass_ppi_true, "m_{p #pi} true data", "l")
    legend.SetTextSize(0.035)
    legend.Draw()
    # Zapisanie histogramu
    c.Print(f"{sim.results_path}/combined_histograms/mass_ppi_comb.png")

    # Histogramy dla pary cząstek KK
    # Normalizacja histogramów
    scale = norm / mass_KK.Integral()
    mass_KK.Scale(scale)
    scale = norm / mass_KK_true.Integral()
    mass_KK_true.Scale(scale)
    # Modyfikacje wizualne
    mass_KK.SetStats(0)
    mass_KK.GetYaxis().SetTitle("normalized events")
    mass_KK.GetXaxis().SetTitleSize(0.05)
    mass_KK.GetYaxis().SetTitleSize(0.05)
    mass_KK.GetXaxis().SetLabelSize(0.041)
    mass_KK.GetYaxis().SetLabelSize(0.041)
    mass_KK_true.SetStats(0)
    mass_KK_true.GetYaxis().SetTitle("normalized events")
    mass_KK_true.GetXaxis().SetTitleSize(0.05)
    mass_KK_true.GetYaxis().SetTitleSize(0.05)
    mass_KK_true.GetXaxis().SetLabelSize(0.041)
    mass_KK_true.GetYaxis().SetLabelSize(0.041)
    mass_KK_true.SetLineColor(632)
    c.Draw()
    mass_KK.Draw('h')
    mass_KK_true.Draw("SAME, h")
    legend = ROOT.TLegend(0.60, 0.85, 1, 1)
    legend.AddEntry(mass_KK, "m_{KK} MC reconstruction", "l")
    legend.AddEntry(mass_KK_true, "m_{KK} true data", "l")
    legend.SetTextSize(0.035)
    legend.Draw()
    # Zapisanie histogramu
    c.Print(f"{sim.results_path}/combined_histograms/mass_KK_comb.png")
