# MAC-VO - Metrics-aware Covariance for Learning-based Stereo Visual Odometry

## Uruchomienie Systemu Wizualnej Odometrii

### Podstawowe polecenie

```bash
python3 MACVO.py --useRR --odom [ODOM_CONFIG] --data [DATA_CONFIG]
```

### Opis parametrów

- `--useRR` - Aktywuje wizualizator Rerun (generuje plik `.rrd` dla wizualizacji 3D w czasie rzeczywistym)
- `--odom [ODOM_CONFIG]` - Ścieżka do pliku konfiguracyjnego sieci neuronowej (folder: `Config/Experiment/MACVO/`)
- `--data [DATA_CONFIG]` - Ścieżka do pliku konfiguracyjnego datasetu (folder: `Config/Sequence/`)

### Dostępne konfiguracje

**Konfiguracje odometrii:**
- `Config/Experiment/MACVO/MACVO_Fast.yaml` - Szybka wersja (12.5 fps na 480x640)
- `Config/Experiment/MACVO/MACVO_Performant.yaml` - Wersja wydajna (7 fps)
- `Config/Experiment/MACVO/Paper_Reproduce.yaml` - Konfiguracja z publikacji

**Datasetów testowych:**
- `Config/Sequence/TartanAir_abandonfac_001.yaml`
- `Config/Sequence/EuRoC_MH01.yaml`
- `Config/Sequence/KITTI.yaml`
- `Config/Sequence/TartanAir_example.yaml`

## Przykłady uruchamiania

### Przykład 1: Szybka bez wizualizacji
```bash
python3 MACVO.py --odom Config/Experiment/MACVO/MACVO_Fast.yaml --data Config/Sequence/TartanAir_abandonfac_001.yaml
```
Uruchamiaystem w szybkim trybie na datasecie TartanAir. Generuje trajektorię i metryki oceny.

### Przykład 2: Wydajna wersja z wizualizacją
```bash
python3 MACVO.py --useRR --odom Config/Experiment/MACVO/MACVO_Performant.yaml --data Config/Sequence/TartanAir_abandonfac_001.yaml
```
Uruchamia system w trybie wydajnym z włączoną wizualizacją RerunVisualizer. Generuje plik `.rrd` do wizualizacji w czasie rzeczywistym oraz trajektorię.

### Przykład 3: Publikacja z danym zakresem ramek
```bash
python3 MACVO.py --useRR --odom Config/Experiment/MACVO/Paper_Reproduce.yaml --data Config/Sequence/EuRoC_MH01.yaml --seq_from 0 --seq_to 500
```
Uruchamia system z konfiguracją publikacji na EuRoC MH01, przetwarzając tylko 500 ramek od ramki 0.

### Przykład 4: Z preloadowaniem danych i zapisem czasomierza
```bash
python3 MACVO.py --useRR --odom Config/Experiment/MACVO/MACVO_Fast.yaml --data Config/Sequence/KITTI.yaml --preload --timing
```
Uruchamia system z preloadowaniem całej sekwencji do RAM (szybsze przetwarzanie) i aktywuje rejestrowanie czasów wykonania.

## Dodatkowe opcje

- `--seq_from [N]` - Początkowa ramka sekwencji (domyślnie: 0)
- `--seq_to [N]` - Końcowa ramka sekwencji do przetworzenia
- `--resultRoot [PATH]` - Katalog do zapisu wyników (domyślnie: `./Results`)
- `--saveplt` - Zapisywanie wizualizacji kowariantności jako jpg dla poszczególnych ramek
- `--preload` - Preloadowanie całej trajektorii do RAM (zmniejsza overhead I/O)
- `--autoremove` - Automatyczne czyszczenie folderu wyników po ukończeniu (przydatne do testowania)
- `--noeval` - Pomijanie ewaluacji sekwencji po uruchomieniu odometrii
- `--timing` - Aktywuje rejestrowanie czasów wykonania dla systemu

## Wyniki
Wyniki i wizualizacje zapisywane są w folderze `Results` w strukturze:
```
Results/
  MACVO-[Config_Name]@[Dataset_Name]/
    [TIMESTAMP]/                   # Czasowy znaczek uruchomienia
      poses.npy          # Oszacowana trajektoria
      ref_poses.npy      # Ground Truth trajektoria (jeśli dostępne w datasecie)
      elapsed_time.json  # Czasy wykonania (jeśli --timing)
      [Project_Name].rrd # Wizualizacja (jeśli --useRR)
      config.yaml        # Konfiguracja eksperymentu
```

**Ważne:** Za każdym uruchomieniem MACVO.py tworzy **nowy folder z aktualnym timestampem**. Poprzednie wyniki nigdy nie będą nadpisane - możesz bezpiecznie porównywać wyniki między różnymi uruchomieniami!

## Wizualizacja z Ground Truth

Ground Truth jest **automatycznie zapisywany** w pliku `ref_poses.npy` jeśli dataset zawiera pozycje referencyjne.

### Opcja 1: Porównanie trajektorii na PNG
```bash
PYTHONPATH=/home/macvo/workspace python3 Evaluation/PlotSeq.py --spaces Results/MACVO-Performant@abf001/02_07_165334/
```
Generuje pliki PNG z obydwiema trajektoriami (Ground Truth + estymaty) oraz analizą błędów.

**Wygenerowane pliki (w folderze `Results/`):**
- `MACVO-Performant@abf001_Trajectory.png` - Wizualizacja 3D trajektorii (GT + estymaty)
- `MACVO-Performant@abf001_TranslationErr.png` - Błędy translacji
- `MACVO-Performant@abf001_RotationErr.png` - Błędy rotacji
- `Combined_RTEcdf.png` - Kumulative Translation Error
- `Combined_ROEcdf.png` - Cumulative Rotation Error

> **Ważne:** 
> - Uruchamiaj polecenia z głównego katalogu `/home/macvo/workspace`
> - Zamień TIMESTAMP (np. `02_07_165334`) na rzeczywisty znaczek czasowy z folderu Results
> - Użyj `--recursive` flag aby przeszukać strukturę z timestampami: `--recursive Results/MACVO-Performant@abf001/`

### Opcja 2: Wizualizacja w Rerun z Ground Truth
Uruchom najpierw MACVO.py z wizualizacją:
```bash
python3 MACVO.py --useRR --odom Config/Experiment/MACVO/MACVO_Fast.yaml --data Config/Sequence/TartanAir_abandonfac_001.yaml
```

Następnie w Rerun UI otwórz plik `.rrd`:
```bash
rerun Results/MACVO-Fast@abf001/MACVO-Fast@abf001.rrd
```

Alternatywnie - załaduj istniejące wyniki w Pythonie:
```python
from Utility.Trajectory import Trajectory
from Utility.Sandbox import Sandbox

# Załaduj wyniki z konkretnym timestampem
sandbox = Sandbox.load("Results/MACVO-Performant@abf001/02_07_165334/")
gt_traj, est_traj = Trajectory.from_sandbox(sandbox)

# gt_traj  - Ground Truth trajektoria
# est_traj - Oszacowana trajektoria
```

## Dostępne wyniki (abandonfac_001)

- `Results/MACVO-Fast@abf001/` - Szybka wersja na TartanAir abandonfac_001
- `Results/MACVO-Performant@abf001/02_07_163521/` - Wydajna wersja (run 1)
- `Results/MACVO-Performant@abf001/02_07_165334/` - Wydajna wersja (run 2)

Każdy folder zawiera:
- `poses.npy` - estymaty pozycji
- `ref_poses.npy` - ground truth pozycje
- `*.rrd` - wizualizacja Rerun (jeśli wygenerowana z --useRR)
- `config.yaml` - konfiguracja eksperymentu
