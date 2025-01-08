# OpenCap Minimal

OpenCap Minimal ist eine reduzierte Version des Open-Source-Frameworks [OpenCap](https://github.com/stanfordnmbl/opencap-core), das präzise und zugängliche 
Bewegungsanalysen aus Videos von Standard-Smartphone-Kameras ermöglicht. Es wurde vom Stanford Neuromuscular 
Biomechanics Laboratory entwickelt, um detaillierte 3D-Motion-Capture-Daten ohne teure oder komplexe Hardware zu 
erstellen. 

Dieses Repository stellt die minimalen notwendigen Skripte bereit, um die Hauptpipeline von OpenCap lokal auszuführen 
(Videos -> 3D-Pose). Es eignet sich für Benutzer, die OpenCap lokal testen oder integrieren möchten.

## Features
- Lokale Ausführung der OpenCap-Pipeline.
- Enthält die notwendigen Skripte und Konfigurationsdateien.
- Bietet Integrationsmöglichkeit für MMPose Modellen.

## Installation
Die Installation erfordert Anaconda oder miniconda als Paketmanager sowie spezifische Builds von ffmpeg und OpenPose. 
Die folgenden Schritte leiten dich durch die Installation:

### 1. Installiere ```Anaconda``` oder ```miniconda```
   1. Herunterladen:
      1. Lade den Graphical Installer von [Anaconda](https://www.anaconda.com/download/success) oder [Miniconda](https://www.anaconda.com/download/success#miniconda) herunter.
      > Hinweis: Miniconda ist eine schlankere Alternative zu Anaconda und wird empfohlen, da nur die wesentlichen Komponenten wie Conda und Python installiert werden. Weitere Details findest du im [Vergleich von Anaconda und Miniconda](https://docs.anaconda.com/distro-or-miniconda/).
   2. Installation:
      1. Folge den Installationsanweisungen des jeweiligen Installers.
   3. Überprüfe die erfolgreiche Installation
      1. Öffne die **Eingabeaufforderung** und gib folgendes ein:
      ```conda --version```
      2. Eine erfolgreiche Installation gibt die installierte Conda-Version aus.

### 2. Installiere spezifische ```ffmpeg``` und ```openpose``` Builds
> Da die OpenCap-Pipeline mit bestimmten Versionen getestet wurde, sollten diese verwendet werden.
1. Lade die spezifischen Versionen von ```ffmpeg``` und ```openpose``` aus dem bereitgestellten [Google Drive](https://drive.google.com/drive/folders/17ihUjaKsc8vwzOuzKWIMndNz_Z7Odm4N) herunter.
2. Entpacke die heruntergeladenen Dateien und benenne die Ordner in ```ffmpeg``` und ```openpose``` um.
3. Verschiebe die Ordner nach C:\\, sodass die Verzeichnisse ```C:\ffmpeg``` und ```C:\openpose``` existieren.
4. Füge ```ffmpeg``` zu den System-PATH-Variablen hinzu:
   1. Drücke die Windows-Taste und suche nach **Umgebungsvariablen bearbeiten**.
   2. Im Fenster *Systemeigenschaften* klicke am unteren rechten Rand auf die Schaltfläche **Umgebungsvariablen...**.
   3. Im Fenster *Umgebungsvariablen* suche in der Liste der **Systemvariablen** nach **Path** und bearbeite sie.
   4. Im Fenster *Umgebungsvariablen bearbeiten* klicke auf **Neu** und füge den Pfad ```C:\ffmpeg\bin``` hinzu.

### 3. Installiere Visual Studio "Desktopentwicklung mit C++"
1. Lade den [Visual Studio Installer](https://visualstudio.microsoft.com/de/vs/community/) herunter.
2. Führe den Installer aus und wähle während der Installation das Workload **Desktopentwicklung mit C++** aus.

### 4. Erstellen der Entwicklungsumgebung
> Für eine schnelle Einrichtung kannst du die bereitgestellte YAML-Umgebungsdatei verwenden, um 
> eine Anaconda-Umgebung zu erstellen.

1. ```conda env create -f opencap_env.yml```
2. Aktiviere die Umgebung in deiner IDE oder über die Kommandozeile:
```conda activate opencap_min```


## Verwendung von Modellen aus dem MMPose Model Zoo
Aufgrund von Versionskonflikten ist eine direkte Integration von MMPose in OpenCap derzeit nicht möglich. Um die 
Funktionalität von MMPose dennoch in die OpenCap-Pipeline einzubinden, wurde die folgende Lösung implementiert:

**Lösung: Separate Anaconda-Umgebung für MMPose-Inferenz**

Für die Ausführung der MMPose-Inferenz wird eine separate Anaconda-Umgebung erstellt, die unabhängig von der 
Haupt-OpenCap-Umgebung arbeitet. Während der OpenCap-Pipeline wird im Pose Estimation Schritt ein Subprozess gestartet, 
der die MMPose-Inferenz in dieser dedizierten Umgebung ausführt.

Vorteile dieser Lösung:
- Isolierung der Abhängigkeiten: Konflikte zwischen den Abhängigkeiten von OpenCap und MMPose werden vermieden.
- Modularität: Die MMPose-Umgebung kann unabhängig aktualisiert oder modifiziert werden, ohne die OpenCap-Hauptumgebung 
zu beeinträchtigen.
- Flexibilität: Ermöglicht die Nutzung der neuesten Modelle aus dem MMPose Model Zoo, ohne die OpenCap-Kompatibilität 
zu gefährden.

### Erstellung der MMPose-Umgebung
1. Conda-Umgebung mit Python 3.8 erstellen: ```conda create --name mmpose python=3.8 -y```
2. Umgebung aktivieren: ```conda activate mmpose```
3. PyTorch installieren (auf korrekte Versionen achten): ```conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=12.1 fsspec -c pytorch -c nvidia -c conda-forge -y```
4. Installation OpenMMLab-Pakete:
   1. Installation von MIM, OpenMMLab Paketmanager (ermöglicht das einfache installieren von Modellen): ```pip install -U openmim```
   2. Installation von MMEngine: ```pip install mmengine```
   3. Installation MMCV: ```pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html```
5. Installation von MMPose: ```mim install "mmpose>=1.1.0"```
6. Überprüfe korrekte Installation (in ```mmpose``` Umgebung):
   1. ```python demo/image_demo.py ./mmpose/demo/demo.jpg ./mmpose/demo/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py ./mmpose/demo/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth --out-file vis_results.jpg --draw-heatmap```
   Führt testweise eine 2D Posenerkennung anhand des zuvor heruntergeladenen Modells durch.
   2. Es sollte die Datei ```vis_results.jpg``` mit dargestellten Keypoints im Stammverzeichnis des Projekts sichtbar sein.

