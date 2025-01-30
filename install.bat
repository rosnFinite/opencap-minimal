@echo off

REM Erstelle die conda Umgebung aus der environment.yml
echo Erstelle conda-Umgebung aus environment.yml...
call conda env create -f environment.yml

REM Aktivieren der Umgebung
echo Aktivieren der conda-Umgebung...
call conda activate mmpose

REM Installiere die restlichen pip-Pakete (openmim, mmengine, mmcv) mit den entsprechenden Argumenten
echo Installiere openmim, mmengine und mmcv...
call pip install --upgrade openmim
call pip install mmengine
call pip install mmcv==2.1.0 --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
call mim install "mmdet>=3.1.0"
call mim install "mmpose>=1.1.0"

REM Skript abgeschlossen
echo Installation abgeschlossen!
pause