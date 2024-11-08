# list_windows.py

import pygetwindow as gw

def list_windows():
    windows = gw.getAllTitles()
    for win in windows:
        print(win)

if __name__ == "__main__":
    list_windows()
