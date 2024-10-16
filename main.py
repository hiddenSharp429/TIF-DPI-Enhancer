'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-14 20:03:47
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-16 22:17:06
FilePath: /TIFF DPI Enhancer/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 