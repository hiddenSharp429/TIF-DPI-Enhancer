'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-14 20:04:59
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-14 21:56:59
FilePath: /TIFF DPI Enhancer/processing/tif_reader.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE%E8%AE%BE%E7%BD%AE
'''
from PIL import Image
import numpy as np

class TIFReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def get_dpi(self):
        with Image.open(self.file_path) as img:
            dpi = img.info.get('dpi', (72, 72))
        return dpi[0]  # Assuming horizontal and vertical DPI are the same

    def read_image(self):
        with Image.open(self.file_path) as img:
            # Convert image to RGB mode if it's not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
