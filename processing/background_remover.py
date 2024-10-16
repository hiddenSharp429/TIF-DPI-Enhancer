'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-14 20:05:15
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-14 20:05:25
FilePath: /TIFF DPI Enhancer/processing/background_remover.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np

class BackgroundRemover:
    def remove_background(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 使用形态学操作来改善mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask