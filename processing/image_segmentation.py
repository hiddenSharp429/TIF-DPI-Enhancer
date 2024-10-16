'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-15 22:42:24
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-16 22:58:22
FilePath: /TIFF DPI Enhancer/processing/image_segmentation.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE%E5%8E%9F%E5%85%B3
'''
import cv2
import numpy as np

class ImageSegmenter:
    def __init__(self):
        pass

    def segment_image(self, image):
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 文字区域检测
        text_mask = self.detect_text(gray)
        
        # 图像区域检测
        image_mask = self.detect_image_areas(image, gray)
        
        # 确保区域不重叠
        image_mask[text_mask > 0] = 0
        
        return text_mask, image_mask

    def detect_text(self, gray_image):
        # 多尺度文本检测
        scales = [0.5, 1.0, 1.5]
        text_masks = []

        for scale in scales:
            # 调整图像大小
            resized = cv2.resize(gray_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            
            # 使用自适应阈值处理
            binary = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # 边缘检测
            edges = cv2.Canny(resized, 100, 200)
            
            # 组合二值图像和边缘
            combined = cv2.bitwise_or(binary, edges)
            
            # 使用形态学操作来连接相近的文本区域
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            connected = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            
            # 移除小的噪点
            denoised = self.remove_small_regions(connected, min_size=10)
            
            # 将结果调整回原始大小
            mask = cv2.resize(denoised, (gray_image.shape[1], gray_image.shape[0]), interpolation=cv2.INTER_LINEAR)
            text_masks.append(mask)

        # 合并所有尺度的结果
        final_mask = np.max(text_masks, axis=0)
        
        # 再次应用形态学操作以连接相近区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        return final_mask

    def detect_image_areas(self, image, gray_image):
        # 使用Canny边缘检测
        edges = cv2.Canny(gray_image, 100, 200)
        
        # 使用膨胀操作连接边缘
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建图像区域掩码
        image_mask = np.zeros_like(gray_image)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 过滤小区域
                # 计算轮廓的复杂度
                perimeter = cv2.arcLength(contour, True)
                complexity = perimeter / np.sqrt(area)
                
                # 如果轮廓不太复杂，可能是图像区域
                if complexity < 15:
                    cv2.drawContours(image_mask, [contour], 0, (255), -1)
        
        # 使用颜色信息进一步细化图像区域
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:,:,1]
        high_saturation = saturation > 100
        image_mask[high_saturation] = 255
        
        return image_mask

    def remove_small_regions(self, mask, min_size=100):
        # 标记连通区域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 移除小区域
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                mask[labels == i] = 0
        
        return mask

    def refine_masks(self, text_mask, image_mask, original_image):
        # 使用颜色信息进一步细化掩码
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:,:,1]
        
        # 假设高饱和度区域更可能是图像
        high_saturation = saturation > 100
        image_mask[high_saturation] = 255
        text_mask[high_saturation] = 0
        
        # 移除小的孤立区域
        text_mask = self.remove_small_regions(text_mask, min_size=50)
        image_mask = self.remove_small_regions(image_mask, min_size=100)
        
        return text_mask, image_mask
