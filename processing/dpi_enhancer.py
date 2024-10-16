'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-14 20:05:38
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-16 21:43:35
FilePath: /TIFF DPI Enhancer/processing/dpi_enhancer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE%E8%AE%BE%E7%BD%AE
'''
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from processing.image_segmentation import ImageSegmenter

class DPIEnhancer:
    def __init__(self): 
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel("models/EDSR_x4.pb")
        self.sr.setModel("edsr", 4)
        self.segmenter = ImageSegmenter()

    def enhance(self, image, content_mask, original_dpi, target_dpi, interpolation_method):
        # 图像分割
        text_mask, image_mask = self.segmenter.segment_image(image)
        
        print(f"Original image shape: {image.shape}")
        print(f"Text mask shape: {text_mask.shape}")
        print(f"Image mask shape: {image_mask.shape}")

        # 计算缩放因子
        scale_factor = target_dpi / original_dpi

        # 对文字区域进行OCR和重新渲染
        enhanced_text = self.enhance_text_with_ocr(image, text_mask, original_dpi, target_dpi)
        print(f"Enhanced text shape: {enhanced_text.shape}")
        print(f"Enhanced text dtype: {enhanced_text.dtype}")
        print(f"Enhanced text min-max values: {enhanced_text.min()}-{enhanced_text.max()}")

        # 对图像区域进行增强
        enhanced_image_area = self.enhance_image_area(image, image_mask, original_dpi, target_dpi)
        print(f"Enhanced image area shape: {enhanced_image_area.shape}")
        print(f"Enhanced image area dtype: {enhanced_image_area.dtype}")
        print(f"Enhanced image area min-max values: {enhanced_image_area.min()}-{enhanced_image_area.max()}")

        # 确保两个图像具有相同的尺寸
        if enhanced_text.shape != enhanced_image_area.shape:
            enhanced_text = cv2.resize(enhanced_text, (enhanced_image_area.shape[1], enhanced_image_area.shape[0]))

        # 合并增强后的区域
        enhanced_image = cv2.addWeighted(enhanced_text, 1, enhanced_image_area, 1, 0)
        
        # 应用阈值，将灰色区域转换为黑白
        _, enhanced_image = cv2.threshold(enhanced_image, 128, 255, cv2.THRESH_BINARY)

        print(f"Combined enhanced image shape: {enhanced_image.shape}")
        print(f"Combined enhanced image dtype: {enhanced_image.dtype}")
        print(f"Combined enhanced image min-max values: {enhanced_image.min()}-{enhanced_image.max()}")

        # 检查是否有全白的情况
        if np.all(enhanced_image == 255):
            print("Warning: The enhanced image is all white!")

        # 如果需要进一步缩放以达到目标DPI
        if scale_factor != 4:  # EDSR模型是4倍放大
            height, width = enhanced_image.shape[:2]
            new_height = int(height * (scale_factor / 4))
            new_width = int(width * (scale_factor / 4))
            enhanced_image = cv2.resize(enhanced_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        return enhanced_image

    def enhance_text_with_ocr(self, image, text_mask, original_dpi, target_dpi):
        # 对文字区域应用OCR
        text_area = cv2.bitwise_and(image, image, mask=text_mask)
        gray_text = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY)
        
        print(f"Text area shape: {text_area.shape}")
        print(f"Gray text shape: {gray_text.shape}")
        print(f"Gray text min-max values: {gray_text.min()}-{gray_text.max()}")

        # 使用pytesseract获取文字信息，包括边界框
        text_data = pytesseract.image_to_data(gray_text, output_type=pytesseract.Output.DICT)
        
        print(f"Number of detected words: {len(text_data['text'])}")
        print(f"Sample words: {text_data['text'][:5]}")  # 打印前5个检测到的单词

        # 估计原始字体大小
        estimated_font_size = self.estimate_font_size(text_data)
        print(f"Estimated font size: {estimated_font_size}")

        # 创建一个新的高分辨率图像用于渲染文字
        scale_factor = target_dpi / original_dpi
        new_height, new_width = int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor)
        enhanced_text = np.zeros((new_height, new_width, 3), dtype=np.uint8)  # 创建黑色背景

        # 使用高分辨率字体重新渲染文字
        pil_image = Image.fromarray(enhanced_text)
        draw = ImageDraw.Draw(pil_image)
        
        words_rendered = 0
        for i, word in enumerate(text_data['text']):
            if int(text_data['conf'][i]) > 60 and word.strip():  # 只渲染置信度高于60%的非空文字
                font_size = max(int(estimated_font_size * scale_factor), 12)  # 确保字体大小至少为12
                font = self.get_default_font(font_size)
                x = int(text_data['left'][i] * scale_factor)
                y = int(text_data['top'][i] * scale_factor)
                draw.text((x, y), word, font=font, fill=(255, 255, 255))  # 白色文字
                words_rendered += 1

        print(f"Number of words rendered: {words_rendered}")

        enhanced_text = np.array(pil_image)

        print(f"Enhanced text array shape: {enhanced_text.shape}")
        print(f"Enhanced text array dtype: {enhanced_text.dtype}")
        print(f"Enhanced text array min-max values: {enhanced_text.min()}-{enhanced_text.max()}")

        # 如果enhanced_text全黑，尝试直接使用放大的原始文本区域
        if enhanced_text.max() == 0:
            print("Warning: Enhanced text is all black. Using scaled original text area.")
            enhanced_text = cv2.resize(text_area, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        return enhanced_text

    def get_default_font(self, font_size):
        try:
            # 尝试使用 TrueType 字体
            font = ImageFont.truetype(font="arial.ttf", size=font_size)
        except IOError:
            try:
                # 如果 arial.ttf 不
                font = ImageFont.truetype(font="arial.ttf", size=font_size)
            except IOError:
                # 如果仍然失败，使用默认位图字体
                font = ImageFont.load_default()
                print("Warning: Using default bitmap font. Text quality may be poor.")
        return font

    def estimate_font_size(self, text_data):
        heights = [text_data['height'][i] for i in range(len(text_data['text'])) 
                   if text_data['text'][i].strip() and int(text_data['conf'][i]) > 60]
        if heights:
            return int(np.median(heights))  # 使用中位数作为估计值
        else:
            return 12  # 如果无法估计，则返回默认值

    def enhance_image_area(self, image, image_mask, original_dpi, target_dpi):
        # 对图像区域应用EDSR模型
        image_area = cv2.bitwise_and(image, image, mask=image_mask)
        enhanced_image = self.sr.upsample(image_area)
        
        print(f"Enhanced image area shape: {enhanced_image.shape}")
        print(f"Enhanced image area dtype: {enhanced_image.dtype}")
        print(f"Enhanced image area min-max values: {enhanced_image.min()}-{enhanced_image.max()}")

        # 如果需要，进一步调整大小以匹配目标DPI
        scale_factor = target_dpi / original_dpi
        if scale_factor != 4:  # EDSR模型是4倍放大
            height, width = enhanced_image.shape[:2]
            new_height = int(height * (scale_factor / 4))
            new_width = int(width * (scale_factor / 4))
            enhanced_image = cv2.resize(enhanced_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        return enhanced_image

    def sharpen_text(self, image):
        # 使用锐化滤镜增强文字清晰度
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)

    def save_image(self, image, output_path, dpi):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_path, dpi=(dpi, dpi))
