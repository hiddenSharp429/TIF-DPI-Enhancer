<!--
 * @Author: hiddenSharp429 z404878860@163.com
 * @Date: 2024-10-14 20:06:58
 * @LastEditors: hiddenSharp429 z404878860@163.com
 * @LastEditTime: 2024-10-14 20:07:44
 * @FilePath: /TIFF DPI Enhancer/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# 图像处理工具

## 功能

1. 背景去除
2. DPI增强
3. 锐化
4. 伽马校正

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 背景去除

```bash
python processing/background_remover.py
```

2. DPI增强

```bash
python processing/dpi_enhancer.py
```

3. 锐化

```bash
python processing/sharpen_image.py
```

4. 伽马校正

```bash
python processing/gamma_correction.py
```