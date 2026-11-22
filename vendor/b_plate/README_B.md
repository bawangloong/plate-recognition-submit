# B 模块：车牌颜色识别 + 字符分割

大作业《车牌识别系统》中 **B 同学**负责的部分：把一块车牌拆成按顺序排列的字符图片，
输出颜色、字符数量和分割状态，交给 C（字符识别）和 D（GUI）。

## 一、目录结构

```
plate_B/
├── plate_color.py        颜色识别
├── segment_chars.py      字符分割（含连通域、投影、粘连拆分）
├── run_demo.py           批量演示：生成四联过程图 + 汇总表
├── make_test_plates.py   生成合成车牌（自测用，A 的真实数据到位前使用）
├── samples/              合成测试图（蓝/黄/绿/白/黑，含 7 位、8 位、倾斜）
├── output/               运行结果：*_panel.png（四联图）、*_chars/（字符切片）、summary.csv
├── README.md
└── report_B.md           报告正文（可直接并入大作业报告）
```

## 二、依赖

只需要两个库，**不需要 OpenCV**：

```
numpy
Pillow
```

> 选择说明：字符分割用的 Otsu、形态学、连通域、投影都能用 numpy + Pillow 实现，
> 省掉 OpenCV 后 D 同学用 PyInstaller 打包会小很多，离线部署也更稳。
> 若老师要求统一用 cv2，只需把 `otsu_threshold / morph / connected_components`
> 换成 `cv2.threshold / cv2.morphologyEx / cv2.connectedComponentsWithStats`，其余逻辑不变。

## 三、运行

```bash
# 1) 生成合成测试图（首次运行）
python make_test_plates.py

# 2) 批量跑一遍，输出过程图与汇总
python run_demo.py

# 3) 单独测某张
python run_demo.py samples/blue_7.png

# 4) 只测颜色 / 只测分割
python plate_color.py samples/*.png
python segment_chars.py samples/*.png
```

## 四、接口（给 C、D 同学）

```python
import numpy as np
from PIL import Image
from plate_color import classify_plate_color, is_dark_text_on_light_plate
from segment_chars import segment_plate

img = np.array(Image.open("plate.png").convert("RGB"))   # RGB 数组即可

# --- 颜色 ---
c = classify_plate_color(img)
# {'color': 'blue'|'yellow'|'green'|'white'|'black'|'unknown',
#  'label': '蓝牌', 'confidence': 1.0, 'detail': {...}}

# --- 分割 ---
res = segment_plate(
    img,
    prefer_dark_text=is_dark_text_on_light_plate(c["color"]),  # 黄/绿/白牌是深字
    debug=True,
)
# res['chars']  : [numpy.uint8]，每张 40(高) × 20(宽)，按从左到右排序
# res['boxes']  : [(x, y, w, h)]  在归一化图坐标系中的位置
# res['count']  : 字符个数
# res['status'] : 'ok'（7 或 8 个） / 'count_abnormal' / 'no_char'
# res['stages'] : 中间过程图（debug=True 时）norm / gray / bw / segs / polarity
```

**约定**（已与 C 商量好）：
- 字符图统一 **40 × 20（高 × 宽）**，灰度、**白色为笔画、黑色为背景**（`uint8`，0/255）
- 排序为**从左到右**，不含中间小圆点
- 一个字符一张图，C 直接喂给特征提取即可

## 五、方法概要

**颜色识别**：转 HSV → 统计饱和像素的色相分布（蓝 190–260°、黄 20–65°、绿 65–190°）→
占比过半即判定；饱和度不足时按亮度分白牌/黑牌。输出置信度，难以判断时返回 `unknown`。

**字符分割**：
1. 按高度归一化到 140 px（保持长宽比）
2. 灰度 → Otsu → 按前景占比自动确定极性（深字/浅字）
3. 形态学闭运算补断笔
4. 连通域分析：去除贴边块（边框）、小面积噪点、上下边缘的铆钉
5. 垂直投影切分 → 合并 ≤2 px 的断笔 → 过宽块按投影最小处拆分（处理字符粘连）
6. 按高度过滤小圆点，按宽度过滤噪点
7. 归一化到 40×20，按 x 排序输出

**7 位 / 8 位**：7 位为普通车牌（省份汉字 + 字母 + 5 位），8 位为新能源车牌。
本模块不区分位数，只输出"实际切出的字符个数"；字符数为 7 或 8 时状态记 `ok`，
其余记 `count_abnormal` 并保留切分结果，交由上层判断。

## 六、已知限制

1. **倾斜车牌需要 A 先矫正**。本模块假设输入是方向正常、字符完整的车牌。
   `samples/blue_tilt.png` 直接送入会只切出 5 个字符，这是预期结果（见 report_B.md 失败案例）。
2. 强反光、严重模糊、字符被污渍连成一片时，分割数量会异常，需要调 `morph` 的核大小。
3. 新能源绿牌有渐变底色，当前按"绿色"处理；若遇到渐变较深导致二值化不稳，
   可在 `segment_chars.py` 里改为按局部自适应阈值。
4. 英文字母 O/0、I/1 等形近字符不在此模块处理，属于 C 的识别问题。

## 七、验证记录

见 `output/summary.csv`，以及每张图对应的 `*_panel.png`（四联过程图）。
报告用的统计见 `report_B.md` 第 4 节。
