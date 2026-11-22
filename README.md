车牌识别课程提交文件

评测从仓库根目录导入 `pipeline.run(image_path)`。该函数把检测、矫正、颜色、分割和识别串成一次调用，返回可序列化的结果字典。

```python
from pipeline import run
result = run("samples/normal_blue.png")
```

发布页：https://github.com/bawangloong/plate-recognition-submit/releases/tag/v1.0

1. A_plate_detection(1).zip
https://github.com/bawangloong/plate-recognition-submit/releases/download/v1.0/A_plate_detection_1.zip

2. plate_char_C(1).zip
https://github.com/bawangloong/plate-recognition-submit/releases/download/v1.0/plate_char_C_1.zip

3. 车牌识别系统_D部分_完整版含exe.zip
https://github.com/bawangloong/plate-recognition-submit/releases/download/v1.0/plate_D_full_with_exe.zip