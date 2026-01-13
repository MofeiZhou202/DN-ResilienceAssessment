import numpy as np

from src.config import GENERATED_DATA_DIR

# 1. 读取 npz 文件；如果路径不同自行替换
data = np.load(GENERATED_DATA_DIR / 'test_dataset.npz', allow_pickle=True)

# 2. 查看里面有哪些键
print(data.files)