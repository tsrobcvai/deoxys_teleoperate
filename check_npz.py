import numpy as np
import os

folder = "./demos_collected/run001"
for fname in os.listdir(folder):
    if fname.endswith(".npz"):
        print(f"\n检查文件: {fname}")
        arr = np.load(os.path.join(folder, fname))["data"]
        print("shape:", arr.shape)
        # 检查是否全为0
        if np.all(arr == 0):
            print("警告：该文件所有数据全为0！")
        else:
            # 检查每一行是否全为0
            zero_rows = np.where(~arr.any(axis=-1))[0]
            if len(zero_rows) > 0:
                print(f"有 {len(zero_rows)} 行全为0（索引示例：{zero_rows[:10]}）")
            else:
                print("数据正常，未发现全为0的行。")
        # 可选：打印前几行数据
        print("前2行数据示例：\n", arr[:2])