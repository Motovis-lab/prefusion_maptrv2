import pickle
import numpy
import json

pkl_path = "1698819176.664.pkl"
txt_path = "1698819176.664.txt"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

with open(txt_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"✅ 已保存为格式化文本 {txt_path}")
