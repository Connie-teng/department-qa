# 讀取原始檔案
with open("docs/教師介紹.txt", "r", encoding="utf-8-sig") as f:
    content = f.read()

# 把每位教授的格式標準化
import re
lines = content.split("\n")
output = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    # 偵測教授姓名行（中文名字+英文名）
    if re.match(r'^[\u4e00-\u9fff]{2,4}\(', line):
        output.append(f"\n教師姓名：{line}")
    elif line.startswith("職稱:"):
        output.append(line.replace("職稱:", "職稱："))
    elif line.startswith("學歷:"):
        output.append(line.replace("學歷:", "學歷："))
    elif line.startswith("研究領域:"):
        output.append(line.replace("研究領域:", "研究領域："))
    elif line.startswith("辦公室:"):
        output.append(line.replace("辦公室:", "辦公室："))
    else:
        output.append(line)

result = "\n".join(output)
with open("docs/教師介紹.txt", "w", encoding="utf-8-sig") as f:
    f.write(result)

print("完成！")