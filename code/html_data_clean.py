from bs4 import BeautifulSoup
import json

with open('flower_name.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, 'html.parser')

per_flower_sum_dict = {}
flower_index_to_name=[]
flower_label_to_name={}

# 查找所有的<tr>标签
for tr in soup.find_all('tr'):
    labL = tr.find('td', class_='labL').get_text(strip=True)
    numL = tr.find('td', class_='numL').get_text(strip=True)
    labC = tr.find('td', class_='labC').get_text(strip=True)
    numC = tr.find('td', class_='numC').get_text(strip=True)
    labR = tr.find('td', class_='labR').get_text(strip=True)
    numR = tr.find('td', class_='numR').get_text(strip=True)
    per_flower_sum_dict[labL] = numL
    per_flower_sum_dict[labC] = numC
    per_flower_sum_dict[labR] = numR
    print(f"Label: {labL}, Number: {numL}")
    print(f"Label: {labC}, Number: {numC}")
    print(f"Label: {labR}, Number: {numR}")
    flower_index_to_name.append(labL)
    flower_index_to_name.append(labC)
    flower_index_to_name.append(labR)
    print("flower_index_to_name",flower_index_to_name)

for label in range(1,103,1):
        flower_label_to_name[label]=flower_index_to_name[label-1]

# 注意：这里使用json格式保存，因为它易于人类和机器阅读
with open('per_flower_sum.json', 'w', encoding='utf-8') as file:
    json.dump(per_flower_sum_dict, file, ensure_ascii=False, indent=4)
with open('flower_label_to_name.json', 'w', encoding='utf-8') as file:
    json.dump(flower_label_to_name, file, ensure_ascii=False, indent=4)
