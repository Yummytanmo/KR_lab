import json
from openai import OpenAI
from API import API_DS, API_QWEN
import os
os.environ.pop("SSL_CERT_FILE", None)


def generator(name, system, prompt):
    if name == 'deepseek':
        API = API_DS
        url = "https://api.deepseek.com"
        model = "deepseek-chat"
    elif name == 'qwen':
        API = API_QWEN
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = "qwen-plus"
    else:
        raise ValueError("Invalid model name. Choose 'deepseek' or 'qwen'.")
    client = OpenAI(api_key=API, base_url=url)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

def ner(name, sentence):
    system_prompt = (
        "你是一位专门进行命名实体识别的智能助手。请识别以下中文句子中的实体，实体类别包括：\n"
        "Number, 企业/品牌, 音乐专辑, 人物, 歌曲, 城市, 学校, Text, 影视作品, 地点, Date, 学科专业, 文学作品, 奖项, 电视综艺, 作品, 语言, 景点, 图书作品, 行政区, 企业, 国家, 气候。\n"
        "请严格按照以下 JSON 格式输出结果：\n"
        "{\n"
        '    "sentence": "原句",\n'
        '    "entities": [\n'
        '        ["实体类别", "实体值"],\n'
        '        ...\n'
        "    ]\n"
        "}\n"
        "如果没有识别出某一类别，则对应的 entities 数组中不应包含该项。"
    )
    user_prompt = f"请对下面这句话进行命名实体识别：\n\n{sentence}"
    response = generator(name, system_prompt, user_prompt)
    return response

def load_test_data(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    data = [json.loads(d)["text"] for d in lines]
    return data

if __name__ == "__main__":
    test_data = load_test_data("./duie_data/test_example.json")
    results = []
    # name = "qwen"
    name = "deepseek"
    for sentence in test_data:
        result = ner(name, sentence)
        results.append(result)
        
    with open(f"./results/{name}_results.json", "w", encoding="utf-8") as f:
        for result in results:
            f.write(result + "\n")
