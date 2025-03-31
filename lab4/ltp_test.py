from ltp import LTP

ltp = LTP()
sentence = "作为哈尔滨工业大学的学生，张三和李四前往武汉进行社会实践活动。"

result = ltp.pipeline([sentence], tasks = ["cws","ner"])
print(result.ner)