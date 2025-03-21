import jieba

with open('./text.txt', errors='ignore', encoding='gb2312') as fp:
   lines = fp.readlines()
   for line in lines:
       seg_list = jieba.cut(line)
       with open('./data/seg_text.txt', 'a', encoding='utf-8') as ff:
           ff.write(' '.join(seg_list))