def wrap_text(input_file, output_file, line_length=60):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    import textwrap
    wrapped_text = '\n'.join(textwrap.fill(line, width=line_length)
                              for line in text.split('\n'))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(wrapped_text)

# 使用示例
input_path = "./text.txt"
output_path = "./text_wrapped.txt"
wrap_text(input_path, output_path)
