def split_data(file_path, train_file_path, validation_file_path):
    # 读取原始文件的所有行
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 计算训练集的行数（70%）
    train_size = int(len(lines) * 0.7)

    # 将前70%的行写入训练集文件
    with open(train_file_path, 'w', encoding='utf-8') as train_file:
        train_file.writelines(lines[:train_size])

    # 将剩余的30%的行写入验证集文件
    with open(validation_file_path, 'w', encoding='utf-8') as validation_file:
        validation_file.writelines(lines[train_size:])

# 调用函数，传入文件路径和输出文件路径
# 请替换以下路径为您的实际文件路径
split_data('toutiao_cat_data.txt', 'train.txt', 'validation.txt')