from inference import * 

def read_and_classify(file_path):
    # 定义分类code与名称的映射
    categories = {
        '100': 'news_story',
        '101': 'news_culture',
        '102': 'news_entertainment',
        '103': 'news_sports',
        '104': 'news_finance',
        '106': 'news_house',
        '107': 'news_car',
        '108': 'news_edu',
        '109': 'news_tech',
        '110': 'news_military',
        '112': 'news_travel',
        '113': 'news_world',
        '114': 'stock',
        '115': 'news_agriculture',
        '116': 'news_game'
    }

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 移除行尾的换行符并分割字段
            fields = line.strip().split('_!_')
            if len(fields) < 5:
                continue  # 如果字段不足5个，则跳过这条数据

            # 提取新闻ID，分类code，分类名称，新闻字符串和新闻关键词
            news_id, category_code, category_name, news_title, news_keywords = fields[:5]
            
            # 构建prompt
            prompt = f"新闻标题：{news_title}\n新闻关键词：{news_keywords}\n, 它的类别应该是民生、文化、娱乐、体育、财经、房产、汽车、教育、科技、军事、旅游、国际、证券、农业、电竞之一。你觉得它的类别是"
            
            # 使用大语言模型进行分类
            # 假设有一个名为`classify_news`的函数，用于调用大语言模型进行分类
            input = [[w for w in prompt]]
            ret = top_k_inc(lm_model, lm_vocab, device, input, 10, max_len)
            print('大预言模型预测结果：', ret)
            # 这里我们模拟分类结果
            category_result = categories.get(category_code, '未知分类')
            
            # 输出分类结果
            print(f"新闻ID：{news_id}\n分类结果：{category_result}\n")
            # 这里可以添加代码将结果保存到文件或数据库

# 调用函数，传入.txt文件的路径
# read_and_classify('path_to_your_file.txt')


if __name__ == "__main__":
    device = 0
    print("loading...")
    m_path = "/mnt/share/xujing/nuaa_nlp/ckpt_sft/epoch80_batch_309999"
    v_path = "/mnt/share/xujing/nuaa_nlp/model/vocab.txt"
    lm_model, lm_vocab, lm_args = init_model(m_path, device, v_path)
    print("done.")

    max_len = 200
    # qs = ["介绍下南京航空航天大学", "Please introduce Nanjing University of Aeronautics and Astronautics"]
    prompt = []
    
    val_set_path = "/mnt/share/xujing/nuaa_nlp/toutiao-text-classfication-dataset/toutiao_cat_data.txt"
    
    read_and_classify(val_set_path)

    # r4 = top_k_inc(lm_model, lm_vocab, device, input, 10, max_len)
