import fasttext

def test_classfication():
    # 1 训练分类模型
    model = fasttext.train_supervised(input='/Users/circleLee/Develop/workspace_py/nlp/week05/cooking.stackexchange.txt',
                                        lr=0.05,
                                        dim= 100,
                                        ws=5,
                                        epoch= 5)
    # 2 预测
    result = model.predict("How much does potato starch affect a cheese sauce recipe?")
    print('result--->1 ', result)
    # 元组中的第一项代表标签, 第二项代表对应的概率
    # (('__label__baking',), array([0.0735101]))

    # 3 模型保存
    model.save_model("./model_cooking.bin")
    print('模型保存 ok')

    # 4 模型加载
    mymodel = fasttext.load_model("./model_cooking.bin")
    print('模型加载 ok')

if __name__ == '__main__':
    test_classfication()