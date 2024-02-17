# @Author       : Duhongkai
# @Time         : 2024/1/23 11:50
# @Description  : rouge通过rouge_chinese进行计算

from rouge_chinese import Rouge


# 数据输入类型[[1,2,3,4,5],[6,7,8,9,10]]
def get_rouge_l(preds, labels):
    labels_str = [str(label)[1:-1].replace(',', '') for label in labels]
    preds_str = [str(pred)[1:-1].replace(',', '') for pred in preds]
    rouge = Rouge()
    rouge_scores = rouge.get_scores(preds_str, labels_str)
    # 计算均值
    rouge_l = []
    for single_score in rouge_scores:
        rouge_l.append(single_score["rouge-l"]["f"])
    return sum(rouge_l) / len(rouge_l)
