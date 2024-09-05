# Vision Language 



## CLIP: Learning Transferable Visual Models From Natural Language Supervision


摘要翻译:

    SOTA的CV系统通常训练在一个封闭的分类集上，这种有监督学习的限制会导致对于其他任何概念都需要追加新的标签(label)。从图像的原始文字描述(Raw text about images)直接学习图片的描述似乎是一个更加广泛的监督方法。本工作建立了一个 "预测哪一个标题对应哪一个图片" 的简单预训练任务，来高效且可扩展的对网络上收集的400million个图像文字pair进行训练，并证明了学习到的图像表征的有效性。自然语言的描述作为图像输入的输出，在下游任务中，展现出了与完全监督baseline相同的性能。


观点:
* 对比目标 (contrastive objectives) 可以学习到比 预测目标(predictive objective) 更好的表征
* Generative models (生成模型) 同样可以学习高质量的图像表征, 但通常需要比 对比模型 高一个数量级的参数量


方法:
* 学习过程, 采样 N 个 pair 进行单次学习. Text Encoder 和 Image Encoder 分别对文本和图片进行编码, 到相同维度的特征量
  * 使 N 个正确的匹配的 cosine similarity 尽可能大
  * N^2 -N 个错误匹配的 cosine similarity 尽可能小
* 推理过程 (图片分类)
  * 定义一个 text 模板 `a photo of a {object}`
  * 从 label 库提取 object 可选项
  * 填入的 text 作为 Text Encoder 的输入, 和 Image Encoder 的输出进行匹配

```py
# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
```