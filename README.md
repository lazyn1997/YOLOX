学习 YOLOX: Exceeding YOLO Series in 2021 论文，跑通原文demo代码及验证代码，并尝试使用飞桨框架复现。
# 复现赛流程

![](https://lazynnote.oss-cn-shenzhen.aliyuncs.com/typora/image-20210902090946941.png)

# 原论文代码实现

## 评估

* YOLOX-s 模型评估

```shell
python tools/eval.py -n  yolox-s -c weights/yolox_s.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
```

![image-20210906221549801](https://lazynnote.oss-cn-shenzhen.aliyuncs.com/typora/image-20210906221549801.png)![image-20210906221825351](https://lazynnote.oss-cn-shenzhen.aliyuncs.com/typora/image-20210906221825351.png)![image-20210906221335530](https://lazynnote.oss-cn-shenzhen.aliyuncs.com/typora/image-20210906221335530.png)

```shell
python tools/eval.py -n  yolox-s -c weights/yolox_s.pth -b 64 -d 8 --conf 0.001 [--fp16] [--fuse]
```

![image-20210906222920174](https://lazynnote.oss-cn-shenzhen.aliyuncs.com/typora/image-20210906222920174.png)![image-20210906222942514](https://lazynnote.oss-cn-shenzhen.aliyuncs.com/typora/image-20210906222942514.png)![image-20210906223003962](https://lazynnote.oss-cn-shenzhen.aliyuncs.com/typora/image-20210906223003962.png)

* YOLOX-Darknet53 模型评估

```shell
python tools/eval.py -n  yolov3 -c weights/yolox_darknet.pth -b 32 -d 1 --conf 0.001 [--fp16] [--fuse]
```

![image-20210906223205780](https://lazynnote.oss-cn-shenzhen.aliyuncs.com/typora/image-20210906223205780.png)![image-20210906223312883](https://lazynnote.oss-cn-shenzhen.aliyuncs.com/typora/image-20210906223312883.png)![image-20210906223337128](https://lazynnote.oss-cn-shenzhen.aliyuncs.com/typora/image-20210906223337128.png)

* 原论文结果

| Model                                                        | size | mAPval 0.5:0.95 | mAPtest 0.5:0.95 | Speed V100 (ms) | Params (M) | FLOPs (G) | weights                                                      |
| ------------------------------------------------------------ | ---- | --------------- | ---------------- | --------------- | ---------- | --------- | ------------------------------------------------------------ |
| [YOLOX-s](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_s.py) | 640  | 40.5            | 40.5             | 9.8             | 9.0        | 26.8      | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
| [YOLOX-m](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_m.py) | 640  | 46.9            | 47.2             | 12.3            | 25.3       | 73.8      | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
| [YOLOX-l](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_l.py) | 640  | 49.7            | 50.1             | 14.5            | 54.2       | 155.6     | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
| [YOLOX-x](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_x.py) | 640  | 51.1            | **51.5**         | 17.3            | 99.1       | 281.9     | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |
| [YOLOX-Darknet53](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolov3.py) | 640  | 47.7            | 48.0             | 11.1            | 63.7       | 185.3     | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth) |

***对比发现模型成功跑通，AP 值符合原论文结果。***

## 预测

```shell
python tools/demo.py image -n yolov3 -c weights/yolox_darknet.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
```

![image-20210906225440765](https://lazynnote.oss-cn-shenzhen.aliyuncs.com/typora/image-20210906225440765.png)

* 预测结果

![dog](https://lazynnote.oss-cn-shenzhen.aliyuncs.com/typora/dog.jpg)

***从预测图可以看出，不同类别被正确框选出来，且预测正确，具有较高的置信度。***

# 复现结果
* 训练语句基本跑通，按照官方
