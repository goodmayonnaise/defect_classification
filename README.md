### data structure
```
./data
├─aug1
│  ├─test
│  │  ├─defect
│  │  │  ├─ defect
│  │  │  └─ false
│  │  └─false
│  │  │  ├─ defect
│  │  │  └─ false
│  ├─train
│  │  ├─defect
│  │  └─false
│  └─val
│      ├─defect
│      └─false
    .
    .
    .
└─split2
    ├─test
    │  ├─defect
    │  └─false
    ├─train
    │  ├─defect
    │  └─false
    └─val
        ├─defect
        └─false

```


### train 
```python train.py data\split2 --arch "efficientnet-b3" --save-dir weights\efficientnet-b3 --epochs 100 --batch-size 4 --pretrained --image-size 300 --lr 0.001 --warmup 5 ```

### heatmap 
```python heatmap.py data\split2 --arch "efficientnet-b3" --weight weights\efficientnet-b3\best_model.pth.tar --batch-size 1 --gpu 0```
```
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 0 PRED : 0 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 0 PRED : 0 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 0 PRED : 0 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 0 PRED : 0 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 0 PRED : 1 | correct : False
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 0 PRED : 1 | correct : False
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif    | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif | GT : 1 PRED : 1 | correct : True
[SAVE] weights\efficientnet-b3-aug1\data\aug1\test\heatmap\.tif        | GT : 1 PRED : 1 | correct : True
Done.

### inference
```python inference.py data\split2 --arch "efficientnet-b3" --weight weights\efficientnet-b3\best_model.pth.tar --batch-size 1 --image-size 300```
```
=> creating model 'efficientnet-b3'

Using image size 300
target:defect   pred:defect     path:data/aug1/test\defect\.tif    True
target:defect   pred:defect     path:data/aug1/test\defect\.tif    True
target:defect   pred:defect     path:data/aug1/test\defect\.tif    True
target:defect   pred:defect     path:data/aug1/test\defect\.tif    True
target:defect   pred:false      path:data/aug1/test\defect\.tif    False
target:defect   pred:false      path:data/aug1/test\defect\.tif    False
target:false    pred:false      path:data/aug1/test\false\.tif     True
target:false    pred:false      path:data/aug1/test\false\.tif     True
target:false    pred:false      path:data/aug1/test\false\.tif     True
target:false    pred:false      path:data/aug1/test\false\.tif     True
target:false    pred:false      path:data/aug1/test\false\.tif     True
target:false    pred:false      path:data/aug1/test\false\.tif     True
target:false    pred:false      path:data/aug1/test\false\.tif     True
target:false    pred:false      path:data/aug1/test\false\.tif     True
target:false    pred:false      path:data/aug1/test\false\.tif     True
target:false    pred:false      path:data/aug1/test\false\.tif     True
target:false    pred:false      path:data/aug1/test\false\.tif     True
target:false    pred:false      path:data/aug1/test\false\.tif  True
target:false    pred:false      path:data/aug1/test\false\.tif  True
target:false    pred:false      path:data/aug1/test\false\.tif  True
target:false    pred:false      path:data/aug1/test\false\.tif  True
target:false    pred:false      path:data/aug1/test\false\.tif  True
target:false    pred:false      path:data/aug1/test\false\.tif  True
target:false    pred:false      path:data/aug1/test\false\.tif  True
target:false    pred:false      path:data/aug1/test\false\.tif  True
target:false    pred:false      path:data/aug1/test\false\.tif True
 * Acc@1 92.308

./weights/efficientnet-b3-aug1 weight result
epoch : 21
validation accuracy : 96.0
test accuracy : 92.30769348144531
save dir : .\weights\efficientnet-b3-aug1\data\aug1\test/res1.txt
```
