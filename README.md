# Category Dictionary Guided Unsupervised Domain Adaptation for Object Detection

### Data Preparation
Public datasets are available from official websites or mirrors. If your folder structure is different, you may need to change the corresponding paths in config files.

```text
scl
├── cfgs
├── lib
├── weights
├── data
│   ├── cityscape
│   │   ├── annos
│   │   ├── annos_multi
│   │   ├── imgs
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── train_car.txt
│   │   ├── val_car.txt
│   ├── foggycity
│   │   ├── annos
│   │   ├── annos_multi
│   │   ├── imgs
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── train_car.txt
│   │   ├── val_car.txt
|   ├── watercolor
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   │   ├── Main
│   │   │   |    ├── train.txt
│   │   │   |    ├── test.txt
│   │   ├── JPEGImages
|   ├── sim10k
│   │   ├── Annotations
│   │   ├── JPEGImages
│   │   ├── train.txt
│   │   ├── val.txt
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   │   ├── Annotations
│   │   │   ├── JPEGImages
│   │   │   ├── ImageSets
│   │   ├── VOC2012
│   │   │   ├── Annotations
│   │   │   ├── JPEGImages
│   │   │   ├── ImageSets
```
### Pretrained Model

We used two pretrained models in our experiments, VGG and ResNet101. You can download these two models from:

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and put them into the data/pretrained_model/.

#### CDG model

The trained checkpoints can be accessed by [Google drive](https://drive.google.com/drive/folders/1Zq4TtcwD1y2JfCnFPKh5BQP2U8ukxUhz?usp=sharing).

### Compilation

As the code is a little old, both pytorch-0.4.0 and pytorch-1.0 (or higher) are needed. 

Compile the cuda dependencies using following simple commands:

```
cd lib
# Under pytorch-0.4.0
sh make_1.sh 
# Under pytorch-1.0
sh make_2.sh
```
### Train
#### Get the annotator
To train a faster R-CNN model with vgg16 on sim10k, simply run:
```
 python train_source.py \
            --dataset sim10k --net vgg16 --bs $BATCH_SIZE
```

For citycape dataset, we first use cyclegan(https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to translate the source data into the style of target and then train a source model based on the new transferred source data.
```
 python train_source.py \
            --dataset cityscape --net vgg16 --bs $BATCH_SIZE 
```

For watercolor dataset, feature alignment by adversarial training is performed to get the annotator.
```
 python train_adv.py \
            --dataset pascal_voc_0712_water --dataset_t watercolor --net res101 --bs $BATCH_SIZE 
```

#### Extract training features for learning dictionaries
Extarct foreground features on sim10k:
```
 python generate_gt_features.py --dataset sim10k --net vgg16 --r True --load_name $PATH_TO_ANNOTATOR
```

Extarct background features on sim10k:
```
 python generate_bg_features.py --dataset sim10k  --net vgg16 --r True --load_name $PATH_TO_ANNOTATOR
```

#### Learn dictionaries
Learn dictionaries for foreground classes:
```
 python learn_dicts.py
```
Learn background dictionaries by changing the settings in the 'learn_dicts.py' file.

#### Generate pseudo labels for target data

```
 python generate_inv_matrix.py

 python generate_annos.py --dataset_t cityscape_car --net vgg16 --r True --load_name $PATH_TO_ANNOTATOR
```

#### Perform self-training 
When doing self-training, the annotations path should be changed in the definition of dataset class accordingly.
```
 python train_cdg.py --dataset $DATASET_SOURCE --dataset_t $DATASET_TARGET --net $NET --bs $BATCH_SIZE
```

 ### Test
```
python test_net.py --dataset $DATASET --net $NET --r True --load_name $PATH_TO_ANNOTATOR
```