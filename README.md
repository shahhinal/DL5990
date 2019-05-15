## CycleGAN Implementation
##### This project implements CycleGAN algorithm introduced by [Jun-Yan Zhu](http://people.csail.mit.edu/junyanz/)
1. [Original Paper](https://arxiv.org/pdf/1703.10593.pdf)
2. [Original Implementation](https://github.com/junyanz/CycleGAN/)
3. This project is re-implementation of https://github.com/xhujoy/CycleGAN-tensorflow
#### Libraries Required to Run This Project:
1. Tensorflow 1.0
2. Pyton 3.6
3. Numpy 1.11.0
4. Scipy 0.17.0
5. Pillow 3.3.0
6. GPU based system
#### To Execute program:
1. Install prerequisite library
2. Datasets Used:<br/> 
   [sunflower2daisy](https://github.com/luoxier/CycleGAN_Tensorlayer/tree/master/datasets/sunflower2daisy)<br/>
   [horse2zebra](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)<br/>
   [face2cartoon](https://cvit.iiit.ac.in/research/projects/cvit-projects/cartoonfaces)   
3. Download dataset and train the model<br/> 
   ``` python main.py --dataset_dir= hore2zebra ``` <br/>
   Models are saved to ``` ./checkpoints/``` directory (can change the path using  argument ```--checkpoint_dir=your_dir```)
4. To test the model <br/>
   ``` python main.py --dataset_dir=horse2zebra --phase=test --which_direction=AtoB ```
5. To visualize the traianing details: 
   ``` tensorboard --logdir=./logs ```
#### Results of this implementation
![ResultImages](https://user-images.githubusercontent.com/35668737/57800836-350c8480-7707-11e9-85b0-6c913201e715.jpg)




