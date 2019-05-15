## Assignment for class CS-5990
#### Task: <br/>
  To implement test function for translating images in both direction from source domain->target domain and target->source
  
#### Steps to Execute:<br/>
 1. Download [horse2zebra](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) dataset that has test folder consisting images of horses and zebras
 2. Download the pretrained checkpoint file from [here](https://drive.google.com/drive/folders/1AlhIHSQMESKvC1LVCVgwALWxArtYdhGC). 
    This checkpoint file is of model trained on horse2zebra dataset. Copy the files in ./checkpoints directory
 3. Edit model.py and implement:<br/>
      - load() function: to load checkpoint file<br/>
      - test() function: to translate test images in both the direction.<br/> 
        For more details about implementation refer [model.py](./model.py) file.
 4. To test images, execute following commands on Google Colab: <br/>
    ```"!python main.py --phase='test', --which_direction='AtoB'" ``` <br/>
    Give checkpoint file directory path in command line argument:<br/>
    ```--checkpoint_dir=your_dir ``` <br/>
    To change the directory where samples are saved, pass the path in command line argument:<br/>
    ``` --sample_dir=your_dir ```
 5. Submit the generated output images of horse->zebra and zebra->horse
