# Fine-Grained Neural Network Explanation by Identifying Input Features with Predictive Information

This repository is the official implementation.

<p align="center"> 
    <img alt="Example GIF" width="300" src="resources/demo.gif"><br>
    Example Image Mask
</p>

---
## Requirements
1. Install `torch` and `torchvision` (and `torchtext` for NLP tasks) 
   following the official instructions of [pytorch](https://pytorch.org/get-started/locally/)

2. Install `mmcv` or `mmcv-full` following the official instructions of [mmcv](https://github.com/open-mmlab/mmcv).

3. Install additional requirements with `pip install -r requirements.txt`.

4. Install the package in develop mode: `python setup.py develop`. 


## Run Attribution
1. Download ImageNet validation set. Format the sets to 
   `torchvison.dataset.ImageFolder` style if necessary. Use 
   [this script](tools/generate_small_imagenet.py) to generate two small 
   sets: estimation set and attribution set. The estimation set is for 
   estimating the mean and standard deviation of hidden features, while the 
   attribution set consists of images for the neural network to explain. 
   Cop [this json file](resources/imagenet_class_index.json) to the dataset 
   root. The dataset should have following structure:
   ```shell 
   .
   |-- annotations
   |   `-- attribution
   |   |   |-- n01440764
   |   |   |-- n01443537
   |   |   |-- n01484850
   |   |   ...  
   |-- imagenet_class_index.json
   `-- images
       |-- attribution
       |   |-- n01440764
       |   |-- n01443537
       |   |-- n01484850
       |   ...
       `-- estimation 
       |   |-- n01440764
       |   |-- n01443537
       |   |-- n01484850
       |   ...
   ```
   **Note** that the `annotations/` directory is only necessary for evaluating 
   localization ability of attribution methods (the EHR metric proposed in the 
   paper). 
   
   We also provide a preprocessed small ImageNet dataset, which can be 
   downloaded from 
   [this link](https://drive.google.com/file/d/1LBKQ4BR3zepfnQAKCumkABHYjXBanBBL/view?usp=sharing)
   
2. Create a directory under this repository: `mkdir data`, and link the 
   imagenet data path to `data/imagenet` : 
   `ln -s path/to/imagenet_data/ data/imagenet`.
3. Create a working directory, say `work_dir`.
4. Run training script with specified configuration file (e.g. 
   [vgg16_imagenet](configs/vgg_imagenet.py)) to train the attributor:
    ```shell
   python tools/vision/train.py 
       configs/vgg_imagenet.py \
       --work-dir work_dir \
       --gpu-id 0 \
       --pbar 
    ```
5. Check the results saved in `work_dir`: `input_masks/` consists of 
   the final attribution maps, while `feat_masks/` consists of the 
   attribution maps produced by the feature IBA (the original 
   [IBA](https://arxiv.org/abs/2001.00396))
 
## Pre-trained Models
Like many attribution methods, our method can only be applied in a per-image 
manner. For each new image, the `Attributor` will train new components 
(`FeatureIBA`, `WGAN`, `InputIBA`). 
Thus, there is no need to provide any pre-trained models here.
   
## Example Results
Here is an example of attribution maps produced by various attribution 
methods. By inspection, we can see that the attribution map of our method 
is much more fine-grained than the other ones. 
![Example Result](resources/example_results.jpg)

## License
This repository is released under the MIT license.


