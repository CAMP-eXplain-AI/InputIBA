# Run Evaluation
We assume the `$PWD` is `path/to/InputIBA/` in the following tutorials.

## Computer Vision: Sanity Check, Insertion Deletion, Sensitivity-N, EHR
Experiments of this part require an addition json file that records the 
predicted probability of the ground truth class of each image. We use it to 
filter out the samples that have low confidence on target class. For the small 
ImageNet dataset downloaded from the aforementioned link, we also provided 
a json file [here](resources/target_scores.json). One can also obtain the json
file by running:
`python tools/vision/get_target_scores.py configs/vgg_imagenet.py 
workdirs/vgg_imagenet/target_scores/ target_scores.json`.

If using our provided json file, copy the file from `resources` to 
`workdirs` by `mkdir workdirs/vgg_imagenet/target_scores/ && cp 
resources/target_scores.json workdirs/vgg_imagenet/target_scores/`
### Sanity Check
1. Assume the `workdirs/vgg_imagenet/input_masks/` contains the final 
   attribution maps. Run:
```shell 
python tools/vision/sanity_check.py \
   configs/sanity_check.py \
   workdirs/vgg_imagenet/input_masks/ \
   workdirs/vgg_imagenet/sanity_check/ \
   vgg_sanity_check.json \
   --scores-file workdirs/vgg_imagenet/target_scores/target_scores.json
```
2. Check results in `workdirs/vgg_imagenet/sanity_check/`

### Insertion Deletion
1. Run 
```shell
python tools/vision/insertion_deletion.py \
  configs/vgg_imagenet.py \
  workdirs/vgg_imagenet/input_masks/ \
  workdirs/vgg_imagenet/insertion_deletion/ \
  vgg_insertion_deletion.json \
  --scores-file workdirs/vgg_imagenet/target_scores/target_scores.json \
  --sigma 15 \
  --num-samples 2000
```
2. Check results in `workdirs/vgg_imagenet/insertion_deletion`.

### Sensitivity-N
1. Run
```shell
python tools/vision/sensitivity_n.py \
  configs/vgg_imagenet.py \
  workdirs/vgg_imagenet/input_masks/ \
  workdirs/vgg_imagenet/sensitivity_n/ \
  vgg_sensitivity_n.json \
  --scores-file workdirs/vgg_imagenet/target_scores/target_scores.json \
  --num-masks 100 \
  --num-samples 1000
```
2. Check results in `workdirs/vgg_imagenet/sensitivity_n/`.


### EHR
1. Run 
```shell
python tools/vision/evaluate_ehr.py \
  configs/vgg_imagenet.py \
  workdirs/vgg_imagenet/input_masks/ \
  workdirs/vgg_imagenet/ehr/ \
  vgg_ehr.json \
  --weight \
  --scores-file workdirs/vgg_imagenet/target_scores/target_scores.json \
```
2. Check the files in `workdirs/vgg_imagenet/ehr/`.

## NLP: Insertion Deletion, Sensitivity-N
Assume `workdirs/lstm_imdb/input_masks/` stores the final attribution maps.
### Insertion Deletion
1. Run:
```shell
python tools/nlp/nlp_insertion_deletion.py \
  configs/deep_lstm.py \
  workdirs/lstm_imdb/input_masks/ \
  workdirs/lstm_imdb/insertion_deletion/ \
  lstm_insertion_deletion.json
```
2. Check the results in `workdirs/lstm_imdb/insertion_deletion/`.

### Sensitivity-N
1. Run:
```shell
python tools/nlp/nlp_sensitivity_n.py \
  configs/deep_lstm.py \
  workdirs/lstm_imdb/input_masks/ \
  workdirs/lstm_imdb/sensitivity_n/ \
  lstm_sensitivity_n.json \
  --num-masks 100
```
2. Check the results in `work_dirs/lstm_imdb/sensitivity_n/`.




