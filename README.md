# mvp-score-modelling

## Evaluation

### FID

`real_path` and `generated_path` need to have another subdirectory containing all the images, this is because `ImageFolder` is designed to work with a dataset where each class is in a separate subdirectory.

```sh
python mvp_score_modelling/evaluation/fid.py \
    --real_path output/images/real \
    --generated_path output/images/generated \
    --batch_size 32 --image_size 299 --feature 64 
```

TODO:
- [ ] write a script to save the test dataset of celebahq locally