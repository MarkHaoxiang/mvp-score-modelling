# mvp-score-modelling

## Evaluation

### FID

`real_path` and `generated_path` need to have another subdirectory containing all the images, this is because `ImageFolder` is designed to work with a dataset where each class is in a separate subdirectory.

use `save_celeb.py` to save images from the test dataset to a local directory (evenness of male and female images are ensured)

```sh
python mvp_score_modelling/evaluation/fid.py \
    --real_path output/celebahq/test \
    --generated_path output/test_images/generated \
    --batch_size 32 --image_size 299 --feature 64 \
    --float64 True --seed 0
```

TODO:
- [x] write a script to save the test dataset of celebahq locally