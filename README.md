# Info

This project is an end-to-end data science workflow of loading input data, training an SVD recommendation model, and outputting personalized movie title recommendations & predicted ratings. In addition to the MovieLens 32M dataset, my own Letterboxd ratings were also exported and supplemented into the MovieLens dataset just to provide a personalized dimension to this project.

## Project Structure

```
├───sandbox
│       1_prep_data_eda.ipynb
│       2_train_predict_svd.ipynb
└───src
│       train_svd_params.py
```

Training script can be run with cmd arguments from `src` directory via:

`python train_svd_params.py --training_data ratings_combined.parquet --n_epochs 20 --lr_all 0.005 --reg_all 0.02`
