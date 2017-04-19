# Evaluation Files

This folder contains the evaluation results of the entire test set and the sample test set (filename ending _sample_).

- `vgg_results.csv`: Test set results of the image only VGG16 model
- `vgg_w_length_results.csv`: Results of the model combining image and length feature.
- `vgg_w_date_results.csv`: Results of the model combining image and date.
- `vgg_w_all_results.csv`: Results of the model combining image with length and date features. This model reaches the highest accuracy score on the test set data.

The .csv files have identical structure, where each row contains information for one image of the test set. The .csv files contain the following columns (header in the first row):

- `Index`: Unique identifier number for each test set sample.
- `image_path`: relative image path from within the data folder. There is one folder for each species, so that each image_path has the format species/image_name.png
- `video_name`: identifier of which video the image was originally taken from. Can be used to identify unique fish in combination with the column `label`, because the data was chosen such that only videos were taken for each species, in which only one fish of that species appears.
- `label`: This is the true class label. The number to name translation can be found in `release/examples/class_names.py`
- `pred`: In this column you can find the per-image classification result.
- `score0` to `score6`: This columns contain the per-class prediction probabilities. The `pred` column is chosen by taking the class with the highest probability for each image.
- `prediction_max_prob`: This column contains the **per-fish** classification result base on the "max_prob" method.
- `prediction_mode`: This column contains the **per-fish** classification result based on the "mode" method.

For more detailed explanation of how the "max_prob" and the "mode" column is derived read the description in the README.md in the root folder.
