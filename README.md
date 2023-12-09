# EEG-to-Text-Project
Natural Language Processing (CS5624) Course Project in Virginia Tech


# Preprocessing the data
Add the mat files to the ```dataset/raw``` folder. Eg. ```dataset/raw/Task1_SR/{subject_mat_files}```
The ```dataset/raw``` folder should have three tasks' data: Task1_SR, Task2_NR, Task3_TSR.

Run

```bash scripts/preprocess_datasets.sh```

This will save the pickle files to ```dataset/processed``` directory

# Training the decoding model
After processing the datasets run

```bash scripts/train.sh```

This will train the decoder model for converting EEG to text embeddings. The models are saved in ```save_data/checkpoints/``` folder.

# Evaluate the decoding model
To evaluate the trained model run

```bash scripts/eval.sh```

This will save an output file with the target and predicted sentences in ```saave_data/eval_results``` folder.


#Zero-shot sentiment classification model
In contrast to the code you need to run using shell script, for the zero-shot sentiment classification model, you can just run jupyter notebook.

train_eval_sentiment_textbased.ipynb: this includes code for training and evaluating zero-shot classification model.
train_eval_sentiment_baseline.ipynb: this includes codes for training and evaluating baseline classification model.
