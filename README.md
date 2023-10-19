# EEG-to-Text-Project
Natural Language Processing (CS5624) Course Project in Virginia Tech


# Preprocessing the data
Add the mat files to the ```dataset/raw``` folder. Eg. ```dataset/raw/Task1_SR/{subject_mat_files}```

run 

```python preprocess_mat_files.py --task {task_name}```

This will save the pickle files to ```dataset/processed``` directory
