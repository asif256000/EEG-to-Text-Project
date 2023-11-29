# python3 eval.py \
#         -checkpoint ./save_data/checkpoints/all_tasks_skip_step1/best/final.pt \
#         -config ./save_data/config/config_all_tasks_skip_step1.pickle


# python3 eval.py \
#         -checkpoint ./save_data/checkpoints/all_tasks_all_steps/best/final.pt \
#         -config ./save_data/config/config_all_tasks_all_steps.pickle


python3 eval.py \
        -checkpoint ./save_data/checkpoints/task1_all_steps/best/final.pt \
        -config ./save_data/config/config_task1_all_steps.pickle


python3 eval.py \
        -checkpoint ./save_data/checkpoints/task2_all_steps/best/final.pt \
        -config ./save_data/config/config_task2_all_steps.pickle


python3 eval.py \
        -checkpoint ./save_data/checkpoints/task3_all_steps/best/final.pt \
        -config ./save_data/config/config_task3_all_steps.pickle
