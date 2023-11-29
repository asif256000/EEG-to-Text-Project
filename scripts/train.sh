# python3 train.py \
#     -task task1_task2_task3 \
#     -skip1 \
#     -ne1 20 \
#     -ne2 30 \
#     -lr1 0.00005 \
#     -lr2 0.0000005 \
#     -b 32 \
#     -save ./save_data/checkpoints/all_tasks_skip_step1

# python3 train.py \
#     -task task1_task2_task3 \
#     -ne1 20 \
#     -ne2 30 \
#     -lr1 0.00005 \
#     -lr2 0.0000005 \
#     -b 32 \
#     -save ./save_data/checkpoints/all_tasks_all_steps

python3 train.py \
    -task task1 \
    -ne1 20 \
    -ne2 30 \
    -lr1 0.00005 \
    -lr2 0.0000005 \
    -b 32 \
    -save ./save_data/checkpoints/task1_all_steps

python3 train.py \
    -task task2 \
    -ne1 20 \
    -ne2 30 \
    -lr1 0.00005 \
    -lr2 0.0000005 \
    -b 32 \
    -save ./save_data/checkpoints/task2_all_steps

python3 train.py \
    -task task3 \
    -ne1 20 \
    -ne2 30 \
    -lr1 0.00005 \
    -lr2 0.0000005 \
    -b 32 \
    -save ./save_data/checkpoints/task3_all_steps