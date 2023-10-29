import argparse


def get_config(case):
    if case == 'train':
        parser = argparse.ArgumentParser(description='Arguments for training EEG-to-text model')

        parser.add_argument('--task-name', '-task', type=str, default='task1',
                            help='choose from {task1,task1_task2, task1_task2_task}')
        
        parser.add_argument('--num-epoch-step1', '-ne1', type=int, default=20)
        parser.add_argument('--num-epoch-step2', '-ne2', type=int, default=30)
        parser.add_argument('--learning-rate-step1', '-lr1', type=float, default=0.00005)
        parser.add_argument('--learning-rate-step2', '-lr2', type=float, default=0.0000005)
        parser.add_argument('--batch-size', '-b', type=int, default=32)

        parser.add_argument('--use-random-init', '-rand', action='store_true',
                            help='use random initialization for the pretrained model')
        parser.add_argument('--skip-step-one', '-skip1', action='store_true',
                            help='skip step one of the training process')
        parser.add_argument('--load-step-one', '-load1', action='store_true')
        parser.add_argument('--load-step-one-path', '-load1path', type=str, default=None)

        parser.add_argument('--eeg-type', '-eeg', type=str, default='GD',
                            help='choose from {GD, FFD, TRT}')
        parser.add_argument('--bands', '-bands', nargs='+', default=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'],
                            help='specify frequency bands to use')
        parser.add_argument('--add-CLS-token', '-CLS', action='store_true')
        parser.add_argument('--save-path', '-save', type=str, default='./save_data')

        args = vars(parser.parse_args())

    elif case == 'eval':
        parser = argparse.ArgumentParser(description='Arguments for evaluating EEG-to-text model')

        parser.add_argument('--checkpoint-path', '-checkpoint', type=str, default=None, required=True)
        parser.add_argument('--config-path', '-config', type=str, default=None, required=True)
        parser.add_argument('--output-results-path', '-output', type=str, default='./save_data/eval_results/results.txt')
        args = vars(parser.parse_args())
    
    return args