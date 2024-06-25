import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on WRF and ERA fields')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--model', '-m', type=str, default='BERTunet',
                        help='Choose model architecture: ')
    parser.add_argument('--scheduler', type=str, default='MultiStepLR',
                        help='Choose scheduler type: MultiStepLR, linearWarmUp')
    parser.add_argument('--run-mode', '-r', dest='run_mode', type=str, default='train',
                        help='run mode: train, test or train+test')

    parser.add_argument('--run-id', dest='run_id', type=int, default=4,
                        help='if run mode is test select run id')
    parser.add_argument('--best-epoch-id', dest='best_epoch_id', type=int, default=15,
                        help='if run mode is test select best epoch id')
    parser.add_argument('--draw-plots', type=int, default=1,
                        help='If draw resulting plots')
    parser.add_argument('--weighted-meaner', type=int, default=1,
                        help='If use weighted mean for loss function')
    parser.add_argument('--loss-kernel', type=str, default='gauss',
                        help='Choose loss kernel: mean, gauss')
    parser.add_argument('--running-env', type=str, default='docker',
                        help='Specify where you run the script: docker, io, home')
    return parser.parse_args()
