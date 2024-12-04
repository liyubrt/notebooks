import argparse
import os
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '2'


# Initialize distributed training
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def get_args_parser():
    parser = argparse.ArgumentParser('FSDP training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # * Finetuning params
    parser.add_argument('--finetune', action='store_true', help='Perform finetune.')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'OpenImages', 'COYO300M'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--output-dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 30)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    init_distributed_mode(args)
    rank = torch.distributed.get_rank()
    
    # Create your model
    model = MyModel()
    
    # Wrap the model with FSDP
    model = FSDP(model)
    
    # Define your optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    # Define your loss function
    loss_fn = nn.MSELoss()
    
    # Training loop
    for epoch in range(10):
        # Create dummy data
        data = torch.randn(100, 10).to(rank)
        target = torch.randn(100, 30).to(rank)
    
        # Forward pass
        output = model(data)
        loss = loss_fn(output, target)
    
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Print loss
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")