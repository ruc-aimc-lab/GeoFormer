import io
import os

import torch.distributed as dist


_print = print


def print(*argc, **kwargs):
    if not dist.is_available() or not dist.is_initialized():
        _print(*argc, **kwargs)
        return

    output = io.StringIO()
    kwargs['end'] = ''
    kwargs['file'] = output
    kwargs['flush'] = True
    _print(*argc, **kwargs)

    s = output.getvalue()
    output.close()

    s = '[rank {}] {}'.format(dist.get_rank(), s)
    _print(s)


def reduce_mean(tensor, nprocs=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    nprocs = nprocs if nprocs else dist.get_world_size()
    rt = rt / nprocs
    return rt


def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def get_world_size(): return int(os.environ['WORLD_SIZE'])
def get_rank(): return int(os.environ['RANK'])
def get_local_rank(): return int(os.environ['LOCAL_RANK'])
