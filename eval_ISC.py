import os

import argparse
from argparse import Namespace
import os

import eval_tool.immatch as immatch
from eval_tool.immatch.utils.data_io import lprint
import eval_tool.immatch.utils.my_helper as helper
from eval_tool.immatch.utils.model_helper import parse_model_config


def eval_ische(
        root_dir,
        config_list,
        task='homography',
        h_solver='degensac',
        ransac_thres=2,
        match_thres=None,
        odir='outputs/fire',
        save_npy=False,
        print_out=False,
        debug=False,
):
    # Init paths

    cache_dir = os.path.join(root_dir, odir, 'cache')
    result_dir = os.path.join(root_dir, odir, 'results', task)
    if task == 'homography':
        mauc = 0
        data_root = os.path.join(root_dir, 'data/datasets/ISC-HE')
        im1_path = os.path.join(data_root, 'query')
        im2_path = os.path.join(data_root, 'refer')
        gd_path = os.path.join(data_root, 'gd')
        raw_data = []
        for i in os.listdir(im1_path):
            name = i.split('_')[0]
            i1 = os.path.join(im1_path, i)
            i2 = os.path.join(im2_path, name + '_1.jpg')
            txt = os.path.join(gd_path, name + '_2-' + name + '_1.txt')
            raw_data.append((i1, i2, txt))
        match_pairs = raw_data
    else:
        data_root = os.path.join(root_dir, 'data/datasets/ISC-HE')
        with open(os.path.join(data_root, 'index.txt'), 'r') as f:
            match_pairs = f.readlines()
            match_pairs = [m.replace('\n', '').split(',') for m in match_pairs]
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

        # Iterate over methods
    for config_name in config_list:
        # Load model
        args = parse_model_config(config_name, 'isc-he', root_dir)
        class_name = args['class']

        # One log file per method
        log_file = os.path.join(result_dir, f'{class_name}.txt')
        log = open(log_file, 'a')
        lprint_ = lambda ms: lprint(ms, log)

        # Iterate over matching thresholds
        thresholds = match_thres if match_thres else [args['match_threshold']]
        lprint_(f'\n>>>> Method={class_name} Default config: {args} '
                f'Thres: {thresholds}')

        for thres in thresholds:
            args['match_threshold'] = thres  # Set to target thresholds

            # Init model
            model = immatch.__dict__[class_name](args)

            matcher = lambda im1, im2: model.match_pairs(im1, im2)

            # Init result save path (for matching results)
            result_npy = None
            if save_npy:
                result_tag = model.name
                if args['imsize'] > 0:
                    result_tag += f".im{args['imsize']}"
                if thres > 0:
                    result_tag += f'.m{thres}'
                result_npy = os.path.join(cache_dir, f'{result_tag}.npy')

            lprint_(f'Matching thres: {thres}  Save to: {result_npy}')

            # Eval on the specified task(s)
            mauc = helper.eval_my(
                matcher,
                match_pairs,
                model.name,
                task=task,
                scale_H=getattr(model, 'no_match_upscale', False),
                h_solver=h_solver,
                ransac_thres=ransac_thres,
                lprint_=lprint_,
                debug=debug,
            )
        log.close()
        return mauc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark HPatches')
    parser.add_argument('--gpu', '-gpu', type=str, default='0')
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--odir', type=str, default='outputs/isc-he')
    parser.add_argument('--config', type=str, nargs='*', default=['geoformer'])
    parser.add_argument('--match_thres', type=float, nargs='*', default=None)
    parser.add_argument(
        '--task', type=str, default='homography',
        choices=['homography']
    )
    parser.add_argument(
        '--h_solver', type=str, default='cv',
        choices=['degensac', 'cv']
    )
    parser.add_argument('--ransac_thres', type=float, default=3)
    parser.add_argument('--save_npy', action='store_true')
    parser.add_argument('--print_out', action='store_true', default=True)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    eval_ische(
        args.root_dir, args.config,
        h_solver=args.h_solver,
        ransac_thres=args.ransac_thres,
        match_thres=args.match_thres,
        odir=args.odir,
        save_npy=args.save_npy,
        print_out=args.print_out,
        task='homography'
    )


