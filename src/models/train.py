#coding:utf-8
import os, sys, time
import shutil, yaml
import argparse
import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

import source.yaml_utils as yaml_utils
from evaluation import sample_generate_light, calc_inception

from dataset import StampDataset
from updater import Updater

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/sn_base.yml', help='path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='index of gpu to be used')

    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--data_dir', type=str, default='../../data/interim')
    parser.add_argument('--results_dir', type=str, default='../../models/results/gans',
                        help='directory to save the results to')

    parser.add_argument('--inception_model_path', type=str, default='../../data/external/inception_score.model',
                        help='path to the inception model')
    parser.add_argument('--snapshot', type=str, default='',
                        help='path to the snapshot')
    parser.add_argument('--loaderjob', type=int,
                        help='number of parallel data loading processes')
    args = parser.parse_args()

    config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(config.batchsize))
    print('# iteration: {}'.format(config.iteration))
    print('Learning Rate: {}'.format(config.adam['alpha']))
    print('')

    # Load models
    def load_module(fn, name):
        """
        @param: fn: model file name
        @param: name: model name in file
        """
        mod_name = os.path.splitext(os.path.basename(fn))[0]
        mod_path = os.path.dirname(fn)
        sys.path.insert(0, mod_path)
        return getattr(__import__(mod_name), name)

    gen_model = load_module(
                os.path.join(args.base, config.models['generator']['fn']),
                config.models['generator']['name'])
    gen = gen_model(**config.models['generator']['args'])
    dis_model = load_module(
                os.path.join(args.base, config.models['discriminator']['fn']),
                config.models['discriminator']['name'])
    dis = dis_model(**config.models['discriminator']['args'])
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu(device=args.gpu)
        dis.to_gpu(device=args.gpu)

    models = {"gen": gen, "dis": dis}

    # Optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0., beta2=0.9):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        return optimizer
    opt_gen = make_optimizer(
        gen, alpha=config.adam['alpha'], beta1=config.adam['beta1'], beta2=config.adam['beta2'])
    opt_dis = make_optimizer(
        dis, alpha=config.adam['alpha'], beta1=config.adam['beta1'], beta2=config.adam['beta2'])
    opts = {"opt_gen": opt_gen, "opt_dis": opt_dis}

    # Dataset
    train = StampDataset(os.path.join(args.base, args.data_dir), True)
    train_iter = chainer.iterators.SerialIterator(train, batch_size=config.batchsize)

    kwargs = config.updater['args'] if 'args' in config.updater else {}
    kwargs.update({
        'models': models,
        'iterator': train_iter,
        'optimizer': opts,
    })
    updater = Updater(**kwargs, device=args.gpu)

    def create_result_dir(base_dir, output_dir, config_path, config):
        """https://github.com/pfnet-research/sngan_projection/blob/master/train.py"""
        result_dir = os.path.join(base_dir, output_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        def copy_to_result_dir(fn, result_dir):
            bfn = os.path.basename(fn)
            shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

        copy_to_result_dir(
            os.path.join(base_dir, config_path), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, config.models['generator']['fn']), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, config.models['discriminator']['fn']), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, config.updater['fn']), result_dir)

    create_result_dir(args.base, args.results_dir, args.config_path, config)

    trainer = training.Trainer(updater,
                                (config.iteration, 'iteration'),
                                out=os.path.join(args.base, args.results_dir))

    report_keys = ["loss_dis", "loss_gen", "inception_mean", "inception_std"]

    # Set up logging
    trainer.extend(extensions.snapshot(), trigger=(config.snapshot_interval, 'iteration'))
    for m in models.values():
        trainer.extend(extensions.snapshot_object(
            m, m.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(config.snapshot_interval, 'iteration'))

    # Write a log of evaluation statistics
    trainer.extend(extensions.LogReport(keys=report_keys,
                                        trigger=(config.display_interval, 'iteration')))
    #trainer.extend(extensions.PrintReport(report_keys), trigger=(config.display_interval, 'iteration'))

    # evaluation functions
    trainer.extend(sample_generate_light(gen, os.path.join(args.base, args.results_dir), rows=10, cols=10),
                   trigger=(config.evaluation_interval // 10, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(calc_inception(gen, n_ims=5000, splits=1, path=os.path.join(args.base, args.inception_model_path)),
                   trigger=(config.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)

    #
    trainer.extend(extensions.ProgressBar(update_interval=config.progressbar_interval))
    ext_opt_gen = extensions.LinearShift('alpha', (config.adam['alpha'], 0.),
                                         (config.iteration_decay_start, config.iteration), opt_gen)
    ext_opt_dis = extensions.LinearShift('alpha', (config.adam['alpha'], 0.),
                                         (config.iteration_decay_start, config.iteration), opt_dis)

    trainer.extend(ext_opt_gen)
    trainer.extend(ext_opt_dis)
    if args.snapshot:
        print("Resume training with snapshot:{}".format(args.snapshot))
        chainer.serializers.load_npz(args.snapshot, trainer)

    # Run the training
    print("start training")
    trainer.run()

if __name__ == '__main__':
    main()
