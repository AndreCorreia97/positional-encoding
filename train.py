import collections
import itertools
import os
import math
import torch

from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter

"""
Namespace(adam_betas='(0.9, 0.999)', adam_eps=1e-08, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, arch='transformer_wmt_en_de', attention_dropout=0.0, bucket_cap_mb=150, clip_norm=0.0, criterion='label_smoothed_cross_entropy', data=['data-bin/gigaWord'], ddp_backend='c10d', decoder_attention_heads=8, decoder_embed_dim=512, decoder_embed_path=None, decoder_ffn_embed_dim=2048, decoder_input_dim=512, decoder_layers=6, decoder_learned_pos=False, decoder_normalize_before=False, decoder_output_dim=512, device_id=0, distributed_backend='nccl', distributed_init_method=None, distributed_port=-1, distributed_rank=0, distributed_world_size=1, dropout=0.3, encoder_attention_heads=8, encoder_embed_dim=512, encoder_embed_path=None, encoder_ffn_embed_dim=2048, encoder_layers=6, encoder_learned_pos=False, encoder_normalize_before=False, fix_batches_to_gpus=False, fp16=False, fp16_init_scale=128, keep_interval_updates=-1, label_smoothing=0.1, left_pad_source='True', left_pad_target='False', log_format=None, log_interval=1000, lr=[0.001], lr_scheduler='inverse_sqrt', lr_shrink=0.1, max_epoch=100, max_sentences=None, max_sentences_valid=None, max_source_positions=1024, max_target_positions=1024, max_tokens=3584, max_update=0, min_loss_scale=0.0001, min_lr=1e-09, momentum=0.99, no_epoch_checkpoints=False, no_progress_bar=False, no_save=False, no_token_positional_embeddings=False, optimizer='adam', optimizer_overrides='{}', ordinary_sinpos=True, raw_text=False, relu_dropout=0.0, represent_length_by_ldpe=False, represent_length_by_lrpe=True, reset_lr_scheduler=False, reset_optimizer=False, restore_file='checkpoint_last.pt', save_dir='output', save_interval=1, save_interval_updates=0, seed=2723, sentence_avg=False, share_all_embeddings=True, share_decoder_input_output_embed=False, skip_invalid_size_inputs_valid_test=False, source_lang='wp_source', target_lang='wp_target', task='translation', train_subset='train', update_freq=[16], upsample_primary=1, valid_subset='valid', validate_interval=1, warmup_init_lr=1e-07, warmup_updates=4000, weight_decay=0.0)
"""

device_id = torch.device("cuda:0")
seed = 2723
distributed_world_size = 1
distributed_rank = 0
train_subset = 'train'


def main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    print(args)

    if not torch.cuda.is_available():
        print('Training on CPU is not supported')
        exit()

    # Set pytorch to use the GPU
    torch.cuda.set_device(device_id)

    # Initialize weights with the seed used by the authors
    torch.manual_seed(seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load dataset splits
    load_dataset(task, ['train', 'valid'])

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)

    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    )
    dummy_batch = task.dataset('train').get_dummy_batch(args.max_tokens, max_positions)

    # Build trainer
    trainer = Trainer(args, task, model, criterion, dummy_batch)

    # Initialize dataloader
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )

    # Train
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    # Max epoch, max updates or learning rate gets too small
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

    # Save the model
    save_model(args, trainer, epoch_itr, valid_losses[0])
    print('Done.')


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""

    # Update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(fix_batches_to_gpus=args.fix_batches_to_gpus)
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf
    num_batches = len(epoch_itr)
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0 and num_updates > 0:
            valid_losses = validate(args, trainer, task, epoch_itr, [first_valid])
            save_model(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats)

        valid_losses.append(stats['valid_loss'])
    return valid_losses


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = '{:.3f}'.format(trainer.get_meter('train_loss').avg)
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss').avg
        stats['nll_loss'] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss').avg
    stats['ppl'] = get_perplexity(nll_loss)
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = '{:.3f}'.format(trainer.get_meter('gnorm').avg)
    stats['clip'] = '{:.0%}'.format(trainer.get_meter('clip').avg)
    stats['oom'] = trainer.get_meter('oom').avg
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = '{:.3f}'.format(trainer.get_meter('loss_scale').avg)
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = round(trainer.get_meter('train_wall').sum)
    return stats


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['valid_loss'] = trainer.get_meter('valid_loss').avg
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss').avg
        stats['valid_nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('valid_loss').avg
    stats['valid_ppl'] = get_perplexity(nll_loss)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_model, 'best'):
        stats['best'] = min(save_model.best, stats['valid_loss'])
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_model(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_model, 'best') or val_loss < save_model.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_model, 'best', val_loss)
    if val_loss is not None:
        save_model.best = min(val_loss, prev_best)
    extra_state = {
        'best': save_model.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def load_dataset(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e


if __name__ == '__main__':
    main()
