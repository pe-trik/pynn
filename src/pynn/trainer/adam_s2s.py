# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import math

import torch
from torch.cuda.amp import autocast

from . import EpochPool, ScheduledOptim
from . import cal_ce_loss, load_last_chkpt, save_last_chkpt

def train_epoch(model, data, opt, eps, device, b_update, b_sync, n_print,
                teacher_force=1., weight_noise=False, grad_clip=0., grad_norm=False, fp16=False):
    ''' Epoch operation in training phase'''
    model.train()
    
    total_loss, n_token_total, n_token_correct = 0., 0, 0
    prints, p_loss, p_token, p_correct = n_print, 0., 0, 0

    updates, steps = 0, 0
    n_seq = 0
    num_labels = 0

    data_len = len(data)
    total_labels = data.get_total_labels()
    loader = data.create_loader()
    opt.zero_grad()
    for batch_i, batch in enumerate(loader):
        # prepare data
        #if batch_i > 10:
        #    break

        src_seq, src_mask, tgt_seq = map(lambda x: x.to(device), batch)      

        num_labels += (tgt_seq.count_nonzero() - (tgt_seq.size(0)*2))
        gold = tgt_seq[:, 1:]
        tgt_seq = tgt_seq[:, :-1]
        last = (batch_i == data_len)
        n_seq += tgt_seq.size(0)
 
       
        
        #total_labels = data.get_total_labels()
        
        try:
            # forward
            with autocast(enabled=fp16):
                # Gaussian weight noise
                if weight_noise: opt.apply_weight_noise()

                # teacher forcing or sampling
                sampling = teacher_force < 1.
                if sampling:
                    pred, src_seq, src_mask = model(src_seq, src_mask, tgt_seq)            
                    pred = torch.argmax(pred, dim=-1).detach()
                    sample = pred.clone().bernoulli_(1. - teacher_force)
                    pred = pred * sample
                    pred = torch.cat((tgt_seq[:, :1], pred[:, :-1]), dim=1)
                    pred = pred * tgt_seq.gt(2).type(pred.dtype)
                    tgt_seq = tgt_seq * pred.eq(0).type(pred.dtype) + pred

                pred = model(src_seq, src_mask, tgt_seq, encoding=not sampling)[0]
                pred = pred.view(-1, pred.size(2))
                loss, loss_data, n_correct, n_token = cal_ce_loss(pred, gold, eps)
                if torch.isnan(loss.data):
                    print("    inf loss at %d" % n_seq); continue
            # backward
            opt.backward(loss)
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                print('    WARNING: ran out of memory on GPU at %d' % n_seq)
                torch.cuda.empty_cache(); continue
            raise err

        updates += n_token; steps += 1
        # update parameters
        if last or (updates >= b_update and b_update > 0) or (steps >= b_sync and b_sync > 0):
            norm = updates if grad_norm else 1
            opt.step_and_update_lr(grad_clip, norm)
            opt.zero_grad()
            updates, steps = 0, 0

        # note keeping
        total_loss += loss_data

        n_token_total += n_token;  n_token_correct += n_correct
        p_loss += loss_data; p_token += n_token; p_correct += n_correct
        
        if n_seq > prints and n_print > 0:
            ppl = math.exp(min(p_loss/p_token, 100))
            pred = p_correct * 1. / p_token
            print('    Seq: {:6d}, lr: {:.7f}, ppl: {:9.4f}, '\
                  'updates: {:6d}, correct: {:.2f}'.format(n_seq, opt.lr, ppl, opt.steps, pred), flush=True)
            prints += n_print
            p_loss, p_token, p_correct = 0., 0, 0


        if num_labels > total_labels:
            if updates != 0:
                norm = updates if grad_norm else 1
                opt.step_and_update_lr(grad_clip, norm)
                opt.zero_grad()
                updates, steps = 0, 0
            break

    print("{}/{} tokens seen out of total".format(num_labels, total_labels))    
    loss_per_token = total_loss / n_token_total
    accuracy = n_token_correct / n_token_total
    return loss_per_token, accuracy

def eval_epoch(model, data, device, fp16=False):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    total_loss, n_token_total, n_token_correct = 0., 0, 0

    loader = data.create_loader()

    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            # prepare data
            src_seq, src_mask, tgt_seq = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]
            tgt_seq = tgt_seq[:, :-1]

            # forward
            with autocast(enabled=fp16):
                pred = model(src_seq, src_mask, tgt_seq)[0]
                loss, loss_data, n_correct, n_token = cal_ce_loss(pred, gold)
                if torch.isnan(loss.data):
                    print("    inf loss at batch %d" % batch_i); continue
            total_loss += loss_data
            n_token_total += n_token
            n_token_correct += n_correct

    loss_per_token = total_loss / n_token_total
    accuracy = n_token_correct / n_token_total
    return loss_per_token, accuracy
    
def train_model(model, datasets, epochs, device, cfg, args, fp16=False, dist=False):
    ''' Start training '''
    model_path = cfg['model_path']
    lr = cfg['lr']
    grad_norm = cfg.get('grad_norm', False)
    eps = cfg.get('label_smooth', 0.)
    teacher_force = cfg.get('teacher_force', 1.)
    weight_decay = cfg.get('weight_decay', 0.)
    weight_noise = cfg.get('weight_noise', False)

    n_warmup = cfg.get('n_warmup', 0)
    n_const = cfg.get('n_const', 0)
    n_print = cfg.get('n_print', 1000)
    b_input = cfg.get('b_input', 20000)
    b_sample = cfg.get('b_sample', 64)
    b_update = cfg.get('b_update', 8000)
    b_sync = cfg.get('b_sync', 0)

    n_save = cfg.get('n_save', 5)
    n_print = 0 if dist and device > 0 else n_print

    opt = ScheduledOptim(n_warmup, n_const, lr)
    model_opt = opt.initialize(model, device, weight_decay=weight_decay, dist=dist)

    tr_data, cv_data = datasets
    pool = EpochPool(n_save)
    epoch_i, _ = load_last_chkpt(model_path, model, opt)
    
    for k, v in cv_data.items():
        if k != "codeswitch":
            v.initialize(b_input, b_sample)
        else:
            v.initialize(b_input, b_sample, 1.0)

    while epoch_i < epochs:
        tr_data.initialize(b_input, b_sample, args.cs_ratio, args.cs_noswitches)
        tr_data.set_epoch(epoch_i)
        epoch_i += 1 
        if n_print > 0: print('[ Epoch', epoch_i, ']')
        start = time.time()
        tr_loss, tr_accu = train_epoch(model_opt, tr_data, opt, eps, device, b_update, b_sync, n_print,
                                       teacher_force, weight_noise, grad_norm=grad_norm, fp16=fp16)
        if dist and device > 0: continue
        print('  (Training)   ppl: {:8.5f}, accuracy: {:3.3f} %, elapse: {:3.3f} min'.format(
                 math.exp(min(tr_loss, 100)), 100*tr_accu, (time.time()-start)/60))

        total_cv_loss = 0
        for k, v in cv_data.items():
            start = time.time()
            cv_loss, cv_accu = eval_epoch(model_opt, v, device, fp16=fp16)
            print('  (Validation) {} ppl: {:8.5f}, accuracy: {:3.3f} %, elapse: {:3.3f} min'.format(
                     k, math.exp(min(cv_loss, 100)), 100*cv_accu, (time.time()-start)/60))
            total_cv_loss += cv_loss

        if math.isnan(total_cv_loss): continue
        model_file = model_path + '/epoch-{}.pt'.format(epoch_i)
        pool.save(total_cv_loss, model_file, model)
        save_last_chkpt(model_path, epoch_i, model, opt)