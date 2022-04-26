import argparse
import os
import json
import numpy as np
import pandas as pd

import time

import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from collections import Counter

from random import shuffle
import pickle

import json
import string
from typing import Tuple, List, Dict
from itertools import groupby
from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision
import torchvision.transforms as transforms

from models import Encoder, DecoderWithAttention
from utils import accuracy, adjust_learning_rate, clip_gradient, readImg, save_checkpoint, AverageMeter, CaptionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default=r"D:\flickr-image-dataset\flickr30k_images")
    parser.add_argument("--captions_path", type=str, default=r"D:\flickr-image-dataset\flickr30k_images\results.csv")
    parser.add_argument("--images_dir", type=str, default=r"D:\flickr-image-dataset\flickr30k_images\flickr30k_images")
    parser.add_argument("--output_dir", type=str, default=r"D:\Image_Captioning_Checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--attention_dim", type=int, default=512)
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--decoder_lr", type=float, default=4e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--alpha_c", type=float, default=1.0)
    #parser.add_argument("--best_bleu4", type=float, default=1e-4)
    #parser.add_argument("--epochs_since_improvement", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--fine_tune_encoder", type=str2bool, default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    parser.add_argument("--num_vocabs", type=int, default=10000)

    args = parser.parse_args()

    return args


def load_dataset(captions_path):
    df = pd.read_csv(captions_path, sep='|', header=None)
    records = df.to_records(index=False)
    records = list(records)[1:]
    captions_mapping = {}
    text_data = []
    
    for image, captions in tqdm(groupby(records, key=lambda x: x[0])):
        caption_list = list(captions)
        # print(type(caption_list[0][2]), caption_list[0][2])
        caption_list = list(map(lambda x: str(x[2]).strip(), caption_list))
        captions_mapping[image] = caption_list
        text_data.extend(caption_list)

    return captions_mapping, text_data


def split_data(captions_mapping):
    all_images = list(captions_mapping.keys())

    np.random.shuffle(all_images)

    # 3. Split into training and validation sets and test sets
    train_size = int(len(captions_mapping) * 0.8)
    valid_size = int(len(captions_mapping) * 0.95)

    train_data = {img_name: captions_mapping[img_name] for img_name in all_images[:train_size]}
    valid_data = {img_name: captions_mapping[img_name] for img_name in all_images[train_size:valid_size]}
    test_data = {img_name: captions_mapping[img_name] for img_name in all_images[valid_size:]}

    return train_data, valid_data, test_data


def filter_data(data, rem_punct):
    filter_data = {}
    for image in data:
        lines = data[image]
        filter_lines = []
        for line in lines:
            filter_line = line.split()
            filter_line = [word.lower() for word in filter_line]
            filter_line = [word.translate(rem_punct) for word in filter_line]
            #filter_line = [word for word in filter_line if len(word) > 1]
            filter_line = [word for word in filter_line if word.isalpha()]
            filter_line = " ".join(filter_line)
            
            filter_lines.append(filter_line)
        filter_data[image] = filter_lines
    return filter_data


def add_start_end(data, images):
    start = "<start> "
    end = " <end>"
    cap = []
    for image in images:
        if image == "":
            continue
        caption = [start + text + end for text in data[os.path.basename(image)]]
        cap += caption

    return cap


def calc_max_len(all_captions):
    len_cap =  np.array([len(text.split()) for text in all_captions])
    max_seq_len = len_cap.max()
    return max_seq_len


def build_word_map(all_captions, num_vocabs, tokenizer_file, min_threshold=0):
    if not os.path.exists(tokenizer_file):
        sent = [k.split(' ') for k in all_captions]
        sentences = [y for x in sent for y in x]

        count_words = dict(Counter(sentences))
        count_words = dict(sorted(count_words.items(), key=lambda x: x[1], reverse=True))
        count_words = dict(list(count_words.items())[:num_vocabs])
        words = [w for w in count_words.keys() if count_words[w] > min_threshold]
        #unk_words = [w for w in count_words.keys() if count_words[w] < min_threshold]
        word_map = {k: v + 1 for v, k in enumerate(words)}
        word_map['<unk>'] = len(word_map) + 1
        #word_map['<start>'] = len(word_map) + 1
        #word_map['<end>'] = len(word_map) + 2
        word_map['<pad>'] = 0
        with open(tokenizer_file, "w") as f:
            json.dump(word_map, f)
            f.close()
    else:
        with open(tokenizer_file, "r") as f:
            word_map = json.load(f)

    return word_map


def word2ind(x, word_map):
    if x in word_map.keys():
        return word_map[x]
    else:
        return word_map['<unk>']


def train(train_loader: DataLoader, encoder: Encoder, decoder: DecoderWithAttention, criterion, optim_encoder, optim_decoder, epoch, alpha_c, grad_clip, print_freq):
    decoder.train()
    encoder.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5acc = AverageMeter()
    
    start = time.time()
    
    for i , (imgs,caps,caplens) in enumerate(train_loader):
        data_time.update(time.time()-start)
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        
        #Forward Prop
        
        imgs = encoder(imgs)
        scores,cap_sorted, decode_lengths, alphas, sort_ind = decoder(imgs,caps,caplens)
        
        targets = cap_sorted[:,1:]
        
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        
        loss = criterion(scores,targets)
        
        #Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        
        #Backprop
        optim_decoder.zero_grad()
        if optim_encoder:
            optim_encoder.zero_grad()
        loss.backward()
        
        if grad_clip:
            clip_gradient(optim_decoder,grad_clip)
            if optim_encoder:
                clip_gradient(optim_encoder,grad_clip)
        
        #Update weights
        optim_decoder.step()
        if optim_encoder:
            optim_encoder.step()
        
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5acc.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5acc))


def validate(val_loader, encoder,decoder, criterion, alpha_c, print_freq, word_map):
    decoder.eval()
    if encoder:
        encoder.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores  = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets  = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4*100))

    return bleu4


def evaluate(beam_size,loader):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            vocab_size = len(word_map)+2
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds].long()]
            c = c[prev_word_inds[incomplete_inds].long()]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        try:
            
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        except:
            i=0

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4


if __name__ == "__main__":
    args = get_args()


    path = args.path
    captions_path = args.captions_path
    images_dir = args.images_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    emb_dim = args.emb_dim
    attention_dim = args.attention_dim
    decoder_dim = args.decoder_dim
    dropout = args.dropout
    start_epoch = args.start_epoch
    epochs = args.epochs
    encoder_lr = args.encoder_lr
    decoder_lr = args.decoder_lr
    grad_clip = args.grad_clip
    alpha_c = args.alpha_c
    best_bleu4 = 0
    epochs_since_improvement = 0
    print_freq = args.print_freq
    fine_tune_encoder = args.fine_tune_encoder
    checkpoint = args.checkpoint

    num_vocabs = args.num_vocabs

    tokenizer_file = os.path.join(output_dir, args.tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    captions_mapping, text_data = load_dataset(captions_path=captions_path)

    train_data, valid_data, test_data = split_data(captions_mapping=captions_mapping)

    x_train = list(map(lambda x: os.path.join(images_dir, x), train_data.keys()))
    x_val = list(map(lambda x: os.path.join(images_dir, x), valid_data.keys()))
    x_test = list(map(lambda x: os.path.join(images_dir, x), test_data.keys()))

    rem_punct = str.maketrans('', '', string.punctuation)

    train_data = filter_data(data=train_data, rem_punct=rem_punct)
    valid_data = filter_data(data=valid_data, rem_punct=rem_punct)
    test_data = filter_data(data=test_data, rem_punct=rem_punct)

    cap_train = add_start_end(data=train_data, images=x_train)
    cap_val = add_start_end(data=valid_data, images=x_val)
    cap_test = add_start_end(data=test_data, images=x_test)


    all_captions = cap_train +cap_val + cap_test
    print("Five captions: {}".format(all_captions[:5]))

    max_len = calc_max_len(all_captions=all_captions)

    word_map = build_word_map(all_captions=all_captions, num_vocabs=num_vocabs, tokenizer_file=tokenizer_file, min_threshold=0)

    tokenized_dataset = {}
    splits = ['train','val','test']
    caption_split = [cap_train, cap_val,cap_test]
    for i,split in enumerate(splits):
        token_caption = []
        token_caption_len = []
        for cap in caption_split[i]:
            s = list(map(lambda x: word2ind(x, word_map=word_map),cap.split(' ')))
            token_caption.append(list(s) + [word_map['<pad>']]*(max_len-len(s)))
            token_caption_len.append(len(cap.split(' ')))
        tokenized_dataset[split] = [token_caption,token_caption_len]

    transforms_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    train_data = CaptionDataset(x_train, tokenized_dataset['train'],'TRAIN',transform=transforms.Compose([normalize]))
    val_data = CaptionDataset(x_val, tokenized_dataset['val'],'VAL',transform=transforms.Compose([normalize]))
    test_data = CaptionDataset(x_test, tokenized_dataset['test'],'TEST',transform=transforms.Compose([normalize]))

    batch = batch_size
    train_loader = DataLoader(train_data,batch_size=batch, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data,batch_size=batch, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data,batch_size=1, shuffle=True, pin_memory=True)

    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim = emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size = len(word_map)+2,
                                   dropout=dropout)
    optim_decoder = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
    encoder = Encoder()
    encoder.fine_tune(fine_tune_encoder)
    optim_encoder = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr) if fine_tune_encoder else None


    if checkpoint:
        checkpoint = torch.load(checkpoint)
        decoder = checkpoint["decoder"]
        encoder = checkpoint["encoder"]
        optim_decoder.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        if fine_tune_encoder:
            optim_encoder.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        if fine_tune_encoder and encoder:
            encoder.fine_tune(fine_tune_encoder)
            optim_encoder = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)
    decoder.to(device)
    encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)


    bleu4_scores = []
    for epoch in tqdm(range(start_epoch,epochs)):
        if epochs_since_improvement > 8:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
                adjust_learning_rate(optim_decoder, 0.8)
                if fine_tune_encoder:
                    adjust_learning_rate(optim_encoder, 0.8)
        train(train_loader,encoder,decoder,criterion,optim_encoder,optim_decoder,epoch,alpha_c=alpha_c, grad_clip=grad_clip, print_freq=print_freq)
        recent_bleu4 = validate(val_loader,encoder,decoder,criterion, alpha_c=alpha_c, print_freq=print_freq, word_map=word_map)
        bleu4_scores.append(recent_bleu4)
        is_best = recent_bleu4 > best_bleu4
        if recent_bleu4 > best_bleu4:
            best_bleu4 = recent_bleu4
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        save_checkpoint(output_dir, 'caption_model', epoch, epochs_since_improvement, encoder, decoder, optim_encoder,
                        optim_decoder, recent_bleu4, is_best)


    beam_size=1
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size,test_loader) * 100))

    print("\nBLEU-4 score @ beam size of %d is %.4f." % (3, evaluate(3, test_loader) * 100))
