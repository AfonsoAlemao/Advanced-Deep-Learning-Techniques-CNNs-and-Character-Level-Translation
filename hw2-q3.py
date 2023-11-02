import argparse
import random
from functools import partial

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from data import collate_samples, MTDataset, PAD_IDX, SOS_IDX, EOS_IDX
from models import Encoder, Decoder, Seq2Seq, Attention, reshape_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    m = np.zeros([len(str2) + 1, len(str1) + 1], dtype=int)
    for x in range(1, len(str2) + 1):
        m[x, 0] = m[x - 1, 0] + 1
    for y in range(1, len(str1) + 1):
        m[0, y] = m[0, y - 1] + 1
    for x in range(1, len(str2) + 1):
        for y in range(1, len(str1) + 1):
            if str1[y - 1] == str2[x - 1]:
                dg = 0
            else:
                dg = 1
            m[x, y] = min(
                m[x - 1, y] + 1, m[x, y - 1] + 1, m[x - 1, y - 1] + dg
            )
    return m[len(str2), len(str1)]


def train(data, model, lr, n_epochs, padding_idx):

    train_iter, val_iter, test_iter = data

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_err_rates = []

    # Training the Model
    for epoch in range(n_epochs):

        for src, tgt in train_iter:
            src_lengths = (src != PAD_IDX).sum(1)
            src, tgt = src.to(device), tgt.to(device)
            src_lengths = src_lengths.to(device)

            optimizer.zero_grad()
            outputs, _ = model(src, src_lengths, tgt)
            # print(tgt[:, 1:].reshape(-1).shape)
            # print(outputs.reshape(-1, outputs.shape[-1]).shape)
            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]), tgt[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()

        print("Epoch: [%d/%d], Loss: %.4f" % (epoch + 1, n_epochs, loss))

        val_err_rate = test(model, val_iter, "val")

        val_err_rates.append(val_err_rate)

    test_err_rate = test(model, test_iter, "test", examples_idx=[42, 233, 512])

    return (val_err_rates, test_err_rate)


def test(model, data_iter, data_type, examples_idx=None):
    # Test the Model
    model.eval()
    error_rates = []
    true_strs = []
    pred_strs = []

    with torch.no_grad():
        for jj, (src, tgt) in enumerate(data_iter):
            src_lengths = (src != PAD_IDX).sum(1)
            src, tgt = src.to(device), tgt.to(device)
            src_lengths = src_lengths.to(device)

            # Tensor with start symbol index
            tgt_pred = torch.full(
                [data_iter.batch_size, 1],
                SOS_IDX,
                dtype=torch.long,
                device=device,
            )

            i = 1
            stop = False
            final_seq = [tgt_pred.view(-1)]

            encoder_outputs, final_enc_state = model.encoder(src, src_lengths)
            dec_state = final_enc_state

            if dec_state[0].shape[0] == 2:
                dec_state = reshape_state(dec_state)

            while i < 50 and not stop:
                output, dec_state = model.decoder(
                    tgt_pred, dec_state, encoder_outputs, src_lengths
                )

                output = model.generator(output)
                tgt_pred = output.argmax(-1)

                final_seq.append(tgt_pred.view(-1))

                # Stop symbol index
                if int(tgt_pred) == EOS_IDX:
                    stop = True
                i += 1

            final_seq = torch.cat(final_seq)
            true_str = data_iter.dataset.pairs[jj][1]
            true_len = len(true_str)
            pred_str = "".join(
                [
                    data_iter.dataset.output_lang.index2word[idx]
                    for idx in final_seq[1:-1].tolist()
                ]
            )
            error_rate = distance(true_str, pred_str) / true_len
            error_rates.append(error_rate)
            true_strs.append(true_str)
            pred_strs.append(pred_str)

    mean_error_rate = torch.tensor(error_rates).mean().tolist()

    if data_type == "train":
        dt_str = "Training"
    elif data_type == "val":
        dt_str = "Validation"
    else:
        dt_str = "Test"

    print(dt_str + " Error Rate of the model: %.4f" % (mean_error_rate))

    model.train()

    if examples_idx is not None:
        for idx in examples_idx:
            src_str = data_iter.dataset.pairs[idx][0]
            print('true src: "%s"' % (src_str,))
            print('true tgt: "%s"' % (true_strs[idx],))
            print('pred tgt: "%s"' % (pred_strs[idx],))

    return mean_error_rate


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_attn", action="store_const", const=True, default=False
    )

    opt = parser.parse_args()

    configure_seed(opt.seed)

    print("Loading data...")
    train_dataset = MTDataset("train")
    dev_dataset = MTDataset(
        "val",
        train_dataset.input_lang,
        train_dataset.output_lang,
    )
    test_dataset = MTDataset(
        "test",
        train_dataset.input_lang,
        train_dataset.output_lang,
    )

    collate_fn = partial(collate_samples, padding_idx=PAD_IDX)

    train_iter = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_iter = DataLoader(dev_dataset, batch_size=1, shuffle=False)
    test_iter = DataLoader(test_dataset, batch_size=1, shuffle=False)

    data_iters = (train_iter, val_iter, test_iter)

    src_vocab_size = train_dataset.input_lang.n_words
    tgt_vocab_size = train_dataset.output_lang.n_words

    padding_idx = PAD_IDX

    encoder = Encoder(
        src_vocab_size,
        opt.hidden_size,
        padding_idx,
        opt.dropout,
    )

    if opt.use_attn:
        attn = Attention(opt.hidden_size)
    else:
        attn = None

    decoder = Decoder(
        opt.hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        opt.dropout,
    )

    model = Seq2Seq(encoder, decoder).to(device)
    model.train()

    print("Training...")
    val_acc, test_acc = train(
        data_iters,
        model,
        opt.lr,
        opt.n_epochs,
        padding_idx,
    )

    print("Final validation error rate: %.4f" % (val_acc[-1]))
    print("Test error rate: %.4f" % (test_acc))

    plt.plot(np.arange(1, opt.n_epochs + 1), val_acc, label="Validation Set")

    plt.xticks(np.arange(0, opt.n_epochs + 1, step=2))
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.savefig(
        "attn_%s_err_rate.pdf" % (str(opt.use_attn),),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
