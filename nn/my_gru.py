import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class Seq2SeqGRU(nn.Module):

    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder_gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Decoder receives previous output (scalar) as input each step
        self.decoder_gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_size, 2)

    def forward(
        self,
        encoder_inputs: list[torch.Tensor],
        seq_lengths: list[int],
        decoder_targets: list[torch.Tensor],
        teaching_force: bool = True,
    ):
        # Encoder, hidden: (num_layers, B, hidden_size)
        _, hidden = self.encoder_gru(encoder_inputs)

        packed = pack_padded_sequence(encoder_inputs,
                                      seq_lengths,
                                      batch_first=True,
                                      enforce_sorted=False)
        _, hidden = self.encoder_gru(packed)

        # Decoder initialization: first input is last value of encoder_inputs (or zero)
        input_step = encoder_inputs[:, -1:, :]  # (B, 1, 1)

        outputs = []
        output_len = decoder_targets.size(1)
        for t in range(output_len):
            # out_gru: (B,1,hidden)
            out_gru, hidden = self.decoder_gru(input_step, hidden)
            pred = self.out(out_gru.squeeze(1)).unsqueeze(1)  # (B,1,1)
            outputs.append(pred)

            # next input: teacher forcing
            if teaching_force:
                input_step = decoder_targets[:, t:t + 1, :]  # use ground-truth
            else:
                input_step = pred

        outputs = torch.cat(outputs, dim=1)  # (B, output_len, 2)
        return outputs



dataset = []
max_len_in, max_len_out = 0, 0
for i in tqdm(range(1, 19)):
    input_file = pd.read_csv(f'data/train/input_2023_w{i:02}.csv')
    output_file = pd.read_csv(f'data/train/output_2023_w{i:02}.csv')

    data_in = input_file[input_file['player_to_predict'] == True]

    data_in["group"] = (data_in["frame_id"] == 1).cumsum()
    grouped_in = data_in.groupby("group")
    output_file["group"] = (output_file["frame_id"] == 1).cumsum()
    grouped_out = output_file.groupby("group")

    for ((_, slice_in), (_, slice_out)) in zip(grouped_in, grouped_out):
        target_in, target_out = (
            torch.from_numpy(slice_in.loc[:, ['x', 'y']].to_numpy(np.float32)),
            torch.from_numpy(slice_out.loc[:,
                                           ['x', 'y']].to_numpy(np.float32)),
        )
        # normalize
        target_in[:, 0] /= 120
        target_in[:, 1] /= 53.3
        target_out[:, 0] /= 120
        target_out[:, 1] /= 53.3

        l_in, l_out = len(target_in), len(target_out)
        max_len_in = max(max_len_in, l_in)
        max_len_out = max(max_len_out, l_out)
        dataset.append([target_in, target_out, l_in, l_out])


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: list[tuple[torch.Tensor, torch.Tensor, int,
                                           int]]):
        super(MyDataset, self).__init__()
        input_, output, in_lengths, out_lengths = [], [], [], []
        for data in dataset:
            input_.append(data[0])
            output.append(data[1])
            in_lengths.append(data[2])
            out_lengths.append(data[3])

        self.input_ = pad_sequence(input_, batch_first=True)
        self.output = pad_sequence(output, batch_first=True)
        self.in_lengths = in_lengths
        self.out_lengths = out_lengths
        self.mask = (
            torch.arange(max(self.out_lengths)).unsqueeze(0)
            < torch.tensor(out_lengths).unsqueeze(1)).unsqueeze(-1).float()

    def __getitem__(self, index):
        return (
            self.input_[index],
            self.output[index],
            self.in_lengths[index],
            self.mask[index],
        )

    def __len__(self):
        return len(self.input_)


# Device configuration - support MPS, CUDA, and CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 512
EPOCHS = 10

ratio = 0.7
split_idx = int(len(dataset) * ratio)
train_loader = torch.utils.data.DataLoader(
    MyDataset(dataset[:split_idx]),
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    MyDataset(dataset[split_idx:]),
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
)

model = Seq2SeqGRU(2, hidden_size=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

acc_train_loss = 0
for epoch in range(EPOCHS):
    print(f"Epoch {epoch}:")
    tqdm_train = tqdm(enumerate(train_loader), desc='Training')

    model.train()
    acc_train_loss = 0
    for i, (input_, output, in_length, mask) in tqdm_train:
        input_, output = input_.to(device), output.to(device)
        mask = mask.to(device)

        preds = model(input_, in_length, output)
        se = (preds - output)**2  # (B, max_dec, 1)
        masked_se = se * mask  # (B, max_dec, 1)

        loss = masked_se.sum() / mask.sum().clamp_min(1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_train_loss += loss.item()
        tqdm_train.set_postfix(train_loss=acc_train_loss / (i + 1))

    model.eval()
    test_loss, num_test = 0, 0
    with torch.no_grad():
        for i, (input_, output, in_length, mask) in tqdm(enumerate(test_loader),
                                                        desc='Testing'):
            input_, output = input_.to(device), output.to(device)
            mask = mask.to(device)

            preds = model(input_, in_length, output, teaching_force=False)
            se = (preds - output)**2  # (B, max_dec, 1)

            masked_se = se * mask  # (B, max_dec, 1)

            test_loss += masked_se.sum()
            num_test += mask.sum()

        test_loss /= num_test
        print(f"\033[34mTest loss : {test_loss:.6f}\033[0m")
