import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1, device = None):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super().__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_size, padding_idx=self.dataset.pad_id)
        self.rnn = rnn_type(input_size=embed_size, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=self.vocab_size)

        self.device = device if device is not None else torch.device('cpu')

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        # This is a placeholder, you may remove it.
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        # Считаем эмбеддинги
        embeddings = self.embedding(indices)
        
        # Упаковываем, учитывая паддинг
        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        # Подаем последовательность в RNN
        # возвращает все скрытые состояния с последнего слоя
        # а вторым тензором набор h_T для каждого слоя (не используем)
        output_packed, _ = self.rnn(packed_embeddings)

        # Распаковываем и паддим результат
        output, _ = pad_packed_sequence(output_packed, batch_first=True, padding_value=self.dataset.pad_id)

        # Голову на скрытые состояния
        logits = self.linear(output)
        return logits


    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        # This is a placeholder, you may remove it.
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        # Токенизируем префикс
        tokens = [self.dataset.bos_id] + self.dataset.text2ids(prefix)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)

        # Считаем эмбеддинги
        embeddings = self.embedding(tokens)

        # Подаем префикс в RNN
        output, hidden_T = self.rnn(embeddings)

        # Голову на скрытые состояния
        logits = self.linear(output) / temp

        # Сэмплируем новый токен и конкатенируем
        new_tokens = Categorical(logits=logits[:, -1:]).sample()
        tokens = torch.cat([tokens, new_tokens], dim=1)

        while tokens.shape[1] < self.dataset.max_length and new_tokens.item() != self.dataset.eos_id:
            # Считаем эмбеддинги для нового слова
            embeddings = self.embedding(new_tokens)

            # Подаем префикс в RNN и дополнительно скрытые состояния, которые уже считали
            output, hidden_T = self.rnn(embeddings, hidden_T)

            # Голову на скрытые состояния
            logits = self.linear(output) / temp

            # Сэмплируем новый токен и конкатенируем
            new_tokens = Categorical(logits=logits[:, -1:]).sample()
            tokens = torch.cat([tokens, new_tokens], dim=1)

        return self.dataset.ids2text(tokens.squeeze())