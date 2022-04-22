import torch
import torch.nn as nn
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    
    def __init__(self,encoded_image_size=14,fine_tune=True):
        super(Encoder,self).__init__()
        self.enc_size = encoded_image_size
        resnet = torchvision.models.resnet101(pretrained=True)
#         We remove the linear and pool layers at the end since we are not doing classification
        modules = list(resnet.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(self.enc_size)
        
        self.fine_tune(fine_tune)
    
    def forward(self,x):
        bp = self.model(x) # (batch,2048,img/32,img/32)
        ap = self.pool(bp) # (batch, 2048,enc_img_size,enc_img_size)
        out = ap.permute(0,2,3,1) #(batch,enc_img_size,enc_img_size,2048)
        return out
    
    def fine_tune(self,fine_tune=True):
        for p in self.model.parameters():
            p.requires_grad = False
#         If we fine tune then we only do with conv layers through blocks 2 to 4
        for c in list(self.model.children())[5:]:
            for p in c.parameters():
                p.requires_grad = True


class Attention(nn.Module):
    def __init__(self,encoder_dim,decoder_dim,attention_dim):
        """
        encoder_dim : size of encoded images
        decoder_dim : size of decoder RNNs
        attention_dim : size of the attention network
        """
        super(Attention,self).__init__()
        self.encoder_att = nn.Linear(encoder_dim,attention_dim)
        self.decoder_att = nn.Linear(decoder_dim,attention_dim)
        self.full_att = nn.Linear(attention_dim,1)#linear layer to calculate the value to be softmaxed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)# softmax layer to calculate the weights
        
    def forward(self,encoder_out,decoder_hidden):
        """
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out) 
        att2 = self.decoder_att(decoder_hidden)
        att_full = self.full_att(att1+att2.unsqueeze(1)).squeeze(2) #(batch,num_pixels)
        alpha = self.softmax(att_full) #(batch,num_pixels)
        attention_weighted_encoding = (encoder_out* alpha.unsqueeze(2)).sum(dim=1) #(batch_size,encoder_dim)
        
        return attention_weighted_encoding, alpha


class Decoder(nn.Module):
    def __init__(self,attention_dim,embed_dim, decoder_dim, vocab_size,encoder_dim = 2048, dropout = 0.5):
        super(Decoder,self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
#         self.dropout = dropout
        
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        
        self.embedding = nn.Embedding(vocab_size,embed_dim) #embedding layer
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim) #linear layer to find initial hidden layer in LSTM
        self.init_c = nn.Linear(encoder_dim, decoder_dim) #linear layer to find initial cell layer in LSTM
        self.f_beta = nn.Linear(decoder_dim, encoder_dim) #linear layer to find create a sigmoid-activated gate
        
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size) #linear layer to find scores over vocabulary
        self.init_weights()
        
    def init_weights(self):
        """
        Initialization over uniform distribution
        """
        self.embedding.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1,0.1)

    def load_pretrained_embeddings(self,embedding):
        """
        Loads pretrained embeddings
        """
        self.embedding.weight = nn.Parameter(embedding)
    def fine_tune_embeddings(self,fine_tune=True):
        """
        Unless using pretrained embeddings, keep it true
        """
        for p in self.embedding.parameters():
            p.requires_grad=fine_tune
            
    def init_hidden_state(self,encoder_out):
        """
        Creates initial hidden and cell state of the LSTM based on the encoded images.
        :encoder_out : encoded images, a tensor of dimension(batch_size, num_of_pixels,encoder_dim)
        :return hidden and cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out) #(batch_size,decoder_dim) output
        c = self.init_c(mean_encoder_out)
        return h,c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
        