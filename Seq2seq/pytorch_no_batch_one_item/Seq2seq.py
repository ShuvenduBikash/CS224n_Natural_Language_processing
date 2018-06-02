import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from text_loader import TextDataset
import seq2seq_models as sm
from seq2seq_models import str2tensor, EOS_token, SOS_token


# test function
def test():
    encoder_hidden = encoder.init_hidden()
    word_input = str2tensor('hello')
    encoder_outputs, encoder_hidden = encoder(word_input, encoder_hidden)
    
    decoder_hidden = encoder_hidden
    
    word_target = str2tensor('pytorch')
    for c in range(len(word_target)):
        decoder_output, decoder_hidden = decoder(word_target[c], decoder_hidden)
        print(decoder_output.size(), decoder_hidden.size())


def train(src, target):
    src_var = str2tensor(src)
    target_var = str2tensor(target, eos=True)
    
    encoded_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(src_var, encoded_hidden)
    
    hidden = encoder_hidden
    loss = 0
    
    for c in range(len(target_var)):
        # we start with SOS token
        token = target_var[c-1] if c else str2tensor(SOS_token)
        output, hidden = decoder(token, hidden)
        loss += criterion(output, target_var[c].view(1))
        
    encoder.zero_grad()
    decoder.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.data[0] / len(target_var)
 
# Translate the given input
def translate(enc_input='thisissungkim.iloveyou.', predict_len=100, temperature=0.9):
    input_var = str2tensor(enc_input)
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_var, encoder_hidden)

    hidden = encoder_hidden

    predicted = ''
    dec_input = str2tensor(SOS_token)
    for c in range(predict_len):
        output, hidden = decoder(dec_input, hidden)

        # Sample from the network as a multi nominal distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Stop at the EOS
        if top_i is EOS_token:
            break

        predicted_char = chr(top_i)
        predicted += predicted_char

        dec_input = str2tensor(predicted_char)

    return enc_input, predicted       


if __name__ == "__main__":
    HIDDEN_SIZE = 100
    N_LAYERS = 1
    BATCH_SIZE = 1
    N_EPOCH = 100
    N_CHARS = 128  # ASCII


    # define encoder, decoder
    encoder = sm.EncoderRNN(N_CHARS, HIDDEN_SIZE, N_LAYERS)
    decoder = sm.DecoderRNN(HIDDEN_SIZE, N_CHARS, N_LAYERS)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    print("Encoder : \n", encoder)
    print("Decoder : \n", decoder)
    
    # testing the models
    test()
    
    # defining optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(dataset=TextDataset(), batch_size=BATCH_SIZE, shuffle=True)
    
    print("Training for %d epochs" % N_EPOCH)
    for epoch in range(1, N_EPOCH+1):
        for i, (src, target) in enumerate(train_loader):
            train_loss = train(src[0], target[0])
            
            if i % 100 is 0:
                print('[(%d %d%%) %.4f]' %
                      (epoch, epoch / N_EPOCH * 100, train_loss))
                print(translate(src[0]), '\n')
                print(translate(), '\n')
        
    
    