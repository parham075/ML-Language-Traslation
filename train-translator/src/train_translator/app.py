
import click
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,Dense
import numpy as np
from loguru import logger
from .ml_helper import lstm_model
@click.command(
    short_help="`train-translator` module which responsible for training a sequence to sequence model",
    help="`train-translator` module which responsible for training a sequence to sequence model",
)

@click.option(
    "--DATA_PATH",
    "DATA_PATH",
    help="path to data",
    required=True,
)
@click.option(
    "--NUM_SAMPLES",
    "NUM_SAMPLES",
    help="NUM_SAMPLES",
    required=True,
)
@click.option(
    "--BATCH_SIZE",
    "BATCH_SIZE",
    help="BATCH_SIZE",
    required=True,
)
@click.option(
    "--EPOCHS",
    "EPOCHS",
    help="EPOCHS",
    required=True,
)
@click.option(
    "--LATENT_DIM",
    "LATENT_DIM",
    help="LATENT_DIM",
    required=True,
)
@click.pass_context
def train_translator(ctx,**params):
    # Vectorize the data
    print(params)
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(params["DATA_PATH"],'r',encoding='utf-8') as file:
        lines = file.read().split('\n')

    #print(min(params["NUM_SAMPLES"],len(lines)-1))
    for line in lines[:min(int(params["NUM_SAMPLES"]),len(lines)-1)]:

        input_text, target_text , _ = line.split('\t')
        # The user use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end of sequence" charecter
        target_text = '\t' + target_text + "\n"
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    # extract some properties
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    logger.info(f"Number of samples: {len(input_texts)}")
    logger.info(f"Number of unique input tokens: {num_encoder_tokens}", )
    logger.info(f"Number of unique target tokens: {num_decoder_tokens}")
    logger.info(f"Max sequence length for input: {max_encoder_seq_length}")
    logger.info(f"Max sequence length for target: {max_decoder_seq_length}")
    input_token_index = dict(
        [(char,i) for i,char in enumerate(input_characters)]
    )
    target_token_index = dict(
        [(char,i) for i,char in enumerate(target_characters)]
    )
    ##############################################################################
    ##############################################################################
    # one hot representation
    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length,num_encoder_tokens),
        dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length,num_decoder_tokens),
        dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(target_texts), max_decoder_seq_length,num_decoder_tokens),
        dtype="float32"
    )

    for i , (input_text, target_text) in enumerate(zip(input_texts,target_texts)):
        for t , char in enumerate(input_text):
            encoder_input_data[i,t,input_token_index[char]]=1.
        encoder_input_data[i, t+1:, input_token_index[' ']]=1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t , target_token_index[char]]= 1.
            if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character
                decoder_target_data[i, t-1, target_token_index[char]]=1.
        decoder_input_data[i, t+1:, target_token_index[' ']] = 1.
        decoder_target_data[i, t:, target_token_index[' ']] = 1.
        ##################################################################
        ##################################################################
        translation_model_obj = lstm_model(
                                  num_encoder_tokens=num_encoder_tokens,
                                  num_decoder_tokens=num_decoder_tokens,
                                  latent_dim=int(params["LATENT_DIM"]),
                                  encoder_input_data= encoder_input_data,
                                  decoder_input_data=decoder_input_data, 
                                  decoder_target_data=decoder_target_data,
                                  batch_size=int(params["BATCH_SIZE"]), epochs=int(params["EPOCHS"])
                                   )
        translation_model_obj.encoder()
        translation_model_obj.decoder()
        translation_model_obj.model()
        translation_model_obj.training()
        
def main():
    train_translator()
main()
    