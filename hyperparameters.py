class K:
    '''
    File containing parameters for the data preparation part, the Seq2Seq model
    and for testing final model performance.
    This class and its parameters are used across all other files for this model.
    '''
    # data parameters
    FILE_SRC = "corpus/europarl-v7.de-en.en"
    FILE_TRG = "corpus/europarl-v7.de-en.de"
    NUMBER_SENTENCES = 1000000
    SEQUENCE_LENGTH = 70
    DO_LOWER_CASE = True
    VOCAB_SIZE_SRC = 37000
    VOCAB_SIZE_TRG = 37000
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.2
    TOKENIZER_SRC_PATH = "sentencepiece/tokenizer_src.model"
    TOKENIZER_TRG_PATH = "sentencepiece/tokenizer_trg.model"
    # model parameters
    EMBEDDING_DIM = 512
    ATTENTION_DIM = 512
    LSTM_UNITS = 512
    DROPOUT = 0.2
    BATCH_SIZE = 128
    EPOCHS = 50
    PATIENCE = 10
    # translation parameters
    ADVANCED_METRICS = False
    NUMBER_TEST_SENTENCES = 10
