import os

base_path = os.path.dirname(os.path.abspath(__file__))


class CFG:

    DEBUG = False
    TRAIN = True
    SUBMISSION_TRAIN = False
    TEST = True

    OUTPUT_DIR = os.path.join(base_path, "runs")
    EXPERIMENT_ID = "exp_8"
    DESCRIPTION = (
        "using longformer for just combined translated data with CLS Token Embedding"
    )

    MODEL_NAME = "xlm-roberta-base"
    MAX_LEN = 512
    # LANGUAGE = "en"
    USE_CLASS_WEIGHTS = False

    SEED = 42
    EPOCHS = 50
    SUBMISSION_EPOCHS = 35
    AUTO_LR_FIND = False
    LEARNING_RATE = 1e-5
    NUM_WORKERS = 32
    FIND_OPTIMAL_NUM_WORKERS = False  # set to True if you want to find the optimal number of workers for your machine #todo: need to integrate this into the code

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, EXPERIMENT_ID)
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, EXPERIMENT_ID, "checkpoints")
    LOG_DIR = os.path.join(OUTPUT_DIR, EXPERIMENT_ID, "logs")

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    TRAIN_DATA_PATH = os.path.join(
        base_path, "data", "combined", "train_updated2.csv"
    )
    DEV_DATA_PATH = os.path.join(base_path, "data", "en", "data_dev.csv")

    TEST_DATA_PATH = os.path.join(base_path, "data", "en", "data_test.csv")
    CLASSES = ["reporting", "opinion", "satire"]
    NUM_CLASSES = len(CLASSES)

    if DEBUG:
        print("Number of classes: ", NUM_CLASSES)

    VAL_SPLIT = 0.1
    SPLIT_STRATEGY = 2  # 1: random, 2: stratified 3: Iterative stratification

    TRAIN_BATCH_SIZE = 2
    VAL_BATCH_SIZE = 2
    TEST_BATCH_SIZE = 2
