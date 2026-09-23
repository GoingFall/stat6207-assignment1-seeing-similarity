"""Run revised training with retained selected checkpoints and predictions."""
from pipelines.v2_train_inat import main

if __name__ == '__main__':
    main(revision=True)
