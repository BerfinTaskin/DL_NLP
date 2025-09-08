from dataclasses import dataclass

@dataclass
class QQPConfig:
    # data (defaults match your repo’s arguments in multitask_classifier.py)
    quora_train: str = "data/quora-paraphrase-train.csv"
    quora_dev:   str = "data/quora-paraphrase-dev.csv"
    quora_test:  str = "data/quora-paraphrase-test-student.csv"

    # training
    batch_size: int = 64
    max_epochs: int = 3
    lr: float = 1e-5
    weight_decay: float = 0.01
    temperature: float = 0.05
    seed: int = 11711
    use_gpu: bool = True
    local_files_only: bool = False
    grad_clip: float = 1.0

    # I/O
    save_dir: str = "models"
    run_name: str = "qqp-improve"

    dev_out:  str = "predictions/bert/quora-paraphrase-dev-output.csv"
    test_out: str = "predictions/bert/quora-paraphrase-test-output.csv"

    # mining for MNR (optional TF-IDF hard negatives)
    mine_hard_negs: bool = True
    max_neg_pool: int = 50000   # limit negatives pool for speed
