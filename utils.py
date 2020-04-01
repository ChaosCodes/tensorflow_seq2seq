import os

def ensure_dir(dir_path):
    if os.path.exists(dir_path):
        return
    os.makedirs(dir_path)


def make_store_path(config):
    parameter = '-'.join([config.embedding_dim, config.hidden_dim, config.batch_size, config.learning_rate])
    return parameter
