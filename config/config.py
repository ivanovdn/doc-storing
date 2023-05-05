import yaml

CONFIG_PATH = "/Users/dmytro.ivanov/Projects/nlp/chat-pdf/config/config.yml"


def config():
    return yaml.safe_load(open(CONFIG_PATH))
