from config.base_config import Config
from model.clip_baseline import CLIPBaseline
from model.clip_trainsformer_hash4_HTPC_STSR import CLIPTransformer


class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        if config.arch == 'clip_baseline':
            return CLIPBaseline(config)
        elif config.arch == 'clip_transformer':
            return CLIPTransformer(config)
        else:
            raise NotImplemented
