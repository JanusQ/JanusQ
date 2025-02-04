
DEFAULTCONFIG = {
    'solver': 'lgd',
    'strongest_weight': 1000,
    'max_weight': 100,
    'high_weight': 10,
    'min_weight': 0.01,
    'step_size': 0.01,
    'steps': 1000,
    'early_stopping_iter':100,
    'sample_method': 'random',
    'base_num': 100,
    'device': 'simulate'
}

class Config:
    def __init__(self, **config):
        self.__dict__.update(DEFAULTCONFIG)
        self.__dict__.update(config)
        pass
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        setattr(self, key, value)
    def __delitem__(self, key):
        delattr(self, key)
    def __iter__(self):
        return iter(self.__dict__)
    def __len__(self):
        return len(self.__dict__)
    def __repr__(self):
        return repr(self.__dict__)

if __name__ == "__main__":
    morphConfig = Config(
    solver = 'lgd',
    max_weight = 100,
    high_weight = 10,
    min_weight = 0.01,
    step_size = 0.01,
    steps = 1000,
    sample_method = 'statevector',
    base_num = 100,
    device = 'simulate'
    )
    print(morphConfig.solver)
    morphConfig.solver = 'sgd'
    print(morphConfig.solver)