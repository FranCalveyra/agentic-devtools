from config.environment import Environment


class Config:
    def __init__(self) -> None:
        self.environment: Environment = Environment()


config = Config()
