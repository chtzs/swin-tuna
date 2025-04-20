from mmseg.registry import MODELS
from mmengine.model import BaseModule

class StateMachine:
    IDLE = 0
    TRAINING = 1
    EVALUATION = 2

class PEFT(BaseModule):
    def __init__(self, init_cfg = None):
        super().__init__(init_cfg)
        self.state = StateMachine.IDLE
    
    def freeze_parameters(self, mode=True):
        pass
    
    def train(self, mode=True):
        if self.state == StateMachine.IDLE or self.state == StateMachine.EVALUATION:
            super().train(mode=mode)
            self.freeze_parameters(mode=mode)
            self.state = StateMachine.TRAINING
        elif self.state == StateMachine.TRAINING:
            # Still on training
            if mode == True:
                pass
            # Evaluation
            else:
                super().train(mode=mode)
                self.freeze_parameters(mode=mode)
                self.state = StateMachine.EVALUATION