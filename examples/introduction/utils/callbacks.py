import copy
import numpy as np

class EarlyStopping():
    def __init__(self, key_to_monitor, patience, min_delta=0.0, retrieve_model=True):
        self.patience = patience
        self.min_delta = min_delta
        self.key_to_monitor = key_to_monitor
        self.retrieve_model = retrieve_model

        self.model_copy = None
        self.best_monitored_value = np.inf

    def on_epoch_end(self, history, model):
        values_to_monitor = history[self.key_to_monitor]
        if len(values_to_monitor) >= max(1, self.patience):
            # if it did not progress of -self.min_delta in patience epochs...
            if values_to_monitor[-self.patience] <= values_to_monitor[-1]+self.min_delta:
                return True

            if self.retrieve_model:
                if values_to_monitor[-1] < self.best_monitored_value:
                    self.model_copy = copy.deepcopy(model)
                    self.best_monitored_value = values_to_monitor[-1]

        return False