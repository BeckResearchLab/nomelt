import dvclive
import dvc.api
from dvclive.utils import standardize_metric_name
import transformers
from typing import Optional

class DVCLiveCallback(transformers.TrainerCallback):
    """HF callback for use with DVCLive.
    
    DVC's implimentation own implimentation has the following non-ideal features:
    - dvc live step is each logging step, instead of each eval step
    - it manually saves model at each epoch. 

    Here, on evaluate the metrics are recorded, and triggers the trainer control to
    save at this gradient ste. Then, on save, the dvclive step is taken.
    """
    
    def __init__(self, live: Optional[dvclive.Live] = None, **kwargs):
        super().__init__()
        self.live = live if live is not None else dvclive.Live(**kwargs)

    def on_log(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ):
        # saves training status as metrics to dvc, note that 
        # this only occurs when an evaluation step is necessary
        # logs are not available in the on_evaluate event
        logs = kwargs["logs"]
        for key, value in logs.items():
            try:
                self.live.log_metric(standardize_metric_name(key, __name__), value)
            except:
                pass # some things floating in the logs may not be recordable by dvc
            self.live.next_step()

    def on_evaluate(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ):
        metrics = kwargs.get('metrics')
        for key, value in metrics.items():
            try:
                self.live.log_metric(standardize_metric_name(key, __name__), value)
            except:
                pass # some things floating in the logs may not be recordable by dvc
        # if not already going to save, make sure it does
        # on_save above will be called and next step starts
        control.should_save=True

    def on_train_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ):  
        self.live.end()