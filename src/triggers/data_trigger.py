from clearml.automation import TriggerScheduler

from src.settings import metadata
from src.train import start_training

scheduler = TriggerScheduler(
    pooling_frequency_minutes=1,
    sync_frequency_minutes=1,
    force_create_task_project=f"{metadata.PROJECT_NAME}/triggers",
    force_create_task_name="Train Trigger",
)


scheduler.add_dataset_trigger(
    schedule_function=start_training,
    name="Training on augmented dataset",
    trigger_project=metadata.PROJECT_NAME,
    trigger_on_tags=["use-training"],
)

scheduler.start()
