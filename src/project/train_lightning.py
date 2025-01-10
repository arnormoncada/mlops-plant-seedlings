import pytorch_lightning as pl
from pytorch_lightning import Trainer
from model import MyAwesomeModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

model = MyAwesomeModel()  # this is our LightningModule
early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=True, mode="min"
)
checkpoint_callback = ModelCheckpoint(
    dirpath="./models", monitor="val_loss", mode="min"
)
trainer = Trainer(default_root_dir="logs", max_epochs=5,
                  limit_train_batches=0.2,
                  limit_val_batches=0.2,
                  limit_test_batches=0.2,
                  callbacks=[early_stopping_callback, checkpoint_callback],
                  logger=pl.loggers.WandbLogger(project="dtu_mlops"))


trainer.fit(model)
