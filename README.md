# cytodata_aics

Generate data:
```
from cytodata_aics.split_dataset import generate_dataset
data_frame = generate_dataset()
data_frame.to_csv("/home/aicsuser/serotiny_data/cells_all_classes.csv")
```

Run training:
```
serotiny train     model=example_vae_3d_class     data=example_vae_dataloader_3d_all_classes     mlflow.experiment_name=cameroncatering     mlflow.run_name=cameron-catering-exp-vae     trainer.gpus=[0]     trainer.max_epochs=100
```