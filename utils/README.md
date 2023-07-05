# Environment

The conda environment used for the project is described in [this yml file](https://github.com/sdesabbata/gnn-urban-form/blob/main/utils/conda-env_gnn-urban-form.yml). Updating the `DYLD_LIBRARY_PATH` as described below might be necessary.

```shell
export DYLD_LIBRARY_PATH=/opt/anaconda3/envs/gnn-urban-form/lib:$DYLD_LIBRARY_PATH
```