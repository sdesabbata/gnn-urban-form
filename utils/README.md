# Environment

The conda environment used for the project is described in [this yml file](https://github.com/sdesabbata/gnn-urban-form/blob/main/utils/conda-env_gnn-urban-form.yml). Updating the `DYLD_LIBRARY_PATH` as described below might be necessary.

```shell
export DYLD_LIBRARY_PATH=/opt/anaconda3/envs/gnn-urban-form/lib:$DYLD_LIBRARY_PATH
```

## Global Urban Street Networks

Create `storage` as a subfolder of the main directory, which will be used to store large data files
not to be syncronised with the GitHub repo. Download the 
[Global Urban Street Networks GraphML](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KA5HJ3) 
files of interest into the `storage` and unzip them.

```shell
mkdir storage
mkdir storage/osmnx
mkdir storage/osmnx/zip
mkdir storage/osmnx/graphml
wget -O storage/osmnx/zip/united_kingdom-GBR_graphml.zip https://dataverse.harvard.edu/api/access/datafile/4287573
unzip storage/osmnx/zip/united_kingdom-GBR_graphml.zip -d storage/osmnx/graphml
```