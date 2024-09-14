**scGGAN: single-cell RNA-seq imputation by graph-based generative adversarial network**

scGGAN is committed to imputing scRNA-seq data that learns the gene-to-gene relations by Graph Convolutional Networks (GCN) and global scRNA-seq data distribution by Generative Adversarial Networks (GAN).

`Environments`

The model is based on Python and pytorch.

`Files`

> data:
>
> preprocessdata.csv: Single cell data obtained through preprocessing (Data cleaning, alignment, normalization, etc) (gene_num * cell_num)
>
> bulk.csv: bulk data (gene_num * bulk_num)
>
> label: cell type label (cell_num * 1)
>
> 4mergenevec:  4 k-mers gene sequence data (gene_num * 4^4)



> code:
>
> cal_net.py : calculate the gene relation network
>
> load_data.py: data loading related codes
>
> models.py: model related codes
>
> train_ggan.py: train model

`How to use`

First, calculate the gene relation network for single cell and bulk.

> single cell: 
>
> python cal_net.py --rna_data=./preprocessdata.csv --sc=1
>
> bulk:
>
> python cal_net.py --rna_data=./bulk.csv --sc=0
>
> parameters: 
>
> rna_data: path of rna_data
>
> sc: 1--single cell (default)  0--bulk

After running, obtain single cell network (*sc.npz*, gene_num * gene_num) and bulk network (*bulk.npz*, gene_num * gene_num) files respectively.

Then, train model

> python train_ggan.py 
>
> parameters:
>
> n_epochs: number of epochs of training
>
> batch_size:  batch size
>
> rna_data: path of single cell data
>
> label: path of label
>
> bulk: whether to use bulk data
>
> bulk_net: path of bulk net
>
> sc_net: path of single cell net
>
> gene_data:  gene sequence data


