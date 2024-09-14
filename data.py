import anndata as ad
import pandas as pd
import scanpy as sc
import sys


SITE1_CELL = 16311
SITE2_CELL = 25171
SITE3_CELL = 32029
SITE4_CELL = 16750
GEX = 2000


adata = ad.read_h5ad('/workspace/scGGAN/data/citeseq_processed.h5ad')
adata.var_names_make_unique()

### preprocess
adata_GEX = adata[:, adata.var["feature_types"] == "GEX"].copy()
adata_ADT = adata[:, adata.var["feature_types"] == "ADT"].copy()
### step 1
sc.pp.normalize_total(adata_GEX, target_sum=1e4)
sc.pp.normalize_total(adata_ADT, target_sum=1e4)
### step 2
sc.pp.log1p(adata_GEX)
sc.pp.log1p(adata_ADT)
### step 3
sc.pp.highly_variable_genes(
    adata_GEX,
    n_top_genes=2000,
    subset=True
)
adata = ad.concat([adata_GEX, adata_ADT], axis=1, merge="first")   # X(:,1): GEX, X(:,2): ADT

adata_sub = adata[16311 + 25171 - 2500:16311 + 25171 + 2500, :].copy()
print(adata_sub)
adata_sub.write_h5ad('/workspace/scGGAN/data/citeseq_preprocessed.h5ad')
sys.exit()

adata_sub.obs.to_csv('/workspace/scGGAN/data/citeseq_preprocessed_cell_metadata.csv')
adata_sub.var.to_csv('/workspace/scGGAN/data/citeseq_preprocessed_gene_metadata.csv')

X = adata_sub.X.toarray()
X[-2500, :GEX] = 0

data = pd.DataFrame(X, index=adata_sub.obs.index, columns=adata_sub.var.index)

chunk_size = 1000  # 每次写入 1000 行

print(f'Start writing.\n')
with open('/workspace/scGGAN/data/citeseq_preprocessed.csv', 'w') as f:
    data.iloc[:0].to_csv(f, index=True)
    
    total_rows = len(data)
    for i in range(0, total_rows, chunk_size):
        data.iloc[i:i + chunk_size].to_csv(f, header=False, index=True)
        print(f'Written {i + chunk_size if i + chunk_size < total_rows else total_rows}/{total_rows} rows')

print("Finished writing the CSV file.")