"""
create a coordinate matrix between two id fields
"""

#test is a pandas dataframe
coords = np.vstack([test[['FirstId','SecondId']].values,test[['SecondId','FirstId']].values])

coords = pd.DataFrame(coords,columns = ('FirstId','SecondId'))
coords = coords.drop_duplicates()

maxId = np.maximum(coords.FirstId.max(),coords.SecondId.max())

data = np.ones(coords.shape[0])

import scipy
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
#coo_matrix((data, (i, j)), [shape=(M, N)])
#to construct from three arrays:
#data[:] the entries of the matrix, in any order
#i[:] the row indices of the matrix entries
#j[:] the column indices of the matrix entries

inc_mat = scipy.sparse.coo_matrix((data, (coords.FirstId.values.ravel(), coords.SecondId.values.ravel()))
                        ,(maxId+1, maxId+1), dtype=np.int8 ) 
                        

rows_FirstId   = inc_mat[test.FirstId.values]
rows_SecondId  = inc_mat[test.SecondId.values]

#dot product to get the similarity between connections between ids
f = rows_FirstId.multiply(rows_SecondId).sum(axis=1)

val,cnts = np.unique(f.A,return_counts=True)



