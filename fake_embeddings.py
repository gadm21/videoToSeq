from VideoDataHandler import * 

import numpy as np








def create_embeddings(n=300):
    embeds = np.random.randint(0,2, n) - np.random.rand(n)
    return embeds













if __name__=='__main__':
    embeds = create_embeddings() 