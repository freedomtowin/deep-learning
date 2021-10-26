import GLV_WORD_CLASSIFIER 
import importlib
importlib.reload(GLV_WORD_CLASSIFIER)

import numpy as np
print(mdl.categorizer(np.array([['jewelry']])))
print(mdl.categorizer(np.array([['horseshoe']])))
print(mdl.categorizer(np.array([['sheep coat']])))