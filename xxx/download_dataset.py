# need to use updated master JupiterCVML code
import sys
sys.path.append('/home/bluerivertech/li.yu/code/JupiterCVML/europa/base/src/europa')

import os
# os.environ["BRT_ENV"] = 'prod'
os.environ['AWS_PROFILE'] = 'jupiter_prod_engineer'
import json
import brtdevkit
print(brtdevkit.__version__)
brtdevkit.log = 'info'

# from aletheia_dataset_creator.dataset_tools.aletheia_dataset_helpers import *
# from aletheia_dataset_creator.config.dataset_config import *
# from dl.dataset.brt_dataset_helpers import *
from brtdevkit.core.db import DBConnector
from brtdevkit.core.db.db_filters import *
from brtdevkit.data import AnnotationJob, LabelMap
from brtdevkit.data import Dataset
from brtdevkit.core import DBConnector as dbc

from datetime import datetime
from collections import Counter
from bson import ObjectId
import pandas as pd
import numpy as np
import imageio

# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import warnings
warnings.simplefilter('ignore')

dataset_name = 'Jupiter_March2022_VT_0_1_pass'
# dataset_dir = os.path.join('/data/jupiter/datasets/', dataset_name)
dataset_dir = os.path.join('/data/jupiter/li.yu/data/', dataset_name)
os.makedirs(dataset_dir, exist_ok=True)

test_dataset = Dataset.retrieve(name=dataset_name)
test_df = test_dataset.to_dataframe()
test_dataset.download(dataset_dir, df=test_df)
