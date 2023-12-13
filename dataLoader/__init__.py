from .llff import LLFFDataset
from .llff_3 import LLFFDataset3
from .llff_6 import LLFFDataset6
from .llff_9 import LLFFDataset9
from .blender import BlenderDataset
from .tankstemple import TanksTempleDataset

dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
               'llff_3':LLFFDataset3,
               'llff_6':LLFFDataset6,
               'llff_9':LLFFDataset9,
               'tankstemple':TanksTempleDataset}
