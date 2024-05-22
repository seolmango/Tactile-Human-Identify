from tactile.data_loader import DataLoader
from tactile.tools import image_watchmap, data_visualization

DataLoader = DataLoader(
    '../live_run',
    {
        'test': ['sch']
    }
)

image = DataLoader.get_data('test', 201)
data_visualization(image, f"test")