from tactile.data_loader import DataLoader
from tactile.tools import image_watchmap, data_visualization

DataLoader = DataLoader(
    '../live_run',
    {
        'alpha-hym': ['test1']
    }
)

print(DataLoader.get_data_length('alpha-hym'))

image = DataLoader.get_data('test', 201)
before_image = DataLoader.get_data('test', 200)
watchmap = image_watchmap(image)
before_watchmap = image_watchmap(before_image)
data_visualization(watchmap, 'watchmap.png')
data_visualization(before_watchmap, 'before_watchmap.png')
data_visualization(watchmap * before_watchmap, 'multi_watchmap.png')