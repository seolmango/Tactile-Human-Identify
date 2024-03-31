from tactile.data_loader import DataLoader
from tactile.tools import image_watchmap, data_visualization

DataLoader = DataLoader(
    '230823_walkingData',
    {
        'ksh': ['1']
    }
)

image = DataLoader.get_data('ksh', 100)
before = DataLoader.get_data('ksh', 99)
image = image_watchmap(image)
before = image_watchmap(before)
data_visualization(image * before, "테스트")