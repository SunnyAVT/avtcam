"""AVT Camera Python top level module.

(C) 2019 Allied Vision Technologies GmbH - All Rights Reserved

"""

__version__ = '0.2.0'

# wait to do for more features support
__all__ = [
    'AVTCamera',
    'bgr8_to_jpeg',
]

from .avt_camera import AVTCamera
from .avt_camera import bgr8_to_jpeg
