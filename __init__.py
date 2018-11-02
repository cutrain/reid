import imageio
imageio.plugins.ffmpeg.download()

import get_data
import get_picture
import get_person
import get_feature
from classifier import exam_id

def reid(input_type, path):
    video = get_data.main(input_type, path)
    pictures = get_picture.main(video)
    ret = []
    for picture in pictures:
        persons = get_person.main(picture)
        res = []
        for person in persons:
            feature = get_feature.main(person)
            pid = exam_id(feature)
            res.append((pid,person))
        ret.append((picture, res))
    return ret

