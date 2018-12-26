import cv2

def get_data(path, input_type='local'):
    # TODO: multiple data type
    assert input_type in ['local', 'ftp', 'sql', 'network'], "type %s not in list" % input_type
    if input_type == 'local':
        video = cv2.VideoCapture(path)
        # video = imageio.get_reader(path, 'ffmpeg')
    else:
        raise(Exception("type %s not realize" % input_type))

    return video

if __name__ == "__main__":
    import sys
    video = main(sys.argv[1], sys.argv[2])
    print(type(video))


