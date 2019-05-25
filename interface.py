import cv2
import threading
from PIL import Image
from . import reid, nearby

class reidCore:
    def __init__(self, *, sample_frames=120, sample_gap=10):
        self.__tasks = {}
        self.__tasks_len = {}
        self.__sample = {}
        self.__sample_frames = sample_frames
        self.__sample_gap = sample_gap

    def gen_sample(self, taskname):
        if taskname not in self.__tasks_len:
            return
        l = self.__tasks_len[taskname]
        temp = self.__tasks[taskname][l-self.__sample_frames:l:self.__sample_gap]
        if len(temp) == 0:
            self.__sample[taskname] = []
            return
        size = list(temp[0].shape[:2])
        while size[1] > 640 or size[0] > 480:
            size[0] //= 2
            size[1] //= 2
        for i in range(len(temp)):
            temp[i] = cv2.resize(temp[i], (size[1], size[0]))
        self.__sample[taskname] = temp

    @staticmethod
    def save_video(images, output_path):
        fourcc = 'X264'
        fps = 30
        shape = images[0].shape
        width = shape[1]
        height = shape[0]
        videoWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        cnt = 0
        tot = len(images)
        for image in images:
            videoWriter.write(image[:,:,::-1])
            cnt += 1
            if cnt % 150:
                print('video save {}/{}'.format(cnt, tot))
        videoWriter.release()

    def __multicam(self, taskname, query_path, video_paths, output_paths):
        print('multicam task "{}" thread start running : {} {} {}'.format(
            taskname, query_path, video_paths, output_paths
        ))
        for i in range(len(video_paths)):
            video_path = video_paths[i]
            print('multicam task "{}" solving {}'.format(taskname, video_path))
            self.__tasks[taskname] = []
            self.__tasks_len[taskname] = 0
            it = reid(query_path, video_path)
            for i in it:
                self.__tasks[taskname].append(i)
                self.__tasks_len[taskname] += 1
            self.gen_sample(taskname)
            print('multicam task "{}":{} solved!'.format(taskname, video_path))
            self.save_video(self.__tasks[taskname], output_paths[i])
            print('multicam task "{}" saved at {}'.format(taskname, output_paths[i]))
            self.__tasks.pop(taskname)
            self.__tasks_len.poo(taskname)
            cnt += 1

    def multicam(self, taskname, query_path, video_paths, output_paths):
        if len(video_paths) != len(output_paths):
            raise Exception("video number & output number not match {},{}".format(len(video_paths, len(output_paths))))
        self.__tasks[taskname] = []
        self.__tasks_len[taskname] = 0
        thread = threading.Thread(target=self.__multicam, args=(taskname, query_path, video_paths, output_paths))
        thread.daemon = True
        thread.start()
        print('multicam task "{}" thread start'.format(taskname))

    def __nearperson(self, taskname, query_path, video_path, output_path, nearby_k=1):
        print('nearperson task "{}" thread start running : {} {} {}'.format(
            taskname, query_path, video_paths, output_path
        ))
        print('nearperson task "{}" getting person'.format(taskname))
        that_person = nearby(query_path, video_path, nearby_k=nearby_k)
        print('nearperson task "{}" got {} person'.format(taskname, len(that_person)))
        self.__tasks[taskname] = []
        self.__tasks_len[taskname] = 0
        it = reid(that_person, video_path)
        for i in it:
            self.__tasks[taskname].append(i)
            self.__tasks_len[taskname] += 1
        self.gen_sample(taskname)
        print('nearperson task "{}" solved!'.format(taskname))
        self.save_video(self.__tasks[taskname], output_path)
        print('nearperson task "{}" saved at {}'.format(taskname, output_path))
        self.__tasks.pop(taskname)
        self.__tasks_len.pop(taskname)

    def nearperson(self, taskname, query_path, video_path, output_path):
        self.__tasks[taskname] = []
        self.__tasks_len[taskname] = 0
        thread = threading.Thread(target=self.__nearperson, args=(taskname, query_path, video_path, output_path))
        thread.daemon = True
        thread.start()
        print('nearperson task "{}" thread start'.format(taskname))

    def sample(self, taskname, output_path):
        print('sample task "{}":{}'.format(taskname, output_path))
        self.gen_sample(taskname)
        if (taskname not in self.__sample) or len(self.__sample[taskname]) == 0:
            return False
        gif_list = []
        for i in self.__sample[taskname]:
            gif_list.append(Image.fromarray(i))
        gif_list[0].save(output_path, save_all=True, append_images=gif_list[1:], duration=int(1000/30), loop=0)
        return True
