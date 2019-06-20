import cv2
import numpy as np
import threading
from PIL import Image
from . import reid, nearby, detect, draw_boxes, get_data

class reidCore:
    def __init__(self, *, sample_frames=120, sample_gap=10):
        self.__tasks = {}
        self.__tasks_len = {}
        self.__sample = {}
        self.__sample_frames = sample_frames
        self.__sample_gap = sample_gap
        self.__stop_flag = {}
        self.__flag = False

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

    def check_flag(self, taskname):
        if self.__flag:
            self.__stop_flag.pop(taskname)
            return False
        if self.__stop_flag[taskname]:
            self.__stop_flag.pop(taskname)
            return False
        return True

    def save_video(self, taskname, images, output_path):
        fourcc = 'X264'
        fps = 30
        shape = images[0].shape
        width = shape[1]
        height = shape[0]
        videoWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        cnt = 0
        tot = len(images)
        for image in images:
            if not self.check_flag(taskname):
                print('task "{}" stoped'.format(taskname))
                videoWriter.release()
                return
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
                if not self.check_flag(taskname):
                    print('multicam task "{}" stoped'.format(taskname))
                    self.__tasks_len.pop(taskname)
                    self.__tasks.pop(taskname)
                    return
                self.__tasks[taskname].append(i)
                self.__tasks_len[taskname] += 1
            self.gen_sample(taskname)
            print('multicam task "{}":{} solved!'.format(taskname, video_path))
            self.save_video(self.__tasks[taskname], output_paths[i])
            print('multicam task "{}" saved at {}'.format(taskname, output_paths[i]))
            self.__tasks_len.pop(taskname)
            self.__tasks.pop(taskname)

    def multicam(self, taskname, query_path, video_paths, output_paths):
        if len(video_paths) != len(output_paths):
            raise Exception("video number & output number not match {},{}".format(len(video_paths, len(output_paths))))
        thread = threading.Thread(target=self.__multicam, args=(taskname, query_path, video_paths, output_paths))
        thread.daemon = True
        self.__stop_flag[taskname] = False
        self.__flag = False
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
            if not self.check_flag(taskname):
                print('nearperson task "{}" stoped'.format(taskname))
                self.__tasks_len.pop(taskname)
                self.__tasks.pop(taskname)
                return
            self.__tasks[taskname].append(i)
            self.__tasks_len[taskname] += 1
        self.gen_sample(taskname)
        print('nearperson task "{}" solved!'.format(taskname))
        self.save_video(self.__tasks[taskname], output_path)
        print('nearperson task "{}" saved at {}'.format(taskname, output_path))
        self.__tasks_len.pop(taskname)
        self.__tasks.pop(taskname)

    def nearperson(self, taskname, query_path, video_path, output_path):
        thread = threading.Thread(target=self.__nearperson, args=(taskname, query_path, video_path, output_path))
        thread.daemon = True
        self.__stop_flag[taskname] = False
        self.__flag = False
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
        gif_list[0].save(output_path, save_all=True, append_images=gif_list[1:], duration=int(1000/60), loop=0)
        return True

    def stop(self, taskname=None):
        if taskname is None:
            self.__flag = True
            return
        if taskname not in self.__stop_flag:
            return
        self.__stop_flag[taskname] = True

class carCore:
    def __init__(self, *, sample_frames=120, sample_gap=10):
        self.__tasks = {}
        self.__tasks_len = {}
        self.__sample = {}
        self.__sample_frames = sample_frames
        self.__sample_gap = sample_gap
        self.__stop_flag = {}
        self.__flag = False

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

    def check_flag(self, taskname):
        if self.__flag:
            self.__stop_flag.pop(taskname)
            return False
        if self.__stop_flag[taskname]:
            self.__stop_flag.pop(taskname)
            return False
        return True

    def save_video(self, taskname, images, output_path):
        fourcc = 'X264'
        fps = 30
        shape = images[0].shape
        width = shape[1]
        height = shape[0]
        videoWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        cnt = 0
        tot = len(images)
        for image in images:
            if not self.check_flag(taskname):
                print('task "{}" stoped'.format(taskname))
                videoWriter.release()
                return
            videoWriter.write(image[:,:,::-1])
            cnt += 1
            if cnt % 150:
                print('video save {}/{}'.format(cnt, tot))
        videoWriter.release()

    def __car(self, taskname, query_path, video_paths, output_paths):
        print('car task "{}" thread start running : {} {} {}'.format(
            taskname, query_path, video_paths, output_paths
        ))
        for i in range(len(video_paths)):
            video_path = video_paths[i]
            print('car task "{}" solving {}'.format(taskname, video_path))
            self.__tasks[taskname] = []
            self.__tasks_len[taskname] = 0

            video = get_data(video_path)
            video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_num in range(video_length):
                ret, image = video.read()
                if not ret:
                    break
                image = np.flip(image, 2)
                bboxes = detect(image, class_='car')
                image = draw_boxes(image, bboxes, copy=False)
                if not self.check_flag(taskname):
                    print('car task "{}" stoped'.format(taskname))
                    self.__tasks_len.pop(taskname)
                    self.__tasks.pop(taskname)
                    return
                self.__tasks[taskname].append(image)
                self.__tasks_len[taskname] += 1
            self.gen_sample(taskname)
            print('car task "{}":{} solved!'.format(taskname, video_path))
            self.save_video(self.__tasks[taskname], output_paths[i])
            print('car task "{}" saved at {}'.format(taskname, output_paths[i]))
            self.__tasks_len.pop(taskname)
            self.__tasks.pop(taskname)

    def car(self, taskname, query_path, video_paths, output_paths):
        if len(video_paths) != len(output_paths):
            raise Exception("video number & output number not match {},{}".format(len(video_paths, len(output_paths))))
        thread = threading.Thread(target=self.__car, args=(taskname, query_path, video_paths, output_paths))
        thread.daemon = True
        self.__stop_flag[taskname] = False
        self.__flag = False
        thread.start()
        print('car task "{}" thread start'.format(taskname))

    def sample(self, taskname, output_path):
        print('sample task "{}":{}'.format(taskname, output_path))
        self.gen_sample(taskname)
        if (taskname not in self.__sample) or len(self.__sample[taskname]) == 0:
            return False
        gif_list = []
        for i in self.__sample[taskname]:
            gif_list.append(Image.fromarray(i))
        gif_list[0].save(output_path, save_all=True, append_images=gif_list[1:], duration=int(1000/60), loop=0)
        return True

    def stop(self, taskname=None):
        if taskname is None:
            self.__flag = True
            return
        if taskname not in self.__stop_flag:
            return
        self.__stop_flag[taskname] = True

