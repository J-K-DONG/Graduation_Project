import os
from ImageProcess import ImageTrack
import datetime
import cv2

if __name__ == "__main__":
    project_address = os.getcwd()
    start_time = datetime.datetime.now()
    fuse_method = "fadeInAndFadeOut"
    tracker = ImageTrack(os.path.join(project_address, "random_images"))
    tracked_image = tracker.start_track_and_fuse()
    end_time = datetime.datetime.now()
    print("The time of tracking is {}".format(end_time - start_time))
    cv2.imwrite("result.jpeg", tracked_image)