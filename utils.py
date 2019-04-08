import PIL
import PIL.Image
import skvideo.io
import skimage.transform
import numpy as np

"""
Largely from TF ver https://github.com/GV1028/videogan
"""

# def save_gen(generated_images, n_ex = 36, epoch = 0, iter = 0):
#     for i in range(generated_images.shape[0]):
#         # cv2.imwrite('/root/code/Video_Generation/gen_images/image_' + str(epoch) + '_' + str(iter) + '_' + str(i) + '.jpg', generated_images[i, :, :, :])
#         PIL.Image.fromarray(np.around(generated_images[i, :, :, :]).astype(np.uint8)).save('/root/code/Video_Generation/gen_images/image_' + str(epoch) + '_' + str(iter) + '_' + str(i) + '.jpg')

def process_and_write_image(images,name):
    images = np.array(images).transpose((0,2,3,4,1))
    images = (images + 1)*127.5
    for i in range(images.shape[0]):
        PIL.Image.fromarray(np.around(images[i,0,:,:,:]).astype(np.uint8)).save("./genvideos/" + name + ".jpg")

def read_and_process_video(files,size,nof):
    videos = np.zeros((size,nof,64,64,3))
    counter = 0
    for file in files:
        vid = skvideo.io.vreader(file)
        curr_frames = []
        i = 0
        
        nr = np.random.randint(20)
        for frame in vid:
            i = i + 1
            if i <= nr:
                continue

            frame = skimage.transform.resize(frame,[64,64])
            curr_frames.append(frame)

            if i >= nr+nof:
                break

        curr_frames = np.array(curr_frames)
        curr_frames = curr_frames*255.0
        curr_frames = curr_frames/127.5 - 1
        videos[counter,:,:,:,:] = curr_frames
        counter = counter + 1

    return videos.transpose((0,4,1,2,3)).astype(np.float32)

def process_and_write_video(videos,name):
    videos = np.array(videos)
    videos = np.reshape(videos,[-1,3,32,64,64]).transpose((0,2,3,4,1))
    vidwrite = np.zeros((32,64,64,3))
    for i in range(videos.shape[0]):
        vid = videos[i,:,:,:,:]
        vid = (vid + 1)*127.5
        for j in range(vid.shape[0]):
            frame = vid[j,:,:,:]
            vidwrite[j,:,:,:] = frame
        skvideo.io.vwrite("./genvideos/" +name + ".mp4",vidwrite)

