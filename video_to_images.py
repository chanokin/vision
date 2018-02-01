import cv2
import sys
import os


if len(sys.argv) != 2:
    print("Usage: python video_to_images.py path-to-video")
    sys.exit(1)

fname = sys.argv[1]

if not os.path.isfile(fname):
    print("File does not exist")
    sys.exit(1)

vid = cv2.VideoCapture(fname)

basename = os.path.basename(fname).replace(".", "_")
cwd = os.getcwd()
outdir = os.path.join(cwd, "%s_output_images"%(basename))

#cleanup
if os.path.isdir(outdir):
    files = os.listdir(outdir)
    for file in files:
        if file.endswith(".png"):
            os.remove(os.path.join(outdir, file))
else:
    os.makedirs(outdir)

frame = 0
while True:
    ret, img = vid.read()
    if not ret:
        break
    frame_name = "%s_frame_%010d.png"%(basename, frame)
    cv2.imwrite(os.path.join(outdir, frame_name), img)
    frame += 1

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

