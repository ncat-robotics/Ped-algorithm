from vidgear.gears import VideoGear
import cv2
import time
streamlines = [0,2,4]
while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            #if 'q' key-pressed break out
            break
        for streamline in streamlines:
            stream = VideoGear(source=streamline, logging=True).start() 
            frame = stream.read()

            if frame is None:
            #if True break the infinite loop
                break
            cv2.imshow("Output Frame" + str(streamline), frame)
            
            if key == ord("w"):
                cv2.imwrite("Image-" + str(streamline) + ".jpg", frame)
            stream.stop()

cv2.destroyAllWindows()
stream1.stop()