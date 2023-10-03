# import the opencv library
import cv2
import tensorflow as tf
import numpy

model = tf.keras.models.load_model("keras_model.h5")


# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()

    # 1. resize the picture
    rezisedImg = cv2.resize(frame, (224, 224))
    
    # 2. expand the image
    expandImg1 = numpy.array(rezisedImg, dtype = numpy.float32)
    expandImg = numpy.expand_dims(expandImg1, axis = 0)

    # 3. normalize the image
    normalizing = expandImg/255.0

    #4. predict the output
    prediction = model.predict(normalizing)
    print(prediction)
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()