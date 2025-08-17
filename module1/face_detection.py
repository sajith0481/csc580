import PIL.Image
import PIL.ImageDraw
import face_recognition

# Load the jpg file (face_detection.jpg) into a numpy array
image = face_recognition.load_image_file("face_detection.jpg")

# Find all the faces in the image
face_locations = face_recognition.face_locations(image)

numberOfFaces = len(face_locations)
print("Found {} face(s) in this picture.".format(numberOfFaces))

# Load the image into a Python Image Library object so that you can draw on top of it and display it
pilImage = PIL.Image.fromarray(image)

for faceLocation in face_locations:
    # Print the location of each face in this image. Each face is a list of co-ordinates in (top, right, bottom, left) order.
    top, right, bottom, left = faceLocation
    print("A face is located at pixel location Top: {}, Left: {}, "
          "Bottom: {}, Right: {}".format(top, left, bottom, right))
    
    # Draw a box around the face
    drawHandle = PIL.ImageDraw.Draw(pilImage)
    drawHandle.rectangle([left, top, right, bottom], outline="red", 
                        width=3)

# Display the image on screen
pilImage.show()