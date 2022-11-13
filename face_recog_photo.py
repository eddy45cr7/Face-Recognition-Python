import face_recognition as frec
import cv2
import os

def read_img(path):
    img = cv2.imread(path)
    (h, w) = img.shape[:2]
    width = 500
    ratio = width/float(w)
    height = int(h*ratio)
    return cv2.resize(img,(width,height))

known_encodings = []
known_names = []
known_dir = 'C:/Users/Admin/Desktop/Face_detector/faces/'

for file in os.listdir(known_dir):
    img = read_img(known_dir+'/'+file)
    img_enc = frec.face_encodings(img)[0]
    known_encodings.append(img_enc)
    known_names.append(file.split('.')[0])

unknown_dir = 'C:/Users/Admin/Desktop/Face_detector/test/'
for file in os.listdir(unknown_dir):
    print("processing : ",file)
    img = read_img(unknown_dir+"/"+file)
    img_enc = frec.face_encodings(img)[0]

    result = frec.compare_faces(known_encodings,img_enc)
    
    for i in range(len(result)):
        if result[i]:
            name = known_names[i]
            (t,r,b,l) = frec.face_locations(img)[0]
            cv2.rectangle(img,(l,t),(r,b),(0,255,0),2)
            cv2.putText(img,name,(l+2,b+20),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
            cv2.imshow('name',img)
