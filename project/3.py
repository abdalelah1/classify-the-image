import cv2
img = cv2.imread('images/employee.png')
classnames = [] 
classfile  = 'files/thing.names'
with open(classfile, 'rt') as f :
    classnames = f.read().rstrip('\n').split('\n')
print(classnames)
cv2.imshow('picture', img)
cv2.waitKey(0)


