Haar cascade bir resim veya videodaki objeleri tanımlamak için kullanılan metottur. Bu metot görüntüdeki pixelleri kareler halinde analiz ederek tanımlamayı gerçekleştirir. Görsel inputun farklı lokasyonlarına Haar özellikleri uygulanarak algoritma ilerler ve böylece tanımlanacak objeye adım adım yaklaşılmış olunur.

Source: https://www.youtube.com/watch?v=PmZ29Vta7Vc http://www.willberger.org/cascade-haar-explained/

dnn OpenCV içerisinde yer alan, görüntü dataları için preprocessing yaparken ve önceden eğitilmiş deep learning modelleriyle sınıflandırma yaparken kullanabildiğimiz bir modüldür.
Caffe, Darknet, Tensorflow gibi modelleri de dnn modülüyle birlikte kullanabiliyoruz.
detect_faces_image.py kodunu çalıştırmak için komut:

python detect_faces_image.py --image ../../images/sad.jpg --prototxt ./model/deploy.prototxt.txt --model model/res10_300x300_ssd_iter_140000.caffemodel

python detect_faces_video.py --prototxt ./model/deploy.prototxt.txt --model model/res10_300x300_ssd_iter_140000.caffemodel 
Source: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
