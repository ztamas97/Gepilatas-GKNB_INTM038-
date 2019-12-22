# szükséges csomagok importálása
import cv2
import numpy as np
import matplotlib.pyplot as plt

#a blob detektálás konstans értékeinek beállítása
##küszöbölés értékei
min_kuszob = 10       
max_kuszob = 200
##a blob méret szenti szüréséhez
min_terulet = 50
##alak szerinti
###inerciális arány
min_inercia_arany = .5
###körkörösségi érték
min_kor = .5

# cap változó a videó objektum eléréséhez
felv = cv2.VideoCapture(1)
#az élőkép fényerejének inicializálása, beállítása
felv.set(15, -4)

#FPS érték kezeléséhez szükséges számláló, valamint listák
szamlalo = 0
olv = [0, 0]
kij = [0, 0]
 
while True:
# listák, változók értékeinek inicializálása
    if szamlalo >= 90000:                
        szamlalo = 0
        olv = [0, 0]
        kij = [0, 0]

# im ben beállítja a frameket
    ret, im = felv.read()                                    

# a filter paramétereinek definiciói
    params = cv2.SimpleBlobDetector_Params()                
    params.filterByArea = True
    params.filterByCircularity = True
    params.filterByInertia = True
    params.minThreshold = min_kuszob
    params.maxThreshold = max_kuszob
    params.minArea = min_terulet
    params.minCircularity = min_kor
    params.minInertiaRatio = min_inercia_arany

# blob detektor objektum létrehozása adott frame esetén, megfelelő paraméterekkel
    detektor = cv2.SimpleBlobDetector_create(params) 
 
# keyponts mint egy lista melyben a pontok megtalálhatóak   
    keypoints = detektor.detect(im)

# jobb értelmezhetőség miatt kirajzolom itt a pontokat
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 255, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# SZÜRKEÁRMYALATOSKÉP
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(imgray,(5, 5),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,7)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imout=cv2.drawContours(im, contours, -1, (0,255,0), 3)
    #print("Number of Contours found = " + str(len(contours))) 

#képernyőre való kijelzés
    cv2.imshow("Kocka ertek", im_with_keypoints)
    cv2.imshow("Kocka szam", imout)


# olv1 tárolja a keypontok számát
    olv1 = len(keypoints)                                

# minden 10. olvasásesetén, megjegyzem a keypontok számát
    if szamlalo % 10 == 0:
        olv.append(olv1)              
## ha az utolsó három olvasott érték megegyezik, validálom az értéket
        if olv[-1] == olv[-2] == olv[-3]:
            kij.append(olv[-1])                    

## ha a legfrisebb olvasás értéke megfelelő(>0) megtörténik egy konverzió, valamint kiíratá a képernyőre
        if kij[-1] != kij[-2] and kij[-1] != 0:
            msg = str(kij[-1])+" érték "+str(int((((len(contours)-kij[-1])/4)+1)/3))+" kockán "
            print(msg)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            Z = im.reshape((-1,3))
            Z = np.float32(Z)
            K = int((len(contours)-kij[-1])/4)+1
            ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            # Now separate the data, Note the flatten()
            A = Z[label.ravel()==0]
            B = Z[label.ravel()==1]
            # Plot the data
            plt.scatter(A[:,0],A[:,1])
            plt.scatter(B[:,0],B[:,1],c = 'r')
            plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
            plt.xlabel('Height'),plt.ylabel('Weight')
            plt.show()
#növelem a számláló értékét 
    szamlalo += 1
# kilépés space esetén
    k = cv2.waitKey(30) & 0xff
    if k == 32:
        break

cv2.destroyAllWindows()
felv.release()