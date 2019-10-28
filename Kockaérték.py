# szükséges csomagok importálása
import cv2
import numpy as np

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
felv = cv2.VideoCapture(0)
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
 
# jobb értelmezhetőség miatt kirajolom itt a pontokat
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 255, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#képernyőre való kijelzés
    cv2.imshow("Kocka ertek szam", im_with_keypoints)

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
            msg = str(kij[-1])
            print(msg)
#növelem a számláló értékét 
    szamlalo += 1
# kilépés space esetén
    k = cv2.waitKey(30) & 0xff
    if k == 32:
        break

cv2.destroyAllWindows()                                     # since we exited the loop above, end the script.
felv.release()