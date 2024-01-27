import cv2
import numpy as np

def process_frame(frame):

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Çerçeveyi HSV'ye dönüştürür.

    lower_red = np.array([0, 100, 100]) #Kırmızı için renk aralığı
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([50, 50, 50]) #Yeşil için renk aralığı
    upper_green = np.array([70, 255, 255])

    lower_blue = np.array([110, 50, 50]) #Mavi için renk aralığı
    upper_blue = np.array([130, 255, 255])


    mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)
    mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    process_contours(contours_red, frame, (0, 0, 255), "Red") #Kırmızı algılanınca, çerçeve kırmızı olur.
    process_contours(contours_green, frame, (0, 255, 0), "Green") #Yeşil algılanınca, çerçeve yeşil olur.
    process_contours(contours_blue, frame, (255, 0, 0), "Blue") #Mavi algılanınca, çerçeve mavi olur.

    
    cv2.imshow("Color Detection", frame) #Açılan pencerenin ismi "Color Detection" yapar.

def process_contours(contours, frame, color, label):
    for contour in contours:
        #Önemsiz, küçük kontürleri görmezden gelir.
        if cv2.contourArea(contour) > 100:

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) #Dikdörtgen çizer.

            center_x = x + w // 2 #Gösterilen rengin, kameradaki koordinatlarını hesaplar.
            center_y = y + h // 2

            cv2.putText(frame, f"{label} ({center_x}, {center_y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    #Video kaydını açar.
    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('k'): #k tuşuna basınca programı kapatır.
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
