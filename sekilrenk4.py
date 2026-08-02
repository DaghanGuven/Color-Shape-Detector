import cv2
import numpy as np

def dinamik_renk_esigi(kare, hedef_renk_alt, hedef_renk_ust):
    """Hedef renge gore dinamik renk esigi uygula."""
    hsv_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)
    maske = cv2.inRange(hsv_kare, hedef_renk_alt, hedef_renk_ust)
    return maske

def kontur_filtresi(konturlar, min_alan):
    """Minimum alana gore konturlari filtrele."""
    filtreli_konturlar = [kontur for kontur in konturlar if cv2.contourArea(kontur) > min_alan]
    return filtreli_konturlar

def renk_tespiti(hsv_renk):
    """HSV degerlerine gore rengi belirle."""
    # HSV'de renk araliklarini tanimla
    mavi_aralik = (np.array([100, 80, 50]), np.array([140, 255, 255]))
    kirmizi_aralik = (np.array([0, 100, 100]), np.array([10, 255, 255]))
    sari_aralik = (np.array([20, 100, 100]), np.array([30, 255, 255]))
    yesil_aralik = (np.array([40, 100, 100]), np.array([80, 255, 255]))
    turuncu_aralik = (np.array([10, 100, 100]), np.array([20, 255, 255]))
    mor_aralik = (np.array([120, 100, 100]), np.array([160, 255, 255]))
    pembe_aralik = (np.array([160, 100, 100]), np.array([180, 255, 255]))

    if mavi_aralik[0][0] <= hsv_renk[0] <= mavi_aralik[1][0] and \
       mavi_aralik[0][1] <= hsv_renk[1] <= mavi_aralik[1][1] and \
       mavi_aralik[0][2] <= hsv_renk[2] <= mavi_aralik[1][2]:
        return "Mavi"
    elif kirmizi_aralik[0][0] <= hsv_renk[0] <= kirmizi_aralik[1][0] and \
         kirmizi_aralik[0][1] <= hsv_renk[1] <= kirmizi_aralik[1][1] and \
         kirmizi_aralik[0][2] <= hsv_renk[2] <= kirmizi_aralik[1][2]:
        return "Kirmizi"
    elif sari_aralik[0][0] <= hsv_renk[0] <= sari_aralik[1][0] and \
         sari_aralik[0][1] <= hsv_renk[1] <= sari_aralik[1][1] and \
         sari_aralik[0][2] <= hsv_renk[2] <= sari_aralik[1][2]:
        return "Sari"
    elif yesil_aralik[0][0] <= hsv_renk[0] <= yesil_aralik[1][0] and \
         yesil_aralik[0][1] <= hsv_renk[1] <= yesil_aralik[1][1] and \
         yesil_aralik[0][2] <= hsv_renk[2] <= yesil_aralik[1][2]:
        return "Yesil"
    elif turuncu_aralik[0][0] <= hsv_renk[0] <= turuncu_aralik[1][0] and \
         turuncu_aralik[0][1] <= hsv_renk[1] <= turuncu_aralik[1][1] and \
         turuncu_aralik[0][2] <= hsv_renk[2] <= turuncu_aralik[1][2]:
        return "Turuncu"
    elif mor_aralik[0][0] <= hsv_renk[0] <= mor_aralik[1][0] and \
         mor_aralik[0][1] <= hsv_renk[1] <= mor_aralik[1][1] and \
         mor_aralik[0][2] <= hsv_renk[2] <= mor_aralik[1][2]:
        return "Mor"
    elif pembe_aralik[0][0] <= hsv_renk[0] <= pembe_aralik[1][0] and \
         pembe_aralik[0][1] <= hsv_renk[1] <= pembe_aralik[1][1] and \
         pembe_aralik[0][2] <= hsv_renk[2] <= pembe_aralik[1][2]:
        return "Pembe"
    else:
        return "Tanimsiz"

def sekil_ve_renk_tespiti(kontur, hsv_kare):
    """Verilen konturun sekil ve rengini belirle."""
    epsilon = 0.02 * cv2.arcLength(kontur, True)
    yaklasik = cv2.approxPolyDP(kontur, epsilon, True)
    kenar_sayisi = len(yaklasik)

    renk = renk_tespiti(hsv_kare[int(kontur.mean(axis=0)[:, 1]), int(kontur.mean(axis=0)[:, 0])])

    if kenar_sayisi == 3:
        return "Ucgen", renk
    elif kenar_sayisi == 4:
        x, y, w, h = cv2.boundingRect(yaklasik)
        en_boy_orani = float(w) / h

        if 0.8 <= en_boy_orani <= 1.2:
            return "Kare", renk
        else:
            return "Dikdortgen", renk
    elif kenar_sayisi == 5:
        return "Besgen", renk
    elif kenar_sayisi == 6:
        return "Altigen", renk
    else:
        (x, y), yaricap = cv2.minEnclosingCircle(yaklasik)
        yuvarlaklik = cv2.contourArea(yaklasik) / (np.pi * yaricap ** 2)

        return "Daire", renk if yuvarlaklik >= 0.6 else "Tanimsiz"

def kare_isleme(kare):
    """Her kareyi isleyerek sekil ve renk tespiti yap."""
    hsv_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)

    # Farkli renkler icin HSV'de hedef renk araliklarini tanimla
    mavi_alt = np.array([100, 80, 50])
    mavi_ust = np.array([140, 255, 255])

    kirmizi_alt = np.array([0, 100, 100])
    kirmizi_ust = np.array([10, 255, 255])

    sari_alt = np.array([20, 100, 100])
    sari_ust = np.array([30, 255, 255])

    yesil_alt = np.array([40, 100, 100])
    yesil_ust = np.array([80, 255, 255])

    turuncu_alt = np.array([10, 100, 100])
    turuncu_ust = np.array([20, 255, 255])

    mor_alt = np.array([120, 100, 100])
    mor_ust = np.array([160, 255, 255])

    pembe_alt = np.array([160, 100, 100])
    pembe_ust = np.array([180, 255, 255])

    # Her renk icin renk esigi uygula
    mavi_maske = dinamik_renk_esigi(kare, mavi_alt, mavi_ust)
    kirmizi_maske = dinamik_renk_esigi(kare, kirmizi_alt, kirmizi_ust)
    sari_maske = dinamik_renk_esigi(kare, sari_alt, sari_ust)
    yesil_maske = dinamik_renk_esigi(kare, yesil_alt, yesil_ust)
    turuncu_maske = dinamik_renk_esigi(kare, turuncu_alt, turuncu_ust)
    mor_maske = dinamik_renk_esigi(kare, mor_alt, mor_ust)
    pembe_maske = dinamik_renk_esigi(kare, pembe_alt, pembe_ust)

    # Farkli renk maskelerini birlestir
    maske = cv2.bitwise_or(cv2.bitwise_or(mavi_maske, kirmizi_maske, sari_maske),
                           cv2.bitwise_or(yesil_maske, turuncu_maske, mor_maske, pembe_maske))

    kernel = np.ones((5, 5), np.uint8)
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, kernel)
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, kernel)

    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtreli_konturlar = kontur_filtresi(konturlar, min_kontur_alani)

    for kontur in filtreli_konturlar:
        sekil, renk = sekil_ve_renk_tespiti(kontur, hsv_kare)

        if renk != "Tanimsiz":  # Eger renk tanimliysa ve "Tanimsiz" degilse, o zaman ciz
            x, y, w, h = cv2.boundingRect(kontur)

            if sekil != "Daire":
                cv2.rectangle(kare, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if sekil == "Daire":
                cv2.circle(kare, (int(x + w / 2), int(y + h / 2)), int((w + h) / 4), (255, 0, 0), 2)

            cv2.putText(kare, f"{renk} {sekil}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return kare

# Kamera baglantisini ac
kamera = cv2.VideoCapture(0)

# Kare boyutlarini ayarla (1920x1080)
kamera.set(3, 1920)
kamera.set(4, 1080)

# Minimum kontur alanini ayarla
min_kontur_alani = 50  # Bunu cok kucuk yaparsaniz, 'gurultu' olarak bircok gereksiz kucuk nesneyi alir

while True:
    ret, kare = kamera.read()

    islenmis_kare = kare_isleme(kare)
    cv2.imshow("Nesne Tespiti", islenmis_kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
