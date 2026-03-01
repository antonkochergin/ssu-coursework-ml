











# Удаление фона c помощью библеотеки
# result_pil = remove(img)
# result_np = np.array(result_pil)  #перевод на язык OpenCV
# gray = cv2.cvtColor(result_np, cv2.COLOR_BGR2GRAY)






 # blurred = cv2.GaussianBlur(gray , (11,11), 0)
# edges = cv2.Canny(blurred, 30, 40)


# # # Обводка
# kernel = np.ones((7, 7), np.uint8)
# img = cv2.dilate(img, kernel, iterations=1)
# cv2.imshow("Result", gray)