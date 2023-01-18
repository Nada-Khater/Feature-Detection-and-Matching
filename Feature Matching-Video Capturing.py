import cv2

# Create SIFT object
sift = cv2.SIFT_create()

# Feature Matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():

    suc, img1 = cap.read()
    img1 = cv2.resize(img1, (400, 400))

    # image to video matching
    img2 = cv2.imread('pic.jpg')
    img2 = cv2.resize(img2, (300, 400))

    # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Detect Features in both frames
    kps1, desc1 = sift.detectAndCompute(img1, None)
    kps2, desc2 = sift.detectAndCompute(img2, None)

    # Match Common Features
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    print("Raw Matches:", len(matches))

    # Draw Matches
    img = cv2.drawMatches(img1, kps1, img2, kps2, matches[:100], None, flags=2)
    cv2.putText(img, f'Raw Matches: {len(matches)}', (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('SIFT', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
