import boto3
import cv2

from credentials import ACCESS_KEY, SECRET_KEY


reko_client = boto3.client("rekognition",
                           aws_access_key_id=ACCESS_KEY,
                           aws_secret_access_key=SECRET_KEY,
                           region_name="us-east-1")

target_class = "Road Sign"

cap = cv2.VideoCapture("./road_sign.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
output_fps = fps * 2

output_video_file = "output_file_road_sign.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_file, fourcc, fps,
                      (int(cap.get(3)), int(cap.get(4))))

frame_num = -1

ret = True
while ret:
    ret, frame = cap.read()

    if ret:

        frame_num += 1
        H, W, _ = frame.shape

        _, buffer = cv2.imencode(".jpg", frame)

        image_bytes = buffer.tobytes()

        response = reko_client.detect_labels(Image={"Bytes": image_bytes},
                                             MinConfidence=50)

        for label in response["Labels"]:
            if label["Name"] == target_class:
                for instance_num in range(len(label["Instances"])):
                    bbox = label["Instances"][instance_num]["BoundingBox"]
                    x1 = int(bbox["Left"] * W)
                    y1 = int(bbox["Top"] * H)
                    width = int(bbox["Width"] * W)
                    height = int(bbox["Height"] * H)

                    cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 3)

                    out.write(frame)

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
