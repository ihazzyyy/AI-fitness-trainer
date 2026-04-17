# camera.py  —  OpenCV + YOLOv8 stream for Flask
import cv2, threading, time
import numpy as np
from ultralytics import YOLO
from utils.angle_calculator import calculate_angle, vertical_angle

KP = {
    "nose":0,"left_shoulder":5,"right_shoulder":6,
    "left_elbow":7,"right_elbow":8,
    "left_wrist":9,"right_wrist":10,
    "left_hip":11,"right_hip":12,
    "left_knee":13,"right_knee":14,
    "left_ankle":15,"right_ankle":16,
}

EXERCISES = ["Bicep Curl", "Squat", "Pushup", "Deadlift"]

class CameraStream:
    def __init__(self):
        self.model     = YOLO("yolov8n-pose.pt")
        self.cap       = None
        self.lock      = threading.Lock()
        self.frame     = None
        self.running   = False
        self.exercise  = "Bicep Curl"

        # Stats
        self.reps      = 0
        self.sets      = 0
        self.direction = 0
        self.counter   = 0.0
        self.feedback  = []
        self.correct   = True

        # direction tracking per exercise
        self.dir_L = 0
        self.dir_R = 0
        self.cnt_L = 0.0
        self.cnt_R = 0.0

        self._start()

    def _start(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            print("[CameraStream] Warning: CAP_DSHOW failed, trying default backend")
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("[CameraStream] ERROR: could not open camera")
            self.running = False
            self.frame = self._black_frame()
            return

        self.running = True
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = cv2.resize(frame, (854, 480))
            processed = self._process(frame)
            with self.lock:
                self.frame = processed

    def _process(self, frame):
        results = self.model(frame, verbose=False)
        kp_data = None
        if results and results[0].keypoints is not None:
            data = results[0].keypoints.data
            if len(data) > 0:
                kp_data = data[0]

        if kp_data is not None:
            self._draw_skeleton(frame, kp_data)
            fb, correct = self._analyze(frame, kp_data)
            with self.lock:
                self.feedback = fb
                self.correct  = correct

        self._draw_hud(frame)
        return frame

    def _analyze(self, frame, kp):
        feedback = []
        ex = self.exercise

        if ex == "Bicep Curl":
            lw=kp[KP["left_wrist"]]; le=kp[KP["left_elbow"]]; ls=kp[KP["left_shoulder"]]
            rw=kp[KP["right_wrist"]]; re=kp[KP["right_elbow"]]; rs=kp[KP["right_shoulder"]]
            la = calculate_angle(lw,le,ls)
            ra = calculate_angle(rw,re,rs)
            self._count_curl(la,"L"); self._count_curl(ra,"R")
            self.reps = int((self.cnt_L + self.cnt_R) / 2)
            ua_l = vertical_angle(ls, le)
            ua_r = vertical_angle(rs, re)
            if ua_l > 30 or ua_r > 30: feedback.append("Keep elbows pinned to sides")
            if abs(la - ra) > 35:       feedback.append("Keep both arms in sync")

        elif ex == "Squat":
            lh=kp[KP["left_hip"]]; lk=kp[KP["left_knee"]]; la_=kp[KP["left_ankle"]]
            rh=kp[KP["right_hip"]]; rk=kp[KP["right_knee"]]; ra_=kp[KP["right_ankle"]]
            ls=kp[KP["left_shoulder"]]
            lka = calculate_angle(lh,lk,la_)
            rka = calculate_angle(rh,rk,ra_)
            avg = (lka+rka)//2
            tl  = vertical_angle(lh, ls)
            if avg <= 100:
                if self.direction == 0: self.counter += 0.5; self.direction = 1
            if avg >= 160:
                if self.direction == 1: self.counter += 0.5; self.direction = 0
            self.reps = int(self.counter)
            if tl > 40: feedback.append("Keep chest up")
            if self.direction==1 and avg > 115: feedback.append("Squat deeper")
            lkx=float(lk[0]); lax=float(la_[0])
            rkx=float(rk[0]); rax=float(ra_[0])
            if lkx - lax > 30: feedback.append("Left knee caving in")
            if rax - rkx > 30: feedback.append("Right knee caving in")

        elif ex == "Pushup":
            ls=kp[KP["left_shoulder"]]; le=kp[KP["left_elbow"]]; lw=kp[KP["left_wrist"]]
            rs=kp[KP["right_shoulder"]]; re=kp[KP["right_elbow"]]; rw=kp[KP["right_wrist"]]
            lh=kp[KP["left_hip"]]; la_=kp[KP["left_ankle"]]
            lea = calculate_angle(ls,le,lw)
            rea = calculate_angle(rs,re,rw)
            avg = (lea+rea)//2
            ba  = calculate_angle(ls,lh,la_)
            if avg <= 90:
                if self.direction == 0: self.counter += 0.5; self.direction = 1
            if avg >= 160:
                if self.direction == 1: self.counter += 0.5; self.direction = 0
            self.reps = int(self.counter)
            if ba < 155: feedback.append("Hips sagging — keep body straight")
            if self.direction==0 and avg < 140: feedback.append("Extend arms fully")
            if self.direction==1 and avg > 110: feedback.append("Lower chest to floor")

        elif ex == "Deadlift":
            ls=kp[KP["left_shoulder"]]; lh=kp[KP["left_hip"]]
            lk=kp[KP["left_knee"]];     la_=kp[KP["left_ankle"]]
            rs=kp[KP["right_shoulder"]]; rh=kp[KP["right_hip"]]
            rk=kp[KP["right_knee"]]
            nose=kp[KP["nose"]]
            lha = calculate_angle(ls,lh,lk)
            rha = calculate_angle(rs,rh,rk)
            avg = (lha+rha)//2
            lka = calculate_angle(lh,lk,la_)
            sp  = calculate_angle(nose,ls,lh)
            if avg <= 60:
                if self.direction == 0: self.counter += 0.5; self.direction = 1
            if avg >= 160:
                if self.direction == 1: self.counter += 0.5; self.direction = 0
            self.reps = int(self.counter)
            if sp < 140:  feedback.append("Keep back flat — no rounding")
            if lka < 80:  feedback.append("Push hips back — not a squat")
            if self.direction==0 and avg < 145: feedback.append("Stand tall — lock out hips")

        return feedback, len(feedback) == 0

    def _count_curl(self, angle, side):
        if side == "L":
            if angle >= 160 and self.dir_L == 1: self.cnt_L += 0.5; self.dir_L = 0
            if angle <= 70  and self.dir_L == 0: self.cnt_L += 0.5; self.dir_L = 1
        else:
            if angle >= 160 and self.dir_R == 1: self.cnt_R += 0.5; self.dir_R = 0
            if angle <= 70  and self.dir_R == 0: self.cnt_R += 0.5; self.dir_R = 1

    def _draw_skeleton(self, frame, kp):
        pairs = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
                 (11,12),(11,13),(13,15),(12,14),(14,16)]
        for a,b in pairs:
            x1,y1 = int(kp[a][0]),int(kp[a][1])
            x2,y2 = int(kp[b][0]),int(kp[b][1])
            if all(v>0 for v in [x1,y1,x2,y2]):
                cv2.line(frame,(x1,y1),(x2,y2),(57,255,20),2)
        for i in range(17):
            x,y = int(kp[i][0]),int(kp[i][1])
            if x>0 and y>0:
                cv2.circle(frame,(x,y),5,(57,255,20),-1)
                cv2.circle(frame,(x,y),8,(255,255,255),1)

    def _draw_hud(self, frame):
        h,w = frame.shape[:2]
        # top bar
        overlay = frame.copy()
        cv2.rectangle(overlay,(0,0),(w,56),(0,0,0),-1)
        cv2.addWeighted(overlay,0.6,frame,0.4,0,frame)
        # neon green accent line
        cv2.line(frame,(0,56),(w,56),(57,255,20),1)
        # exercise name
        cv2.putText(frame, self.exercise.upper(),
                    (16,36), cv2.FONT_HERSHEY_DUPLEX, 0.9,(57,255,20),2,cv2.LINE_AA)
        # reps
        cv2.putText(frame, f"REPS: {self.reps}",
                    (w-200,36), cv2.FONT_HERSHEY_DUPLEX, 0.9,(255,255,255),2,cv2.LINE_AA)
        # form badge
        color  = (57,255,20) if self.correct else (0,60,220)
        label  = "GOOD FORM" if self.correct else "FIX FORM"
        cv2.rectangle(frame,(w-170,64),(w-10,96),color,-1)
        cv2.putText(frame,label,(w-162,86),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv2.LINE_AA)
        # feedback tips
        for i,tip in enumerate(self.feedback[:3]):
            y = 116 + i*26
            tw = cv2.getTextSize(tip,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)[0][0]
            cv2.rectangle(frame,(w-tw-20,y-18),(w-6,y+4),(20,20,20),-1)
            cv2.putText(frame,tip,(w-tw-14,y),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,210,255),1,cv2.LINE_AA)

    def _black_frame(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        text = "Camera unavailable"
        tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        x = (frame.shape[1] - tw) // 2
        y = (frame.shape[0] + th) // 2
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (57,255,20), 2, cv2.LINE_AA)
        return frame

    def generate_frames(self):
        while True:
            with self.lock:
                frame = self.frame
            if frame is None:
                time.sleep(0.03)
                continue
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            time.sleep(0.03)

    def set_exercise(self, exercise):
        with self.lock:
            self.exercise  = exercise
            self.counter   = 0.0
            self.reps      = 0
            self.direction = 0
            self.cnt_L = self.cnt_R = 0.0
            self.dir_L = self.dir_R = 0
            self.feedback  = []

    def get_stats(self):
        with self.lock:
            return {
                "reps":     self.reps,
                "sets":     self.sets,
                "exercise": self.exercise,
                "correct":  self.correct,
                "feedback": self.feedback,
            }

    def reset(self):
        with self.lock:
            self.counter = self.reps = self.sets = 0
            self.cnt_L = self.cnt_R = 0.0
            self.dir_L = self.dir_R = 0
            self.direction = 0
            self.feedback  = []