from flask import Flask,render_template,Response
import cv2

app = Flask(__name__)
cap=cv2.VideoCapture(0)
def cap1():
    while True:
        ret,frame=cap.read()
        frame=cv2.flip(frame,1)
        rete, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(cap1(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)

