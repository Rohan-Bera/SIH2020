from flask import Flask, render_template, session
from flask_socketio import SocketIO, emit
import random

import cv2
import numpy as np
from PIL import Image
import io
import time

from gestureDetection import detectFingerCount


TOT_FRAMES = 120    # total frames to process

app = Flask(__name__)
app.config['SECRET_KEY'] = 'S3Cr3T_kEy'
socketio = SocketIO(app)

@app.route('/')
def webpage():
    return render_template("demowebsite.html")

@socketio.on("captcha_req")
def generate_captcha_challange():
    print("captcha_challange request recieved!")
    captcha = [random.randint(0, 5), random.randint(0, 5), random.randint(0, 5), random.randint(0, 5)]
    session['captcha'] = captcha
    session['captcha_solved'] = False
    session['fingercounts'] = []
    emit("captcha_recv", {'digits': captcha, 'tot_frames': TOT_FRAMES})


@socketio.on("abrupt_close")
def abrupt_close():
    print("captcha window closed abruptly!")
    session.pop("framecount")
    session.pop("fingercounts")

@socketio.on('send_frame')
def socket_image(json_data):
    # convert image in format usable by opencv
    img = Image.open(io.BytesIO(json_data['img'])).convert('RGB')
    img = np.array(img)    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # do finger detection
    start_time = time.process_time()
    fingercount, img = detectFingerCount(img)
    end_time = time.process_time()

    # store the detected fingercount
    fingercounts = session['fingercounts']
    fingercounts.append(fingercount)
    framecount = len(fingercounts)
    session['fingercounts'] = fingercounts

    print("Frame %03d, Time taken %.5f ms, Detected: %d"%(framecount, end_time-start_time, fingercounts[-1]))

    # check if required frames have been recieved
    if framecount >= TOT_FRAMES:        
        detected = getModes(fingercounts)
        print("Detected: ", detected)

        # Captcha SOLVED
        if detected == session['captcha']:
            session['captcha_solved'] = True
            emit("complete", {"solved": True})
            print("Captcha Solved!")

        # captcha FAILED
        else:
            emit("complete", {"solved": False})
            print("Captcha Failed!")

    # send output image for testing purposes
    _, encodedImage = cv2.imencode(".jpg", img) 
    emit('recv_output_frame', {'img':encodedImage.tobytes()})


def getModes(fingercounts):
    frames_per_digit = int(TOT_FRAMES/4)
    d1 = mode([n for n in fingercounts[0:frames_per_digit] if n != -1])
    d2 = mode([n for n in fingercounts[frames_per_digit:frames_per_digit*2] if n != -1])
    d3 = mode([n for n in fingercounts[frames_per_digit*2:frames_per_digit*3] if n != -1])
    d4 = mode([n for n in fingercounts[frames_per_digit*3:frames_per_digit*4] if n != -1])

    return [d1, d2, d3, d4]
    

def mode(count_list):
    if(len(count_list)==0):
        return -1

    freq = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for n in count_list:
        freq[n] += 1
    
    return max(freq.keys(), key=(lambda k: freq[k]))



if __name__ == "__main__":
    socketio.run(app, debug=True)