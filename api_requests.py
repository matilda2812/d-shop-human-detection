''' Module to send an image to SAP Human Detection API
    Works with trial or production Cloud Foundry Account
    Replace placeholder credentials and url before use
    Author: Matilda Stevenson, Presales Development Intern
    Contact: matilda.stevenson@sap.com
    Date: 7/11/18
'''

import requests
from requests.auth import HTTPBasicAuth
import base64
import os
import cv2
import numpy as np


AUTH_URL = YOUR_AUTH_URL_HERE

DETECTION_URL = YOUR DETECTION_URL_HERE

CLIENT_ID = YOUR_CLIENT_ID_HERE

CLIENT_SECRET = YOUR_CLIENT_SECRET_HERE


# get current working directory
cwd = os.getcwd()

# Request oAuth token from SAP using Cloud Foundy ML credentials
def get_authorisation():

    auth = HTTPBasicAuth(CLIENTID,CLIENT_SECRET)

    # send request
    r = requests.get(AUTH_URL,auth=auth)

    # extract token from response
    token = r.json()['access_token']

    return token

# Send image to be analaysed to SAP Inference Service for Human Detection
# Returns coordinates of boxes for each person in image
def send_image_from_video(img_path, img_name, img_result=False):
    token = get_authorisation()
    headers = {'Authorization': 'Bearer ' + token}
    files = {'file': (img_path,open(img_name,'rb'),'image/jpg')}

    if img_result:
        url = DETECTION_URL + "/format:image"
    else:
        url = DETECTION_URL + "/"
    print(url)
    r = requests.post(url, headers=headers,files=files)
    if r.status_code == 200:
        print(r.json())
        boxes = r.json()['detection_boxes']
        return boxes

    else:
        print(REQUEST_ERROR)
