# DrowsinessDetectionSystem
Team Members:
Gaurav Ahuja 16103016
Prayag Panta 16103003
Gurvir Singh 16103085
Kanishk Gautam 16103118


This is a Drowsiness Alert System , meant for detecting a drowsy driver and alarming the driver to prevent accidents

Detects Drowsiness by considering 2 factors:
1. Closing Eyes
2. Yawning

Closing Eyes are detected by calculating Aspect Ratio of the eye for each VideoFrame captured. Once the aspect ratio falls below some 
threshold, then it generates an alarm when given number of consecutive video frames exceed the given limit.

Yawning is detected similarly by calculating aspect ratio for the mouth.
