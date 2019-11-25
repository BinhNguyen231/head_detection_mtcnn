import math
import cv2
import numpy as np
from imutils.video import FPS
import imutils
import dlib
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject

def gen_bbox(hotmap, offset, scale, th):
	h, w = hotmap.shape
	stride = 2
	win_size = 12
	hotmap = hotmap.reshape((h, w))
	keep = hotmap > th
	pos = np.where(keep)
	score = hotmap[keep]
	offset = offset[:, keep]
	x, y = pos[1], pos[0]
	x1 = stride * x
	y1 = stride * y
	x2 = x1 + win_size
	y2 = y1 + win_size
	x1 = x1 / scale
	y1 = y1 / scale
	x2 = x2 / scale
	y2 = y2 / scale
	bbox = np.vstack([x1, y1, x2, y2, score, offset]).transpose()
	return bbox.astype(np.float32)

def nms(dets, thresh, meth='Union'):
	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 2]
	y2 = dets[:, 3]
	scores = dets[:, 4]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]
	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		if meth == 'Union':
			ovr = inter / (areas[i] + areas[order[1:]] - inter)
		else:
			ovr = inter / np.minimum(areas[i], areas[order[1:]])
		inds = np.where(ovr <= thresh)[0]
		order = order[inds + 1]
	return keep

def bbox_reg(bboxes):
	w = bboxes[:, 2] - bboxes[:, 0]
	h = bboxes[:, 3] - bboxes[:, 1]
	bboxes[:, 0] += bboxes[:, 5] * w
	bboxes[:, 1] += bboxes[:, 6] * h
	bboxes[:, 2] += bboxes[:, 7] * w
	bboxes[:, 3] += bboxes[:, 8] * h
	return bboxes

def make_square(bboxes):
	x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2
	y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2
	w = bboxes[:, 2] - bboxes[:, 0]
	h = bboxes[:, 3] - bboxes[:, 1]
	size = np.vstack([w, h]).max(axis=0).transpose()
	bboxes[:, 0] = x_center - size / 2
	bboxes[:, 2] = x_center + size / 2
	bboxes[:, 1] = y_center - size / 2
	bboxes[:, 3] = y_center + size / 2
	return bboxes

def crop_face(img, bbox, wrap=True):
	height, width = img.shape[:-1]
	x1, y1, x2, y2 = bbox
	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
	if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
		print ('[WARN] ridiculous x1, y1, x2, y2')
		return None
	if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
		# out of boundary, still crop the face
		if not wrap:
			return None
		h, w = y2 - y1, x2 - x1
		patch = np.zeros((h, w, 3), dtype=np.uint8)
		vx1 = 0 if x1 < 0 else x1
		vy1 = 0 if y1 < 0 else y1
		vx2 = width if x2 > width else x2
		vy2 = height if y2 > height else y2
		sx = -x1 if x1 < 0 else 0
		sy = -y1 if y1 < 0 else 0
		vw = vx2 - vx1
		vh = vy2 - vy1
		patch[sy:sy+vh, sx:sx+vw] = img[vy1:vy2, vx1:vx2]
		return patch
	return img[y1:y2, x1:x2]

def mtcnn_detection(img, scales, width, height):
	### pnet ###
	bboxes_in_all_scales = np.zeros((0, 4 + 1 + 4), dtype=np.float32)
	for scale in scales:
		w, h = int(math.ceil(scale * width)), int(math.ceil(scale * height))
		data = cv2.resize(img, (w, h))

		blob = cv2.dnn.blobFromImage(data, 1./128, (w,h), (128,128,128), False)
		pnet.setInput(blob)
		prob = pnet.forward("prob")
		bbox_pred = pnet.forward()

		bboxes = gen_bbox(prob[0][1], bbox_pred[0], scale, 0.6)
		keep = nms(bboxes, 0.5)  # nms in each scale
		bboxes = bboxes[keep]
		bboxes_in_all_scales = np.vstack([bboxes_in_all_scales, bboxes])

	# nms in total
	keep = nms(bboxes_in_all_scales, 0.7)
	bboxes_in_all_scales = bboxes_in_all_scales[keep]
	bboxes_in_all_scales = bbox_reg(bboxes_in_all_scales)
	bboxes_in_all_scales = make_square(bboxes_in_all_scales)
	if len(bboxes_in_all_scales) == 0:
		return bboxes_in_all_scales


	### rnet ###
	n = len(bboxes_in_all_scales)
	#data = np.zeros((n, 3, 24, 24), dtype=np.float32)
	blob = np.zeros((n, 3, 24, 24), dtype=np.float32)
	for i, bbox in enumerate(bboxes_in_all_scales):
		face = crop_face(img, bbox[:4])
		img_rnet = cv2.resize(face, (24, 24))
		blob[i] = cv2.dnn.blobFromImage(img_rnet, 0.00781, (24, 24), (128, 128, 128))


	rnet.setInput(blob)
	prob = rnet.forward("prob")
	bbox_pred = rnet.forward()


	prob = prob.reshape(n, 2)
	bbox_pred = bbox_pred.reshape(n, 4)
	keep = prob[:, 1] > 0.7
	bboxes_in_all_scales = bboxes_in_all_scales[keep]
	bboxes_in_all_scales[:, 4] = prob[keep, 1]
	bboxes_in_all_scales[:, 5:9] = bbox_pred[keep]
	keep = nms(bboxes_in_all_scales, 0.7)
	bboxes_in_all_scales = bboxes_in_all_scales[keep]
	bboxes_in_all_scales = bbox_reg(bboxes_in_all_scales)
	bboxes_in_all_scales = make_square(bboxes_in_all_scales)
	if len(bboxes_in_all_scales) == 0:
		return bboxes_in_all_scales

	### onet ###
	n = len(bboxes_in_all_scales)
	#data = np.zeros((n, 3, 48, 48), dtype=np.float32)
	blob = np.zeros((n, 3, 48, 48), dtype=np.float32)
	for i, bbox in enumerate(bboxes_in_all_scales):
		face = crop_face(img, bbox[:4])
		img_onet = cv2.resize(face, (48, 48))
		blob[i] = cv2.dnn.blobFromImage(img_onet, 0.00781, (48, 48), (128, 128, 128))


	onet.setInput(blob)
	prob = onet.forward("prob")
	bbox_pred = onet.forward()

	prob = prob.reshape(n, 2)
	bbox_pred = bbox_pred.reshape(n, 4)
	keep = prob[:, 1] > 0.4
	bboxes_in_all_scales = bboxes_in_all_scales[keep]
	bboxes_in_all_scales[:, 4] = prob[keep, 1]
	bboxes_in_all_scales[:, 5:9] = bbox_pred[keep]
	bboxes_in_all_scales = bbox_reg(bboxes_in_all_scales)
	keep = nms(bboxes_in_all_scales, 0.5, 'Min')
	bboxes_in_all_scales = bboxes_in_all_scales[keep]
	return bboxes_in_all_scales




pnet = cv2.dnn.readNetFromCaffe('proto/p.prototxt', 'tmp/pnet_iter_446000.caffemodel')
rnet = cv2.dnn.readNetFromCaffe('proto/r.prototxt', 'tmp/rnet_iter_116000.caffemodel')
onet = cv2.dnn.readNetFromCaffe('proto/o.prototxt', 'tmp/onet_iter_90000.caffemodel')

#pnet = caffe.Net('proto/p.prototxt', 'tmp/pnet_iter_446000.caffemodel' , caffe.TEST)
#rnet = caffe.Net('proto/r.prototxt', 'tmp/rnet_iter_116000.caffemodel' , caffe.TEST)
#onet = caffe.Net('proto/o.prototxt', 'tmp/onet_iter_90000.caffemodel' , caffe.TEST)

cap = cv2.VideoCapture("head_detection_subfile.mp4")
ret, img = cap.read()
img_resized = imutils.resize(img,  height=384)
img_resized = img_resized[0:384, 85:597]
img_crop = img_resized[120:312, 104:360]
#MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
min_size = 24
factor = 0.509
base = 12. / min_size
height, width = img_crop.shape[:-1]
l = min(width, height)
l *= base
scales = []
trackers = []
while l > 12:
	scales.append(base)
	base *= factor
	l *= factor

fps = FPS().start()

ct = CentroidTracker(maxDisappeared=20, maxDistance=50)
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
total_frame = 0
totalDown = 0
totalUp = 0

while(True):

	ret, img = cap.read()
	if (img is None):
		break
	img_resized = imutils.resize(img, height=384)
	img_resized = img_resized[0:384, 85:597]
	img_crop = img_resized[120:312, 104:360]
	img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
	cv2.rectangle(img_resized, (104,120), (360, 312), (0,255,0),2)
	status = "Waiting"
	rects = []
	if total_frame%5 == 0:
		status = "Detecting"
		trackers = []
		onet_boxes = mtcnn_detection(img_crop, scales, width, height)
		imgdraw_onet = img_resized.copy()
		for i in range(len(onet_boxes)):
			x1, y1, x2, y2, score = onet_boxes[i, :5]
			x1, y1, x2, y2 = int(x1 + 104), int(y1+120), int(x2+104), int(y2+120)
			if(score >= 0.65):
				#cv2.rectangle(imgdraw_onet, (x1, y1), (x2, y2), (0, 0, 255), 2)
				#cv2.putText(imgdraw_onet, '%.03f'%score, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
				t = dlib.correlation_tracker()
				rect = dlib.rectangle(x1, y1, x2, y2)
				t.start_track(img_resized, rect)
				trackers.append(t)

	else:
		for tracker in trackers:
			status = "Tracking"
			tracker.update(img_resized)
			pos = tracker.get_position()

			# unpack the position object
			x1 = int(pos.left())
			y1 = int(pos.top())
			x2 = int(pos.right())
			y2 = int(pos.bottom())
			rects.append((x1,y1,x2,y2))
			cv2.rectangle(img_resized, (x1,y1), (x2, y2), (0,0,255), 2)

	cv2.line(img_resized, (104, 216 ), (360, 216), (0, 255, 255), 2)
	objects = ct.update(rects)

	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			# check to see if the object has been counted or not
			if not to.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
				if direction < 0 and centroid[1] < 240:
					totalUp += 1
					to.counted = True

				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
				elif direction > 0 and (centroid[1] > 180):
					totalDown += 1
					to.counted = True

		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID + 1)
		cv2.putText(img_resized, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(img_resized, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
		#cv2.rectangle(img_resized, )

	info = [
		("Up", totalUp),
		("Down", totalDown),
		("Status", status),
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(img_resized, text, (10, 80 - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	cv2.imshow("mtcnn", img_resized)
	k = cv2.waitKey(1) & 0xff
	if k == 27 :
		break

	total_frame+=1
	fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
cv2.destroyAllWindows()
