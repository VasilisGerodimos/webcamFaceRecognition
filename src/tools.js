const cv = require('opencv4nodejs');

exports.cv = cv;

// -------------- grabFrames ------------------------------
const grabFrames = (videoFile, delay, onFrame) => {
  const cap = new cv.VideoCapture(videoFile);
  let done = false;
  const intvl = setInterval(() => {
    let frame = cap.read();
    // loop back to start on end of stream reached
    if (frame.empty) {
      cap.reset();
      frame = cap.read();
    }
    onFrame(frame);

    const key = cv.waitKey(delay);
    done = key !== -1 && key !== 255;
    if (done) {
      clearInterval(intvl);
      console.log('Key pressed, exiting.');
    }
  }, 0);
};
exports.grabFrames = grabFrames;

// -------------- runVideoFaceRecognition ------------------------------
exports.runVideoFaceRecognition = (src, detectFaces, getPrediction) => grabFrames(src, 60, (frame) => {
  const confidenceLimit = 120;
  console.time('detection time');
  const frameResized = frame.resizeToMax(800);
  // detect faces
  const faceRects = detectFaces(frameResized);
  if (faceRects.objects.length) {
    // draw detection
    const minDetections = 10;
    faceRects.objects.forEach((faceRect, i) => {
      if (faceRects.numDetections[i] < minDetections) {
        return;
      }
      const faceImg = frameResized.getRegion(faceRect);
      const prediction = getPrediction(faceImg);
      let who = prediction.facePredicted;
      const confidence = prediction.confidence;
      if (confidence >= confidenceLimit) {
        who = "unknown";
      }
      const rect = cv.drawDetection(
        frameResized,
        faceRect,
        { color: new cv.Vec(255, 0, 0), segmentFraction: 4 }
      );

      const alpha = 0.4;
      cv.drawTextBox(
        frameResized,
        new cv.Point(rect.x, rect.y + rect.height + 10),
        [{ text: who }],
        alpha
      );
    });
  }

  cv.imshow('face detection', frameResized);
  console.timeEnd('detection time');
});