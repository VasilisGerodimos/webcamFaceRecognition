const { cv, runVideoFaceRecognition } = require('./tools');
const fs = require('fs');
const path = require('path');

if (!cv.xmodules.face) {
   throw new Error('exiting: opencv4nodejs compiled without face module');
}

const basePath = './data/faceRecognition';
const imgsPath = path.resolve(basePath, 'trainImages');
const nameMappings = ['kostas', 'sofianos', 'voula', 'vasilis', 'vagos'];

const imgFiles = fs.readdirSync(imgsPath);

const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);
const getFaceImage = (grayImg) => {
   const faceRects = classifier.detectMultiScale(grayImg).objects;
   if (!faceRects.length) {
      throw new Error('failed to detect faces');
   }
   return grayImg.getRegion(faceRects[0]);
};

const images = imgFiles
   // get absolute file path
   .map(file => path.resolve(imgsPath, file))
   // read image
   .map(filePath => cv.imread(filePath))
   // face recognizer works with gray scale images
   .map(img => img.bgrToGray())
   // detect and extract face
   .map(getFaceImage)
   // face images must be equally sized
   .map(faceImg => faceImg.resize(80, 80));

const trainImages = images;

const labels = imgFiles.map(file => nameMappings.findIndex(name => file.includes(name)));

const runPrediction = (recognizer, img) => {
   const result = recognizer.predict(img);
   console.log('predicted: %s, confidence: %s', nameMappings[result.label], result.confidence);
   const prediction = {
      facePredicted: nameMappings[result.label],
      confidence: result.confidence
   }
   return prediction;
};

const lbph = new cv.LBPHFaceRecognizer();
lbph.train(trainImages, labels);

// ===================================================================

const webcamPort = 0;

function detectFaces(img) {
   // restrict minSize and scaleFactor for faster processing
   const options = {
      minSize: new cv.Size(100, 100),
      scaleFactor: 1.2,
      minNeighbors: 10
   };

   return classifier.detectMultiScale(img.bgrToGray(), options);
}

function getPrediction(img) {
   return runPrediction(lbph, img.bgrToGray().resize(80, 80))
}

runVideoFaceRecognition(webcamPort, detectFaces, getPrediction);