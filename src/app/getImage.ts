"use client";

import { Tensor } from "onnxruntime-web/webgpu";
// import { Tensor } from "onnxruntime-web";

export function imageToTensor(img: HTMLImageElement, targetSize: number = 256): Tensor {
  // Create canvas to extract pixel data
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to get canvas context");
  }
  // Resize image while maintaining aspect ratio
  const { width, height } = img;
  const scale = Math.min(targetSize / width, targetSize / height);
  const newWidth = Math.round(width * scale);
  const newHeight = Math.round(height * scale);

  canvas.width = targetSize;
  canvas.height = targetSize;

  // Fill with black background
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, targetSize, targetSize);

  // Draw image centered
  const offsetX = (targetSize - newWidth) / 2;
  const offsetY = (targetSize - newHeight) / 2;
  ctx.drawImage(img, offsetX, offsetY, newWidth, newHeight);

  // Get pixel data
  const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
  const { data } = imageData;

  // Convert RGBA to grayscale and normalize to [0, 1]
  const tensorData = new Float32Array(targetSize * targetSize);

  for (let i = 0; i < targetSize * targetSize; i++) {
    const pixelIndex = i * 4; // RGBA

    // Convert to grayscale using luminance formula: 0.299*R + 0.587*G + 0.114*B
    const r = data[pixelIndex] / 255.0;
    const g = data[pixelIndex + 1] / 255.0;
    const b = data[pixelIndex + 2] / 255.0;
    const grayscale = 0.299 * r + 0.587 * g + 0.114 * b;

    tensorData[i] = grayscale;
  }

  // Create tensor with shape [1, 1, height, width] (NCHW format with 1 channel for grayscale)
  return new Tensor("float32", tensorData, [1, 1, targetSize, targetSize]);
}

export async function getExampleTensor() {
      const imageElement1 = await loadImageFromDataUrl("/img1.jpg")
      const imageTensor1 = imageToTensor(imageElement1);
      const imageElement2 = await loadImageFromDataUrl("/img2.jpg")
      const imageTensor2 = imageToTensor(imageElement2);


      const batchedData = new Float32Array(imageTensor1.data.length + imageTensor2.data.length);
      batchedData.set(imageTensor1.data as Float32Array, 0);
      batchedData.set(imageTensor2.data as Float32Array, imageTensor1.data.length);

      return new Tensor("float32", batchedData, [2, 1, 256, 256]);
}

export function loadImageFromDataUrl(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = dataUrl;
  });
}